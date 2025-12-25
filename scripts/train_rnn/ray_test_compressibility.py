import shutil
from datetime import datetime
from pathlib import Path

import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
import lightning as L
from pathlib import Path
from sklearn.mixture import GaussianMixture
import secrets

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator


from hyper_lrrnn.training.trainer import RNNLightningModule
from hyper_lrrnn.training.utils import sample_dataset
from hyper_lrrnn.rnn.mixture_low_rank import MixtureLowRankRNN
from hyper_lrrnn.training.loss import lin_reg_r2, lin_reg_acc


root_dir = Path(__file__).parent


class CompressibilityCallback(L.Callback):
    """Callback to evaluate how well a LowRankRNN can be approximated by a MixtureLowRankRNN.

    Behavior:
    - On validation end, if the model exposes `m`, `n`, and `I` parameters, fit Gaussian mixture
      models to the concatenated parameter vectors and compute variance of fitted mixture means/covs
      across random subset fits.
    - Fit a GMM to the full parameter set, instantiate a `MixtureLowRankRNNWithReadout` from the
      fitted GMM parameters (means + cholesky of covariances) and repeatedly resample weights from
      that mixture to estimate mean/median/variance of R^2 and accuracy.

    Notes / assumptions:
    - Subset fraction for random-subset fits: 0.8 (configurable via constructor).
    - Number of subset resamples and number of resamples for model evaluation are configurable.
    """

    def __init__(self, n_mixtures=(1, 2, 3), n_subset_resamples=20, n_model_resamples=20, subset_frac=0.8, random_seed=None, log_frequency=1):
        super().__init__()
        self.n_mixtures = list(n_mixtures)
        self.n_subset_resamples = int(n_subset_resamples)
        self.n_model_resamples = int(n_model_resamples)
        self.subset_frac = float(subset_frac)
        self.rng = np.random.default_rng(random_seed or secrets.randbits(32))
        self.log_frequency = int(log_frequency)
    
    def _build_mixture_model(self, model, gmm):
        n_mix = gmm.n_components
        dim = gmm.means_.shape[1]

        # create a mixture model with same config as underlying model
        input_size = model.I.shape[-1]
        hidden_size = model.m.shape[0]
        rank = model.m.shape[1]
        mix_model = MixtureLowRankRNN(
            input_size, hidden_size, rank=rank,
            num_mixtures=n_mix, alpha=getattr(model, 'alpha', 0.1), activation='tanh',
            base_scale=hidden_size,
        )

        means = gmm.means_.reshape(n_mix, dim)
        covs = gmm.covariances_.reshape(n_mix, dim, dim)
        # compute cholesky for each component
        tril = np.zeros_like(covs)
        for i in range(n_mix):
            # numerical stability: add tiny diag jitter
            try:
                tril[i] = np.linalg.cholesky(covs[i])
            except np.linalg.LinAlgError:
                jitter = 1e-6 * np.eye(dim)
                tril[i] = np.linalg.cholesky(covs[i] + jitter)

        # set mixture parameters
        with torch.no_grad():
            mix_model.means.data = torch.tensor(means, dtype=torch.float32)
            mix_model.scale_tril.data = torch.tensor(tril, dtype=torch.float32)
        return mix_model
    
    def _eval_mixture_rnn(self, model, eval_loader, gmm, resamples=None):
        resamples = resamples or self.n_model_resamples or 1
        mix_model = self._build_mixture_model(model, gmm)
        r2s = []
        accs = []
        mix_model.eval()
        device = next(model.parameters()).device
        mix_model.to(device)
        for draw in range(resamples):
            total_r2 = 0.0
            total_acc = 0.0
            with torch.no_grad():
                for batch in eval_loader:
                    if isinstance(batch, dict):
                        inputs = batch['input'].to(device)
                        targets = batch['target'].to(device)
                    else:
                        inputs, targets = batch
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                    states = mix_model(inputs)
                    # pl_module exposes r2_fn and acc_fn that accept (output, target)
                    r2 = lin_reg_r2(states, targets)
                    acc = lin_reg_acc(states, targets)
                    total_r2 += float(r2)
                    total_acc += float(acc) if not (isinstance(acc, float) and np.isnan(acc)) else 0.0
                    n_batches += 1
            r2s.append(total_r2 / n_batches)
            accs.append(total_acc / n_batches)
        return r2s, accs

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_frequency != 0:
            return
        eval_loader = trainer.val_dataloaders

        model = pl_module.model
        # only run for low-rank models exposing m,n,I
        if not (hasattr(model, 'm') and hasattr(model, 'n') and hasattr(model, 'I')):
            return

        m = model.m.detach().cpu().numpy()
        n = model.n.detach().cpu().numpy()
        I = model.I.detach().cpu().numpy()
        samples = np.concatenate([m, n, I], axis=-1)  # shape (N, dim)
        N, dim = samples.shape

        # 1) Variance of mixtures fit to random subsets
        for n_mix in self.n_mixtures:
            subset_means = []
            subset_covs = []
            subset_r2s = []
            subset_accs = []
            for r in range(self.n_subset_resamples):
                k = max(2, int(self.subset_frac * N))
                idx = self.rng.choice(N, size=k, replace=False)
                subset = samples[idx]
                try:
                    gm = GaussianMixture(n_components=n_mix, covariance_type='full', n_init=3, random_state=int(self.rng.integers(0, 2**31-1)))
                    gm.fit(subset)
                except Exception as e:
                    # if fit fails, skip this resample
                    continue
                # reorder components by first coordinate of their means for simple alignment
                order = np.argsort(gm.means_[:, 0])
                subset_means.append(gm.means_[order])
                subset_covs.append(gm.covariances_[order][:, order])
                r2s, accs = self._eval_mixture_rnn(model, eval_loader, gm, resamples=1)
                subset_r2s.append(np.mean(r2s))
                subset_accs.append(np.mean(accs))

            if len(subset_means) >= 2:
                means_stack = np.stack(subset_means, axis=0).reshape(len(subset_means), -1)  # (R, n_mix * dim)
                var_of_means = np.var(means_stack, axis=0)
                covs_stack = np.stack(subset_covs, axis=0).reshape(len(subset_covs), -1)  # (R, n_mix, dim, dim)
                var_of_covs = np.var(covs_stack, axis=0)
                # aggregate statistics
                stats = {
                    f'mixture_mean_var_mean_{n_mix}': float(np.mean(var_of_means)),
                }
                # covariance variability: use Frobenius norm of covariance matrices
                stats.update({
                    f'mixture_cov_var_mean_{n_mix}': float(np.mean(var_of_covs)),
                })
                stats.update({
                    f'mixture_subset_r2_max_{n_mix}': float(np.max(subset_r2s)),
                    f'mixture_subset_r2_mean_{n_mix}': float(np.mean(subset_r2s)),
                    f'mixture_subset_r2_var_{n_mix}': float(np.var(subset_r2s)),
                    f'mixture_subset_acc_max_{n_mix}': float(np.max(subset_accs)),
                    f'mixture_subset_acc_mean_{n_mix}': float(np.mean(subset_accs)),
                    f'mixture_subset_acc_var_{n_mix}': float(np.var(subset_accs)),
                })
                trainer.logger.log_metrics(stats, step=trainer.global_step)

            # 2) Fit to full samples and evaluate resampled model performance
            try:
                gm_full = GaussianMixture(n_components=n_mix, covariance_type='full', n_init=5, random_state=int(self.rng.integers(0, 2**31-1)))
                gm_full.fit(samples)
            except Exception as e:
                continue

            # Evaluate model performance across several resamples
            r2s, accs = self._eval_mixture_rnn(model, eval_loader, gm_full, resamples=self.n_model_resamples)

            if len(r2s) > 0:
                r2s = np.array(r2s)
                accs = np.array(accs)
                stats = {
                    f'mixture_resample_r2_max_{n_mix}': float(np.nanmax(r2s)),
                    f'mixture_resample_r2_mean_{n_mix}': float(np.nanmean(r2s)),
                    f'mixture_resample_r2_var_{n_mix}': float(np.nanvar(r2s)),
                    f'mixture_resample_acc_max_{n_mix}': float(np.nanmax(accs)),
                    f'mixture_resample_acc_mean_{n_mix}': float(np.nanmean(accs)),
                    f'mixture_resample_acc_var_{n_mix}': float(np.nanvar(accs)),
                }
                trainer.logger.log_metrics(stats, step=trainer.global_step)


# ---------- OPTIONS -----------
PROJECT_STR = "lfads-torch-example"
DATASET_STR = "nlb_mc_maze"
RUN_TAG = datetime.now().strftime("%y%m%d") + "_exampleMulti"
RUN_DIR = Path("/snel/share/runs") / PROJECT_STR / DATASET_STR / RUN_TAG
# ------------------------------

# Set the mandatory config overrides to select datamodule and model
mandatory_overrides = {
    "datamodule": DATASET_STR,
    "model": DATASET_STR,
    "logger.wandb_logger.project": PROJECT_STR,
    "logger.wandb_logger.tags.1": DATASET_STR,
    "logger.wandb_logger.tags.2": RUN_TAG,
}
RUN_DIR.mkdir(parents=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Run the hyperparameter search
tune.run(
    tune.with_parameters(
        run_model,
        config_path="../configs/multi.yaml",
    ),
    metric="valid/recon_smth",
    mode="min",
    name=RUN_DIR.name,
    config={
        **mandatory_overrides,
        "model.dropout_rate": tune.uniform(0.0, 0.6),
        "model.kl_co_scale": tune.loguniform(1e-6, 1e-4),
        "model.kl_ic_scale": tune.loguniform(1e-6, 1e-3),
        "model.l2_gen_scale": tune.loguniform(1e-4, 1e0),
        "model.l2_con_scale": tune.loguniform(1e-4, 1e0),
    },
    resources_per_trial=dict(cpu=3, gpu=0.5),
    num_samples=60,
    local_dir=RUN_DIR.parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=FIFOScheduler(),
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=["valid/recon_smth", "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)
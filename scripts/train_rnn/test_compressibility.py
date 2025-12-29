import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
import lightning as L
from pathlib import Path
from sklearn.mixture import GaussianMixture
import secrets

from hyper_lrrnn.training.trainer import RNNLightningModule
from hyper_lrrnn.training.utils import sample_dataset
from hyper_lrrnn.rnn.mixture_low_rank import MixtureLowRankRNN
from hyper_lrrnn.training.loss import lin_reg_r2, lin_reg_acc
from hyper_lrrnn.training.regularizer import Regularizer


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
            mix_model.mixture_weights.data = torch.tensor(gmm.weights_, dtype=torch.float32)
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
            n_batches = 0
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

@hydra.main(config_path="config", config_name="compress", version_base="1.1")
def main(cfg):
    task = hydra.utils.instantiate(cfg.task)
    train_dataset, test_dataset = sample_dataset(task, **cfg.dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    inputs, targets = next(iter(train_loader))
    input_size = inputs.shape[-1]
    output_size = targets.shape[-1]

    if hasattr(cfg, 'regularizer') and cfg.regularizer is not None:
        regularizer = Regularizer(**cfg.regularizer)
    else:
        regularizer = None

    model = hydra.utils.instantiate(cfg.model)
    model = model(input_size=input_size, output_size=output_size)
    model = RNNLightningModule(model=model, regularizer=regularizer, **cfg.optimizer)

    logger = hydra.utils.instantiate(cfg.logger)
    if callable(logger):
        logger = logger(config=OmegaConf.to_object(cfg))
    # create compressibility callback and wire into trainer
    compress_cb = CompressibilityCallback(n_mixtures=(1, 2), n_subset_resamples=20, n_model_resamples=20, random_seed=0, log_frequency=20)
    trainer = L.Trainer(**cfg.trainer, logger=logger, callbacks=[compress_cb])
    trainer.fit(model, train_loader, test_loader)
    # r2 = trainer.progress_bar_metrics.get('val_r2', np.nan)
    # acc = trainer.progress_bar_metrics.get('val_acc', np.nan)

    model_params = model.model.state_dict()
    task_signal = cfg.task.signal
    task_target = cfg.task.target
    torch.save(model_params, f"{task_signal}_{task_target}_temp.ckpt")


if __name__ == "__main__":
    main()

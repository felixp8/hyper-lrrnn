import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import itertools
import zuko

from pathlib import Path

from hyper_lrrnn.rnn.mixture_low_rank import MixtureLowRankRNN
from hyper_lrrnn.training.loss import lin_reg_r2, lin_reg_acc
from hyper_lrrnn.gen.multi_context_spline import MultiContextNSF
from hyper_lrrnn.gen.mmd import compute_rbf_mmd

use_gpu = torch.cuda.is_available()


# load fonts/rc file for plotting
font_path = Path(__file__).parent.parent.parent / "fonts/Arial.ttf"
matplotlib.font_manager.fontManager.addfont(font_path)

rc_path = Path(__file__).parent.parent.parent / ".matplotlibrc"


# load datasets
dataset_dir = Path(__file__).parent.parent / "approx_rnn" / "datasets"
mixture_dir = Path(__file__).parent.parent / "approx_rnn" / "mixture_params"


# define tasks
task_signals = ["mean", "integral"]
task_targets = ["1", "2", "sum", "diff"]
n_mixtures = 1
heldout_tasks = [("integral", "1")]

base_task_set = "".join([s[0] for s in task_signals])
exp_type = "all" if len(heldout_tasks) == 0 else "heldout"


# set up
sig = "integral"
tgt = "1"

sample_sig = sig
sample_tgt = tgt

data = np.load(mixture_dir / f"{sig}_{tgt}_mixture{n_mixtures}.npz")
params = data["params"]
model_accs = data["model_accs"]
random_accs = data["random_accs"]

task_dataset = np.load(dataset_dir / f"{sig}_{tgt}.npz")


# define model
class TaskConditionedFlow(nn.Module):
    def __init__(self, params_dim, context_dim, bins=16, transforms=4, hidden_features=[128] * 3, multi_context=False):
        super(TaskConditionedFlow, self).__init__()
        self.params_dim = params_dim
        self.context_dim = context_dim
        self.bins = bins
        self.transforms = transforms
        self.hidden_features = hidden_features
        self.multi_context = multi_context
        
        self.sig_embed = nn.Embedding(num_embeddings=len(task_signals), embedding_dim=context_dim)
        self.tgt_embed = nn.Embedding(num_embeddings=len(task_targets), embedding_dim=context_dim)
        
        if self.multi_context:
            self.flow = MultiContextNSF(params_dim, context=context_dim*2, bins=bins, transforms=transforms, hidden_features=hidden_features)
        else:
            self.flow = zuko.flows.NSF(params_dim, context=context_dim*2, bins=bins, transforms=transforms, hidden_features=hidden_features)

    def forward(self, sig, tgt):
        sig_emb = self.sig_embed(sig)
        tgt_emb = self.tgt_embed(tgt)
        context = torch.cat([sig_emb, tgt_emb], dim=-1)
        return self.flow(context)
    
sample_sizes = [50, 100, 150, 200]
run_dir_50 = Path(f"../gen_rnn/{base_task_set}_{n_mixtures}_{exp_type}_50/")
run_dir_100 = Path(f"../gen_rnn/{base_task_set}_{n_mixtures}_{exp_type}_100/")
run_dir_150 = Path(f"../gen_rnn/{base_task_set}_{n_mixtures}_{exp_type}_150/")
standard_dir = Path(f"../gen_rnn/{base_task_set}_{n_mixtures}_{exp_type}/")


# plot MMDs
params = torch.from_numpy(params).to(torch.float)
n_bootstrap = 50
bandwidth = 0.8 if n_mixtures == 2 else 0.25
mmd_baseline = []
for _ in range(n_bootstrap):
    idx = torch.randperm(len(params))
    params_1 = params[idx[:len(params) // 2]]
    params_2 = params[idx[len(params) // 2:]]
    mmd_baseline.append(compute_rbf_mmd(params_1, params_2, bandwidth=bandwidth).item())
mmd_results = {"baseline": mmd_baseline}
mmd_best_checkpoints = {}

for model_key, model_dir in zip(sample_sizes, [run_dir_50, run_dir_100, run_dir_150, standard_dir]):
    checkpoint_dir = model_dir / "checkpoints"
    mmds_all = []
    best_mmds = None
    for checkpoint in sorted(checkpoint_dir.glob("*.pt"))[-1:]:
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        flow = TaskConditionedFlow(
            params_dim=params.shape[-1],
            context_dim=16,
            bins=16,
            transforms=6,
            hidden_features=[256] * 3,
            multi_context=False,
        )
        flow.load_state_dict(checkpoint_data)

        flow.eval()
        if use_gpu:
            flow = flow.cuda()
        with torch.no_grad():
            si = torch.tensor(task_signals.index(sample_sig)).to("cuda:0" if use_gpu else "cpu")
            ti = torch.tensor(task_targets.index(sample_tgt)).to("cuda:0" if use_gpu else "cpu")
            chkpt_mmd = []
            for _ in range(n_bootstrap):
                samples = flow(si, ti).sample((200,))

                chkpt_mmd.append(compute_rbf_mmd(params, samples.detach().cpu(), bandwidth=bandwidth).item())
            print(f"Model: {model_key}, Checkpoint: {checkpoint.name}, MMD: {np.median(chkpt_mmd)}")
            if len(mmds_all) == 0 or np.median(chkpt_mmd) < np.median(mmd_baseline):
                best_mmds = chkpt_mmd
            mmds_all.append(np.median(chkpt_mmd))
    assert best_mmds is not None
    mmd_results[model_key] = best_mmds
    mmd_best_checkpoints[model_key] = sorted(checkpoint_dir.glob("*.pt"))[np.argmin(mmds_all)].name

with matplotlib.rc_context(fname=rc_path):
    fig, axs = plt.subplots(1, 1, figsize=(4, 2))
    sns.violinplot(data=list(mmd_results.values()), fill=False)
    axs.set_xticks(range(len(mmd_results)))
    axs.set_xticklabels(list(mmd_results.keys()))
    plt.ylabel("MMD")
    plt.savefig("mmd_by_scale.png")
    plt.close()


# eval util
rank = 3
input_size = task_dataset["inputs"].shape[-1]
dim = 2 * rank + input_size
def eval_model(params, sig, tgt):
    means = params[:dim * n_mixtures]
    means = means.reshape(n_mixtures, dim)

    tril = params[dim * n_mixtures:]
    tril = tril.reshape(n_mixtures, -1)
    tril_full = np.zeros((n_mixtures, dim, dim))
    for i in range(n_mixtures):
        j, k = np.tril_indices(dim)
        tril_full[i, j, k] = tril[i]

    model = MixtureLowRankRNN(input_size, 500, rank=rank, num_mixtures=n_mixtures, alpha=0.1, activation="tanh")
    model.means.data = torch.tensor(means).to(torch.float)
    model.scale_tril.data = torch.tensor(tril_full).to(torch.float)

    model.eval()
    if use_gpu:
        model = model.cuda()
    temp_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(task_dataset["inputs"][:600]).to(torch.float),
            torch.tensor(task_dataset["targets"][:600]).to(torch.float),
        ),
        batch_size=200,
    )
    with torch.no_grad():
        total_metric = 0
        for batch in temp_loader:
            inputs, targets = batch
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
            states = model(inputs)
            total_metric += lin_reg_acc(states, targets, threshold=0.1, window=10)
    return (total_metric / len(temp_loader))


# plot task acc
acc_results = {"baseline": model_accs}
acc_best_checkpoints = {}

for model_key, model_dir in zip(sample_sizes, [run_dir_50, run_dir_100, run_dir_150, standard_dir]):
    checkpoint_dir = model_dir / "checkpoints"
    median_scores_all = []
    best_scores = None
    for checkpoint in sorted(checkpoint_dir.glob("*.pt"))[-1:]:
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        flow = TaskConditionedFlow(
            params_dim=params.shape[-1],
            context_dim=16,
            bins=16,
            transforms=6,
            hidden_features=[256] * 3,
            multi_context=False,
        )
        flow.load_state_dict(checkpoint_data)

        flow.eval()
        if use_gpu:
            flow = flow.cuda()
        with torch.no_grad():
            si = torch.tensor(task_signals.index(sample_sig)).to("cuda:0" if use_gpu else "cpu")
            ti = torch.tensor(task_targets.index(sample_tgt)).to("cuda:0" if use_gpu else "cpu")
            samples = flow(si, ti).sample((100,))
            samples = samples.detach().cpu().numpy()

            scores = []
            for p in samples:
                scores.append(eval_model(p, sig, tgt))
            if len(median_scores_all) == 0 or np.median(scores) > np.max(median_scores_all):
                best_scores = scores
            median_scores_all.append(np.median(scores))  
            print(f"Model: {model_key}, Checkpoint: {checkpoint.name}, Acc: {np.median(scores)}")
    acc_results[model_key] = best_scores
    acc_best_checkpoints[model_key] = sorted(checkpoint_dir.glob("*.pt"))[np.argmin(median_scores_all)].name

with matplotlib.rc_context(fname=rc_path):
    fig, axs = plt.subplots(1, 1, figsize=(4, 2))
    sns.violinplot(data=list(acc_results.values()), fill=False)
    axs.axhline(np.median(random_accs), color="black", linestyle="--")
    axs.set_xticks(range(len(acc_results)))
    axs.set_xticklabels(list(acc_results.keys()))
    axs.set_ylabel("accuracy")
    axs.set_ylim(0.3, 1.1)
    plt.tight_layout()
    plt.savefig(f'acc_by_scale.png')

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

use_gpu = torch.cuda.is_available()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n-epochs", type=int, default=5000)
parser.add_argument("--heldout", action="store_true")
parser.add_argument("--include-time", action="store_true")
parser.add_argument("--num-mixtures", type=int, default=2)
parser.add_argument("--multi-context", action="store_true")
parser.add_argument("--samples-per-task", type=int, default=50)
args = parser.parse_args()


# load fonts/rc file for plotting
font_path = Path(__file__).parent.parent.parent / "fonts/Arial.ttf"
matplotlib.font_manager.fontManager.addfont(font_path)

rc_path = Path(__file__).parent.parent.parent / ".matplotlibrc"


# load datasets
dataset_dir = Path(__file__).parent.parent / "approx_rnn" / "datasets"
mixture_dir = Path(__file__).parent.parent / "approx_rnn" / "mixture_params"


# define tasks
task_signals = ["mean", "integral", "time"] if args.include_time else ["mean", "integral"]
task_targets = ["1", "2", "sum", "diff"]
n_mixtures = args.num_mixtures
heldout_tasks = [("integral", "1")] if args.heldout else []

base_task_set = "".join([s[0] for s in task_signals])
exp_type = "all" if len(heldout_tasks) == 0 else "heldout"
exp_type += "_mc" if args.multi_context else ""
run_dir = Path(f"./{base_task_set}_{n_mixtures}_{exp_type}_{args.samples_per_task}/")
run_dir.mkdir(exist_ok=True)


# load data
params_all = []
tasks_all = []
accs_all = []
r2s_all = []
task_keys = []
random_accs_all = []

for sig, tgt in itertools.product(task_signals, task_targets):
    task_dataset = np.load(dataset_dir / f"{sig}_{tgt}.npz")
    params = np.load(mixture_dir / f"{sig}_{tgt}_mixture{n_mixtures}.npz")
    params_all.append(params["params"])
    accs_all.append(params["model_accs"])
    r2s_all.append(params["model_r2s"])
    random_accs_all.append(params["random_accs"])
    tasks_all.append(task_dataset)
    task_keys.append((sig, tgt))


# build param dataset
sig_idx = np.concatenate([
    np.ones(p.shape[0]) * task_signals.index(key[0]) 
    for key, p in zip(task_keys, params_all)
])
tgt_idx = np.concatenate([
    np.ones(p.shape[0]) * task_targets.index(key[1])
    for key, p in zip(task_keys, params_all)
])
params_stack = np.concatenate(params_all, axis=0)

heldout_mask = np.zeros(params_stack.shape[0], dtype=np.bool_)
for sig, tgt in task_keys:
    if (sig, tgt) in heldout_tasks:
        heldout_mask = heldout_mask | ((sig_idx == task_signals.index(sig)) & (tgt_idx == task_targets.index(tgt)))
    else:
        match = (sig_idx == task_signals.index(sig)) & (tgt_idx == task_targets.index(tgt))
        count = np.cumsum(match)
        heldout_mask = heldout_mask | (match & (count > args.samples_per_task))

assert sig_idx.shape[0] == tgt_idx.shape[0]
assert params_stack.shape[0] == sig_idx.shape[0]

dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(params_stack[~heldout_mask]).to(torch.float),
    torch.from_numpy(sig_idx[~heldout_mask].astype(int)),
    torch.from_numpy(tgt_idx[~heldout_mask].astype(int))
)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# dim red prep
pca = PCA()
pcs = pca.fit_transform(params_stack)
print(np.cumsum(pca.explained_variance_ratio_)[:10])

with matplotlib.rc_context(fname=rc_path):
    colors = sig_idx * len(task_targets) + tgt_idx
    colors /= colors.max()
    fig = plt.figure(figsize=(2, 2))
    plt.scatter(pcs[:, 0], pcs[:, 1], c=colors, cmap="tab20", s=1)
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=plt.get_cmap("tab20")((sig_i * len(task_targets) + tgt_i) / (len(task_signals) * len(task_targets) - 1)), 
                                label=f"{task_signals[sig_i]}_{task_targets[tgt_i]}",
                                markersize=5)
        for sig_i, tgt_i in itertools.product(range(len(task_signals)), range(len(task_targets)))
    ]
    legend = plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.8, 1.0))
    base_pc_xlim = plt.xlim()
    base_pc_ylim = plt.ylim()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(run_dir / 'pca.png')
    plt.show()

base_plot_xlim = base_pc_xlim[0] - 0.1, base_pc_xlim[1] + 0.1
base_plot_ylim = base_pc_ylim[0] - 0.08, base_pc_ylim[1] + 0.08


# eval util
rank = 3
input_size = tasks_all[0]["inputs"].shape[-1]
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
    task_dataset = tasks_all[task_keys.index((sig, tgt))]
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


# set up model
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


flow = TaskConditionedFlow(
    params_dim=params_stack.shape[-1],
    context_dim=16,
    bins=16,
    transforms=6,
    hidden_features=[256] * 3,
    multi_context=args.multi_context,
)
if use_gpu:
    flow = flow.cuda()

optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-3)


# train
n_epochs = args.n_epochs
plot_every_n_epochs = 50


for epoch in range(n_epochs):
    flow.train()

    for x, sig, tgt in train_dataloader:
        if use_gpu:
            x = x.cuda()
            sig = sig.cuda()
            tgt = tgt.cuda()
        loss = -flow(sig, tgt).log_prob(x)  # -log p(x | c)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss = 0
    for x, sig, tgt in test_dataloader:
        if use_gpu:
            x = x.cuda()
            sig = sig.cuda()
            tgt = tgt.cuda()
        loss = -flow(sig, tgt).log_prob(x)  # -log p(x | c)
        total_loss += loss.mean()

    print(f"Epoch {epoch}: Test Loss: {total_loss.item() / len(test_dataloader)}")

    if epoch % plot_every_n_epochs == 0:
        Path(run_dir / "checkpoints").mkdir(exist_ok=True)
        torch.save(flow.state_dict(), run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")

        for sig, tgt in itertools.product(task_signals, task_targets):
            Path(run_dir / f"{sig}_{tgt}").mkdir(exist_ok=True)
            with torch.no_grad():
                si = torch.tensor(task_signals.index(sig)).to("cuda:0" if use_gpu else "cpu")
                ti = torch.tensor(task_targets.index(tgt)).to("cuda:0" if use_gpu else "cpu")
                samples = flow(si, ti).sample((50,))
            samples = samples.detach().cpu().numpy()

            with matplotlib.rc_context(fname=rc_path):
                fig, axs = plt.subplots(1, 2, figsize=(4, 2))
                pcs_samp = pca.transform(samples)
                mask = (sig_idx == task_signals.index(sig)) & (tgt_idx == task_targets.index(tgt))
                axs[0].scatter(pcs[:, 0][mask], pcs[:, 1][mask], color="tab:blue", s=1)
                axs[0].scatter(pcs_samp[:, 0], pcs_samp[:, 1], color="tab:red", s=1)
                axs[0].set_xlabel("PC1")
                axs[0].set_ylabel("PC2")
                axs[0].set_xlim(*base_plot_xlim)
                axs[0].set_ylim(*base_plot_ylim)

                scores = []
                for p in samples:
                    scores.append(eval_model(p, sig, tgt))
                axs[1].hist(scores, bins=np.linspace(0, 1, 20))
                axs[1].set_xlabel("accuracy")
                axs[1].set_ylabel("count")
                axs[1].set_ylim(0, 50)
                orig_median = np.median(accs_all[task_keys.index((sig, tgt))])
                axs[1].axvline(orig_median, color="black", linestyle="--")
                random_median = np.median(random_accs_all[task_keys.index((sig, tgt))])
                axs[1].axvline(random_median, color="red", linestyle="--")

                plt.tight_layout()
                plt.savefig(run_dir / f"{sig}_{tgt}" / f"epoch_{epoch:04d}.png")
                plt.close()

            # Path(run_dir / "results").mkdir(exist_ok=True)
            # np.savez(run_dir / "results" / f"{sig}_{tgt}_epoch_{epoch:04d}.npz", samples=samples, scores=np.array(scores))
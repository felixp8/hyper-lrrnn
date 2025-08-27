import math
import numpy as np
import pandas as pd
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
n_mixtures = 2
heldout_tasks = [("integral", "1")]

base_task_set = "".join([s[0] for s in task_signals])
exp_type = "all" if len(heldout_tasks) == 0 else "heldout"


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
    

standard_dir = Path(f"../gen_rnn/{base_task_set}_{n_mixtures}_{exp_type}/")
mc_dir = Path(f"../gen_rnn/{base_task_set}_{n_mixtures}_{exp_type}_mc/")


# plot MMDs
mmd_results = []
mmd_baseline = {}
for model_key, model_dir in zip(["base", "structured"], [standard_dir, mc_dir]):
    checkpoint_dir = model_dir / "checkpoints"

    checkpoint = sorted(checkpoint_dir.glob("*.pt"))[-1]
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    flow = TaskConditionedFlow(
        params_dim=params_all[0].shape[-1],
        context_dim=16,
        bins=16,
        transforms=6,
        hidden_features=[256] * 3,
        multi_context=(model_key == "structured")
    )
    flow.load_state_dict(checkpoint_data)

    flow.eval()
    if use_gpu:
        flow = flow.cuda()

    for sig, tgt in task_keys:
        if (sig, tgt) in heldout_tasks:
            continue
        print(f"{model_key}, {sig}, {tgt}")
        params = params_all[task_keys.index((sig, tgt))]
        params = torch.from_numpy(params).to(torch.float)
        n_bootstrap = 20
        bandwidth = 0.8
        # if mmd_baseline.get(f"{sig}_{tgt}") is None:
        #     idx = torch.randperm(len(params))
        #     params_1 = params[idx[:len(params) // 2]]
        #     params_2 = params[idx[len(params) // 2:]]
        #     mmd_baseline[f"{sig}_{tgt}"] = compute_rbf_mmd(params_1, params_2, bandwidth=bandwidth).item()
        with torch.no_grad():
            si = torch.tensor(task_signals.index(sig)).to("cuda:0" if use_gpu else "cpu")
            ti = torch.tensor(task_targets.index(tgt)).to("cuda:0" if use_gpu else "cpu")
            chkpt_mmd = []
            for _ in range(n_bootstrap):
                samples = flow(si, ti).sample((200,))

                mmd_results.append({
                    "model": model_key,
                    "task": f"{sig}_{tgt}",
                    "mmd": compute_rbf_mmd(params, samples.detach().cpu(), bandwidth=bandwidth).item()
                })
mmd_results = pd.DataFrame(mmd_results)

with matplotlib.rc_context(fname=rc_path):
    fig, axs = plt.subplots(1, 1, figsize=(6, 2))
    sns.violinplot(data=mmd_results, x="task", y="mmd", hue="model", split=True, inner="quart", fill=False, ax=axs)
    # i = 0
    # for (sig, tgt) in task_keys:
    #     if (sig, tgt) in heldout_tasks:
    #         continue
    #     axs.plot([i - 0.3, i + 0.3], [mmd_baseline[f"{sig}_{tgt}"]] * 2, color="k", linestyle="--")
    #     i += 1
    # axs.set_xticks(range(len(mmd_results)))
    # axs.set_xticklabels(list(mmd_results["task"]))
    plt.ylabel("MMD")
    plt.savefig("mmd_by_task.png")
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


# plot task acc
acc_model_baseline = {}
acc_random_baseline = {}
acc_results = []

for model_key, model_dir in zip(["base", "structured"], [standard_dir, mc_dir]):
    checkpoint_dir = model_dir / "checkpoints"
    median_scores_all = []
    best_scores = None
    checkpoint = sorted(checkpoint_dir.glob("*.pt"))[-1]
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    flow = TaskConditionedFlow(
        params_dim=params_all[0].shape[-1],
        context_dim=16,
        bins=16,
        transforms=6,
        hidden_features=[256] * 3,
        multi_context=(model_key == "structured")
    )
    flow.load_state_dict(checkpoint_data)

    flow.eval()
    if use_gpu:
        flow = flow.cuda()
    
    for sig, tgt in task_keys:
        if (sig, tgt) in heldout_tasks:
            continue
        print(f"{model_key}, {sig}, {tgt}")
        acc_model_baseline[f"{sig}_{tgt}"] = accs_all[task_keys.index((sig, tgt))]
        acc_random_baseline[f"{sig}_{tgt}"] = random_accs_all[task_keys.index((sig, tgt))]
        with torch.no_grad():
            si = torch.tensor(task_signals.index(sig)).to("cuda:0" if use_gpu else "cpu")
            ti = torch.tensor(task_targets.index(tgt)).to("cuda:0" if use_gpu else "cpu")
            samples = flow(si, ti).sample((50,))
            samples = samples.detach().cpu().numpy()

            scores = []
            for p in samples:
                acc_results.append({
                    "model": model_key,
                    "task": f"{sig}_{tgt}",
                    "accuracy": eval_model(p, sig, tgt)
                })
acc_results = pd.DataFrame(acc_results)

with matplotlib.rc_context(fname=rc_path):
    fig, axs = plt.subplots(1, 1, figsize=(6, 2))
    sns.violinplot(data=acc_results, x="task", y="accuracy", hue="model", split=True, inner="quart", fill=False, ax=axs)
    xlim = axs.get_xlim()
    i = 0
    for (sig, tgt) in task_keys:
        if (sig, tgt) in heldout_tasks:
            continue
        axs.plot([i - 0.3, i + 0.3], [np.median(acc_random_baseline[f"{sig}_{tgt}"])] * 2, color="black", linestyle="--")
        axs.plot([i - 0.3, i + 0.3], [np.median(acc_model_baseline[f"{sig}_{tgt}"])] * 2, color="red", linestyle="--")
        i += 1
    axs.set_ylabel("accuracy")
    axs.set_ylim(0.3, 1.1)
    axs.set_xlim(*xlim)
    plt.tight_layout()
    plt.savefig(f'acc_by_task.png')

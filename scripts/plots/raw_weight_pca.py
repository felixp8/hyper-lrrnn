import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import itertools

from pathlib import Path


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

# load model checkpoints
rank = 3
weights = []
colors = []
for i, (task_signal, task_target) in enumerate(itertools.product(task_signals, task_targets)):
    model_path = Path(f"../train_rnn/checkpoints/{task_signal}_{task_target}")
    model_ckpts = sorted(model_path.glob("*.ckpt"))
    for checkpoint_path in model_ckpts:
        chkpt = torch.load(checkpoint_path, map_location="cpu")
        m = chkpt['m']
        n = chkpt['n']
        I = chkpt['I']
        assert m.shape[-1] == rank
        samples = torch.cat([m, n, I], dim=-1).detach().numpy().flatten()
        weights.append(samples)
        colors.append(i)

weights = np.stack(weights, axis=0)
colors = np.array(colors, dtype=np.float32)


# dim red prep
pca = PCA()
pcs = pca.fit_transform(weights)
print(np.cumsum(pca.explained_variance_ratio_)[:10])

with matplotlib.rc_context(fname=rc_path):
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
    plt.savefig('pca.png')

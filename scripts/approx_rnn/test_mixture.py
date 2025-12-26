import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
import torch
import torch.nn as nn

from pathlib import Path


from hyper_lrrnn.rnn.mixture_low_rank import MixtureLowRankRNN
from hyper_lrrnn.training.loss import lin_reg_r2, lin_reg_acc


Path("mixture_params").mkdir(exist_ok=True)


import argparse
parser = argparse.ArgumentParser(description="Evaluate mixture model parameters.")
parser.add_argument("--task_signal", type=str, default="mean")
parser.add_argument("--task_target", type=str, default="1")
parser.add_argument("--n_mixtures", type=int, default=1)
args = parser.parse_args()


# load dataset
task_signal = args.task_signal
task_target = args.task_target

dataset_path = Path(f"datasets/{task_signal}_{task_target}.npz")
dataset = np.load(dataset_path)

input_size = dataset["inputs"].shape[-1]


# load model checkpoints
rank = 5
model_path = Path(f"../train_rnn/checkpoints/{task_signal}_{task_target}")
model_ckpts = sorted(model_path.glob("*.ckpt"))

n_mixtures = args.n_mixtures
params = []
orig_r2s = []
orig_accs = []

for checkpoint_path in model_ckpts:
    r2 = float(checkpoint_path.stem.split("r2=")[1].split("-acc")[0])
    acc = float(checkpoint_path.stem.split("acc=")[1].split(".ckpt")[0])
    # if acc < 0.55:
    #     continue
    orig_r2s.append(r2)
    orig_accs.append(acc)

    # import pdb; pdb.set_trace()

    chkpt = torch.load(checkpoint_path, map_location="cpu")
    m = chkpt['m']
    n = chkpt['n']
    I = chkpt['I']
    assert m.shape[-1] == rank
    samples = torch.cat([m, n, I], dim=-1).detach().numpy()
    gm = GaussianMixture(n_mixtures, random_state=0)
    gm.fit(samples)

    weights = gm.weights_
    means = gm.means_
    covariances = gm.covariances_

    # enforce canonical ordering
    msort = np.argsort(np.max(means[:, :rank], axis=0))
    nsort = msort
    isort = np.arange(means[0, (2*rank):].shape[0])
    perm = np.concatenate([msort, nsort + rank, isort + 2*rank])
    means = means[:, perm]
    covariances = covariances[:, perm, :][:, :, perm]
    mixture_sort = np.argsort(means[:, 0])
    means = means[mixture_sort]
    covariances = covariances[mixture_sort]
    weights = weights[mixture_sort]

    tril = np.linalg.cholesky(covariances)
    i1, i2 = np.tril_indices(samples.shape[-1])
    if tril.ndim == 2:
        tril = tril[i1, i2]
    else:
        tril = tril[:, i1, i2]
    params.append(np.concatenate([means.flatten(), tril.flatten(), weights.flatten()]))

params = np.stack(params, axis=0)
orig_r2s = np.array(orig_r2s)
orig_accs = np.array(orig_accs)


# evaluate mixture models

temp_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.tensor(dataset["inputs"][:1000]).to(torch.float),
        torch.tensor(dataset["targets"][:1000]).to(torch.float),
    ),
    batch_size=200,
)

random_r2s = []
random_accs = []
model_r2s = []
model_accs = []
dim = 2 * rank + input_size

for param in params:
    means = param[:dim * n_mixtures]
    means = means.reshape(n_mixtures, dim)

    tril = param[dim * n_mixtures:-n_mixtures]
    tril = tril.reshape(n_mixtures, -1)
    tril_full = np.zeros((n_mixtures, dim, dim))
    for i in range(n_mixtures):
        j, k = np.tril_indices(dim)
        tril_full[i, j, k] = tril[i]

    weights = param[-n_mixtures:]

    model = MixtureLowRankRNN(input_size, 500, rank=rank, num_mixtures=n_mixtures, alpha=0.1, activation="tanh")

    model.eval()
    with torch.no_grad():
        total_r2 = 0
        total_acc = 0
        for batch in temp_loader:
            inputs, targets = batch
            states = model(inputs)
            total_r2 += lin_reg_r2(states, targets)
            total_acc += lin_reg_acc(states, targets, threshold=0.1, window=10)
    random_r2s.append(total_r2 / len(temp_loader))
    random_accs.append(total_acc / len(temp_loader))


    assert model.means.shape == means.shape
    assert model.scale_tril.shape == tril_full.shape
    assert model.mixture_weights.shape[0] == weights.shape[0]
    with torch.no_grad():
        model.means.data = torch.tensor(means).to(torch.float)
        model.scale_tril.data = torch.tensor(tril_full).to(torch.float)
        model.mixture_weights.data = torch.tensor(weights).to(torch.float) / np.sum(weights)
    model.eval()
    with torch.no_grad():
        total_r2 = 0
        total_acc = 0
        for batch in temp_loader:
            inputs, targets = batch
            states = model(inputs)
            total_r2 += lin_reg_r2(states, targets)
            total_acc += lin_reg_acc(states, targets, threshold=0.1, window=10)
    # print(total_r2 / len(temp_loader))
    # print(total_acc / len(temp_loader))
    model_r2s.append(total_r2 / len(temp_loader))
    model_accs.append(total_acc / len(temp_loader))
random_r2s = np.array(random_r2s)
random_accs = np.array(random_accs)
model_r2s = np.array(model_r2s)
model_accs = np.array(model_accs)

np.savez(
    f"mixture_params/{task_signal}_{task_target}_mixture{n_mixtures}.npz", 
    params=params, 
    orig_r2s=orig_r2s, 
    orig_accs=orig_accs, 
    random_r2s=random_r2s,
    random_accs=random_accs,
    model_r2s=model_r2s, 
    model_accs=model_accs,
)


# plot results

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(4, 2))
axs[0].scatter(orig_r2s, model_r2s)
axs[1].scatter(orig_accs, model_accs)
axs[0].set_xlabel("Original R2")
axs[1].set_xlabel("Original Acc")
axs[0].set_ylabel("Mixture R2")
axs[1].set_ylabel("Mixture Acc")
axs[0].set_xlim([-0.1, 1.1])
axs[1].set_xlim([-0.1, 1.1])
axs[0].set_ylim([-0.1, 1.1])
axs[1].set_ylim([-0.1, 1.1])
axs[0].plot([-0.1, 1.1], [-0.1, 1.1], 'k--', alpha=0.5)
axs[1].plot([-0.1, 1.1], [-0.1, 1.1], 'k--', alpha=0.5)
fig.tight_layout()
fig.savefig(f"mixture_params/{task_signal}_{task_target}_mixture{n_mixtures}.png", dpi=300)
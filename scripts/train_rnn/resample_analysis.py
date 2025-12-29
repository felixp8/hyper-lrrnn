import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from sklearn.mixture import GaussianMixture

from pathlib import Path
import functools

from hyper_lrrnn.rnn import LowRankRNNWithReadout, LowRankRNN, MixtureLowRankRNN
from hyper_lrrnn.training.utils import sample_dataset
from hyper_lrrnn.tasks.dm import MultiDM2
from hyper_lrrnn.training.loss import lin_reg_r2, lin_reg_acc

## Set up model and data

# run_dir = Path("./results/run-2025-12-26_09-43-04")
run_dir = Path("./results/run-2025-12-26_09-43-18")
# run_dir = Path("./results/run-2025-12-28_01-03-16")

task = functools.partial(
    MultiDM2,
    dt=20,
    signal="freq",
    target="1"
)
train_dataset, test_dataset = sample_dataset(task, dataset_size=2000, train_frac=0.8, noise_std=0.1, seed=0)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False)
inputs, targets = next(iter(test_loader))
input_size = inputs.shape[-1]
output_size = targets.shape[-1]

model = LowRankRNNWithReadout(
    input_size=input_size, output_size=output_size, 
    alpha=0.1, activation="tanh",
    rank=5, hidden_size=500,
    orth_gain=1.0, pr_gain=1.0,
    dropout_rate=0.0,
)

checkpoint = sorted(run_dir.glob("*/*/*.ckpt"))[0]
state_dict = torch.load(checkpoint, map_location="cpu")
model.load_state_dict(state_dict)

model_m = model.m.data # * math.sqrt(model.hidden_size)
model_n = model.n.data # * math.sqrt(model.hidden_size)
model_I = model.I.data
with torch.no_grad():
    model.m.data = model_m
    model.n.data = model_n
    model.I.data = model_I

## Evaluate base model

def eval_model(model, test_loader):
    total_r2 = 0.0
    total_acc = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                inputs = batch['input']
                targets = batch['target']
            else:
                inputs, targets = batch
            states = model(inputs)
            # pl_module exposes r2_fn and acc_fn that accept (output, target)
            r2 = lin_reg_r2(states, targets)
            acc = lin_reg_acc(states, targets)
            total_r2 += float(r2)
            total_acc += float(acc) if not (isinstance(acc, float) and np.isnan(acc)) else 0.0
            n_batches += 1
    avg_r2 = total_r2 / n_batches
    avg_acc = total_acc / n_batches
    return avg_r2, avg_acc

base_r2, base_acc = eval_model(model, test_loader)
print(f"Base model - R2: {base_r2:.4f}, Acc: {base_acc:.4f}")

## Evaluate model with single neurons removed

n_neurons = model.hidden_size
mod_r2s = []
mod_accs = []
for neuron_idx in range(n_neurons):
    print(f"Evaluating model with neuron {neuron_idx} removed...", end='\r')
    # Create copies of m, n, I with the neuron removed
    m_mod = torch.cat([model_m[:neuron_idx, :], model_m[neuron_idx+1:, :]], dim=0)
    n_mod = torch.cat([model_n[:neuron_idx, :], model_n[neuron_idx+1:, :]], dim=0)
    I_mod = torch.cat([model_I[:neuron_idx, :], model_I[neuron_idx+1:, :]], dim=0)
    
    # Create a new model instance with modified parameters
    model_mod = LowRankRNN(
        input_size=input_size, output_size=output_size, 
        alpha=0.1, activation="tanh",
        rank=model.rank, hidden_size=n_neurons - 1,
        orth_gain=1.0, pr_gain=1.0,
        dropout_rate=0.0,
    )
    with torch.no_grad():
        model_mod.m.data = m_mod
        model_mod.n.data = n_mod
        model_mod.I.data = I_mod
    
    # Evaluate modified model
    mod_r2, mod_acc = eval_model(model_mod, test_loader)
    mod_r2s.append(mod_r2)
    mod_accs.append(mod_acc)

plt.hist(mod_r2s, bins=30)
plt.axvline(base_r2, color='r', linestyle='dashed', linewidth=1)
plt.title("Distribution of R2 after single neuron removal")
plt.xlabel("R2")
plt.ylabel("Count")
plt.savefig("r2_distribution_after_single_neuron_removal.png")
plt.clf()

plt.hist(mod_accs, bins=30)
plt.axvline(base_acc, color='r', linestyle='dashed', linewidth=1)
plt.title("Distribution of Accuracy after single neuron removal")
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.savefig("accuracy_distribution_after_single_neuron_removal.png")
plt.clf()

## Evaluate mixture approximations with increasing numbers of mixtures

def build_mixture_model(model, n_mix):
    params = torch.cat([model.m, model.n, model.I], dim=1).detach().cpu().numpy()

    gmm = GaussianMixture(n_components=n_mix, covariance_type='full', n_init=5)
    gmm.fit(params)

    n_mix = gmm.n_components
    dim = gmm.means_.shape[1]

    # create a mixture model with same config as underlying model
    input_size = model.I.shape[-1]
    hidden_size = model.m.shape[0]
    rank = model.m.shape[1]
    mix_model = MixtureLowRankRNN(
        input_size, hidden_size, rank=rank,
        num_mixtures=n_mix, alpha=getattr(model, 'alpha', 0.1), activation='tanh',
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

n_mixtures_list = [1, 2, 3]
mix_r2s = []
mix_accs = []
for n_mix in n_mixtures_list:
    mix_model = build_mixture_model(model, n_mix)
    mix_r2, mix_acc = eval_model(mix_model, test_loader)
    mix_r2s.append(mix_r2)
    mix_accs.append(mix_acc)
    print(f"Mixture model with {n_mix} components - R2: {mix_r2:.4f}, Acc: {mix_acc:.4f}")
# plt.plot(n_mixtures_list, mix_r2s, marker='o', label='Mixture Model R2')
# plt.axhline(base_r2, color='r', linestyle='dashed', label='Base Model R2')
# plt.xlabel("Number of Mixture Components")
# plt.ylabel("R2")
# plt.title("Mixture Model R2 vs Number of Components")
# plt.legend()
# plt.savefig("mixture_model_r2_vs_components.png")
# plt.clf()

# plt.plot(n_mixtures_list, mix_accs, marker='o', label='Mixture Model Acc')
# plt.axhline(base_acc, color='r', linestyle='dashed', label='Base Model Acc')
# plt.xlabel("Number of Mixture Components")
# plt.ylabel("Accuracy")
# plt.title("Mixture Model Accuracy vs Number of Components")
# plt.legend()
# plt.savefig("mixture_model_accuracy_vs_components.png")
# plt.clf()

## Plot all weight 2d marginals

params = torch.cat([model_m, model_n, model_I], dim=1).detach().cpu().numpy()

gmm = GaussianMixture(n_components=n_mix, covariance_type='full', n_init=5)
gmm.fit(params)

rank = model_m.shape[-1]
in_dim = model_I.shape[1]
fig, axs = plt.subplots(2 * rank + in_dim - 1, 2 * rank + in_dim - 1, figsize=(15, 15))
params = torch.cat([model_m, model_n, model_I], dim=1)
param_names = [f"m_{i}" for i in range(rank)] + [f"n_{i}" for i in range(rank)] + [f"I_{i}" for i in range(in_dim)]
for i in range(1, params.shape[1]):
    for j in range(i):
        axs[i - 1, j].scatter(params[:, j].cpu().numpy(), params[:, i].cpu().numpy(), s=1, c=mod_r2s, cmap='viridis')
        if i == params.shape[1] - 1:
            axs[i - 1, j].set_xlabel(param_names[j])
        if j == 0:
            axs[i - 1, j].set_ylabel(param_names[i])
plt.tight_layout()
plt.savefig("weight_2d_marginals.png")
plt.clf()
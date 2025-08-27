import numpy as np
from hyper_lrrnn.training.utils import sample_dataset
from hyper_lrrnn.tasks.dm import MultiDM
import functools
import itertools


task_signal = ["mean", "time", "integral"]
task_target = ["1", "2", "sum", "diff"]

for signal, target in itertools.product(task_signal, task_target):
    task = functools.partial(
        MultiDM,
        signal=signal,
        target=target,
    )

    train_dataset, test_dataset = sample_dataset(task, dataset_size=2000, train_frac=0.8, noise_std=0.1, seed=0)
    inputs, targets = train_dataset.dataset.tensors
    np.savez(f"datasets/{signal}_{target}.npz", inputs=inputs.detach().cpu().numpy(), targets=targets.detach().cpu().numpy())
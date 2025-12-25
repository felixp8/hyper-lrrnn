import numpy as np
from hyper_lrrnn.training.utils import sample_dataset
from hyper_lrrnn.tasks.dm import MultiDM, MultiDM2
import functools
import itertools
from pathlib import Path


Path("datasets").mkdir(exist_ok=True)


task_class = MultiDM2
task_signal = {
    MultiDM: ["mean", "time", "integral"],
    MultiDM2: ["mean", "amp", "freq"],
}[task_class]
task_target = ["1", "2", "sum", "diff"]


for signal, target in itertools.product(task_signal, task_target):
    task = functools.partial(
        task_class,
        signal=signal,
        target=target,
    )

    train_dataset, test_dataset = sample_dataset(task, dataset_size=2000, train_frac=0.8, noise_std=0.1, seed=0)
    inputs, targets = train_dataset.dataset.tensors
    np.savez(f"datasets/{signal}_{target}.npz", inputs=inputs.detach().cpu().numpy(), targets=targets.detach().cpu().numpy())
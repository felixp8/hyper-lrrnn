import numpy as np
import torch

def sample_dataset(task, dataset_size, train_frac, noise_std, kwargs=dict(), seed=0):
    env = task(**kwargs)
    env.reset(seed=seed)

    inputs = []
    targets = []
    for i in range(dataset_size):
        env.new_trial()
        inputs.append(env.ob + np.random.normal(0, noise_std, size=env.ob.shape))
        targets.append(env.gt)
    inputs = np.stack(inputs, axis=0)
    targets = np.stack(targets, axis=0)
    if targets.ndim == 2:
        targets = targets[..., None]
    dataset = torch.utils.data.TensorDataset(torch.Tensor(inputs), torch.Tensor(targets))
    train_size = int(train_frac * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset
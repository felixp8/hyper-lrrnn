import torch


# MMD
# from https://github.com/mackelab/labproject/blob/main/labproject/metrics/MMD_torch.py

def rbf_kernel(x, y, bandwidth):
    dist = torch.cdist(x, y)
    return torch.exp(-(dist**2) / (2.0 * bandwidth**2))


def median_heuristic(x, y):
    return torch.median(torch.cdist(x, y))


def compute_rbf_mmd(x, y, bandwidth=1.0):
    x_kernel = rbf_kernel(x, x, bandwidth)
    y_kernel = rbf_kernel(y, y, bandwidth)
    xy_kernel = rbf_kernel(x, y, bandwidth)
    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd


def compute_rbf_mmd_median_heuristic(x, y):
    # https://arxiv.org/pdf/1707.07269.pdf
    bandwidth = median_heuristic(x, y)
    return compute_rbf_mmd(x, y, bandwidth)


def compute_rbf_mmd_auto(x, y, bandwidth=1.0):
    dim = x.shape[1]
    x_kernel = rbf_kernel(x, x, dim * bandwidth)
    y_kernel = rbf_kernel(y, y, dim * bandwidth)
    xy_kernel = rbf_kernel(x, y, dim * bandwidth)
    mmd = torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    return mmd

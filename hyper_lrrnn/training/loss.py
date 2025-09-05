import numpy as np
import torch
from sklearn.metrics import r2_score


def lin_reg_loss(inputs, targets, alpha=0.0):
    """Linear regression loss"""
    with torch.no_grad():
        X = inputs.reshape(-1, inputs.shape[-1])
        y = targets.reshape(-1, targets.shape[-1])
        if alpha > 0:
            XTy = X.T @ y
            XTX = X.T @ X + alpha * torch.eye(X.shape[-1], device=X.device)
            w = torch.linalg.lstsq(XTX, XTy).solution
        else:
            w = torch.linalg.lstsq(X, y).solution
    return torch.mean((targets - inputs @ w) ** 2)


@torch.no_grad()
def lin_reg_r2(inputs, targets, alpha=0.0):
    X = inputs.reshape(-1, inputs.shape[-1])
    y = targets.reshape(-1, targets.shape[-1])
    if alpha > 0:
        XTy = X.T @ y
        XTX = X.T @ X + alpha * torch.eye(X.shape[-1], device=X.device)
        w = torch.linalg.lstsq(XTX, XTy).solution
    else:
        w = torch.linalg.lstsq(X, y).solution
    pred = inputs @ w
    targets = targets.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    return r2_score(targets.reshape(-1, targets.shape[-1]), pred.reshape(-1, pred.shape[-1]))


@torch.no_grad()
def lin_reg_acc(inputs, targets, alpha=0.0, threshold=0.5, window=10):
    orig_targets_shape = targets.shape
    X = inputs.reshape(-1, inputs.shape[-1])
    y = targets.reshape(-1, targets.shape[-1])
    if alpha > 0:
        XTy = X.T @ y
        XTX = X.T @ X + alpha * torch.eye(X.shape[-1], device=X.device)
        w = torch.linalg.lstsq(XTX, XTy).solution
    else:
        w = torch.linalg.lstsq(X, y).solution
    pred = inputs @ w
    targets = targets.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    targets = targets.reshape(*orig_targets_shape)
    pred = pred.reshape(*orig_targets_shape)
    return accuracy(pred, targets, threshold=threshold, window=window)


@torch.no_grad()
def accuracy(pred, targets, threshold=0.5, window=10):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    model_resp = np.where(
        np.all(np.abs(pred[:, -window:, 0]) < threshold, axis=-1),
        np.zeros(pred.shape[0]),
        np.sign(np.mean(pred[:, -window:, 0], axis=-1))
    )
    correct_resp = np.where(
        np.all(np.abs(targets[:, -window:, 0]) < threshold, axis=-1),
        np.zeros(targets.shape[0]),
        np.sign(np.mean(targets[:, -window:, 0], axis=-1))
    )
    assert not np.any(correct_resp == 0)
    accuracy = np.mean(model_resp == correct_resp)
    return accuracy
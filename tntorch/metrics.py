import torch
import numpy as np
import tntorch as tn


def _process(gt, approx):
    assert np.array_equal(gt.shape, approx.shape)
    if isinstance(gt, tn.Tensor) and not isinstance(approx, tn.Tensor):
        gt = gt.full()
    if not isinstance(gt, tn.Tensor) and isinstance(approx, tn.Tensor):
        approx = approx.full()
    return gt, approx


def relative_error(gt, approx):
    """
    Compute the relative error between two tensors (torch or tntorch)

    :param gt: a torch or tntorch tensor
    :param approx: a torch or tntorch tensor
    :return: a scalar >= 0

    """

    gt, approx = _process(gt, approx)
    if isinstance(gt, torch.Tensor) and isinstance(approx, torch.Tensor):
        return torch.norm(gt-approx) / torch.norm(gt)
    return tn.norm(gt-approx) / tn.norm(gt)


def rmse(gt, approx):
    """
    Compute the RMSE between two tensors (torch or tntorch)

    :param gt: a torch or tntorch tensor
    :param approx: a torch or tntorch tensor
    :return: a scalar >= 0

    """

    gt, approx = _process(gt, approx)
    if isinstance(gt, torch.Tensor) and isinstance(approx, torch.Tensor):
        return torch.norm(gt-approx) / np.sqrt(gt.numel())
    return (gt-approx).norm() / torch.sqrt(gt.size)


def r_squared(gt, approx):
    """
    Compute the R^2 score between two tensors (torch or tntorch)

    :param gt: a torch or tntorch tensor
    :param approx: a torch or tntorch tensor
    :return: a scalar <= 1

    """

    gt, approx = _process(gt, approx)
    if isinstance(gt, torch.Tensor) and isinstance(approx, torch.Tensor):
        return 1 - torch.norm(gt-approx)**2 / torch.norm(gt-torch.mean(gt))**2
    return 1 - tn.normsq(gt-approx) / tn.normsq(gt-tn.mean(gt))

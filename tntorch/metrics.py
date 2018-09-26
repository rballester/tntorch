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


def distance(t1, t2):
    """
    Computes the Euclidean distance between two tensors. Generally faster than tn.norm(t1-t2).

    :param t1:
    :param t2:
    :return: a scalar >= 0

    """

    t1, t2 = _process(t1, t2)
    return torch.sqrt(tn.dot(t1, t1) + tn.dot(t2, t2) - 2 * tn.dot(t1, t2))


def relative_error(gt, approx):
    """
    Computes the relative error between two tensors (torch or tntorch).

    :param gt: a torch or tntorch tensor
    :param approx: a torch or tntorch tensor
    :return: a scalar >= 0

    """

    gt, approx = _process(gt, approx)
    dotgt = tn.dot(gt, gt)
    return torch.sqrt(dotgt + tn.dot(approx, approx) - 2*tn.dot(gt, approx)) / torch.sqrt(dotgt)


def rmse(gt, approx):
    """
    Computes the RMSE between two tensors (torch or tntorch).

    :param gt: a torch or tntorch tensor
    :param approx: a torch or tntorch tensor
    :return: a scalar >= 0

    """

    gt, approx = _process(gt, approx)
    if isinstance(gt, torch.Tensor) and isinstance(approx, torch.Tensor):
        return torch.norm(gt-approx) / np.sqrt(gt.numel())
    return tn.distance(gt, approx) / torch.sqrt(gt.size)


def r_squared(gt, approx):
    """
    Computes the R^2 score between two tensors (torch or tntorch).

    :param gt: a torch or tntorch tensor
    :param approx: a torch or tntorch tensor
    :return: a scalar <= 1

    """

    gt, approx = _process(gt, approx)
    if isinstance(gt, torch.Tensor) and isinstance(approx, torch.Tensor):
        return 1 - torch.norm(gt-approx)**2 / torch.norm(gt-torch.mean(gt))**2
    return 1 - tn.distance(gt, approx)**2 / tn.normsq(gt-tn.mean(gt))

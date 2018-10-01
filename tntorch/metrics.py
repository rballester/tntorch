import torch
import numpy as np
import tntorch as tn


def _process(gt, approx):
    """
    If only one of the arguments is a compressed tensor, we decompress it
    """

    assert np.array_equal(gt.shape, approx.shape)
    is1 = isinstance(gt, tn.Tensor)
    is2 = isinstance(approx, tn.Tensor)
    if is1 and is2:
        return gt, approx
    if is1:
        gt = gt.full()
    if is2:
        approx = approx.full()
    return gt, approx


def dot(t1, t2, k=None):  # TODO support partial dot products
    """
    Computes the dot product between two tensors.

    :param t1: a tensor
    :param t2: a tensor
    :return: a scalar

    """

    t1, t2 = _process(t1, t2)
    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
        return t1.flatten().dot(t2.flatten())

    assert np.array_equal(t1.shape, t2.shape)

    if k is None:
        k = min(t1.ndim, t2.ndim)
    Lprod = torch.ones([t1.cores[-1].shape[-1], t2.cores[-1].shape[-1]])
    for mu in range(t1.ndim-1, t1.ndim-1-k, -1):
        core1 = t1.cores[mu]
        core2 = t2.cores[mu]
        if t1.Us[mu] is None:
            if t2.Us[mu] is not None:
                if core1.dim() == 3:
                    core1 = torch.einsum('iak,aj->ijk', (core1, t2.Us[mu]))
                else:
                    core1 = torch.einsum('ak,aj->jk', (core1, t2.Us[mu]))
        elif t2.Us[mu] is None:
            if core2.dim() == 3:
                core2 = torch.einsum('iak,aj->ijk', (core2, t1.Us[mu]))
            else:
                core2 = torch.einsum('ak,aj->jk', (core2, t1.Us[mu]))
        else:  # Both have Tucker factors
            if core2.dim() == 3:
                core2 = torch.einsum('ar,aj,ijk->irk', (t1.Us[mu], t2.Us[mu], core2))
            else:
                core2 = torch.einsum('ar,aj,jk->rk', (t1.Us[mu], t2.Us[mu], core2))
        if core1.dim() == 3:
            Ucore = torch.einsum('ijk,ka->ija', (core1, Lprod))
        else:
            Ucore = torch.einsum('ji,ik->ijk', (core1, Lprod))
        Vcore = core2
        if Vcore.dim() == 3:
            Lprod = torch.mm(Ucore.reshape([Ucore.shape[0], -1]), torch.t(Vcore.reshape([Vcore.shape[0], -1])))
        else:
            Lprod = torch.einsum('ijs,js->is', (Ucore, Vcore))
    return torch.sum(Lprod)


def distance(t1, t2):
    """
    Computes the Euclidean distance between two tensors. Generally faster than tn.norm(t1-t2).

    :param t1: a tensor
    :param t2: a tensor
    :return: a scalar >= 0

    """

    t1, t2 = _process(t1, t2)
    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
        return torch.norm(t1-t2)
    return torch.sqrt(tn.dot(t1, t1) + tn.dot(t2, t2) - 2 * tn.dot(t1, t2).clamp(0))


def relative_error(gt, approx):
    """
    Computes the relative error between two tensors (torch or tntorch).

    :param gt: a torch or tntorch tensor
    :param approx: a torch or tntorch tensor
    :return: a scalar >= 0

    """

    gt, approx = _process(gt, approx)
    if isinstance(gt, torch.Tensor) and isinstance(approx, torch.Tensor):
        return torch.norm(gt-approx) / torch.norm(gt)
    dotgt = tn.dot(gt, gt)
    return torch.sqrt((dotgt + tn.dot(approx, approx) - 2*tn.dot(gt, approx)).clamp(0)) / torch.sqrt(dotgt.clamp(0))


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


def mean(t):
    """
    Computes the mean of a tensor.

    :param t: a tensor
    :return: a scalar

    """

    return tn.sum(t) / t.size


def var(t):
    """
    Computes the variance of a tensor.

    :param t: a tensor
    :return: a scalar >= 0

    """

    return tn.normsq(t-tn.mean(t)) / t.size


def std(t):
    """
    Computes the standard deviation of a tensor.

    :param t: a tensor
    :return: a scalar >= 0

    """

    return torch.sqrt(tn.var(t))


def normsq(t):
    """
    Computes the squared norm of a tensor.

    :param t: a tensor
    :return: a scalar >= 0

    """

    return tn.dot(t, t)


def norm(t):
    """
    Computes the L^2 (Frobenius) norm of a tensor.

    :param t: a tensor
    :return: a scalar >= 0

    """

    return torch.sqrt(torch.clamp(tn.normsq(t), min=0))

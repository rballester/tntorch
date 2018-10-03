import torch
import numpy as np
import tntorch as tn


def _process(gt, approx):
    """
    If *only one* of the arguments is a compressed tensor, we decompress it
    """

    # assert np.array_equal(gt.shape, approx.shape)
    is1 = isinstance(gt, tn.Tensor)
    is2 = isinstance(approx, tn.Tensor)
    if is1 and is2:
        return gt, approx
    if is1:
        gt = gt.full()
    if is2:
        approx = approx.full()
    return gt, approx


def dot(t1, t2):
    """
    Computes the dot product between two tensors. If their dimensionalities N1 and N2 differ (say, N1 < N2), then only
    the *trailing* N1 dimensions will be contracted; the result will have dimension N2 - N1.

    Example:
    tn.rand([2, 3, 4]).dot(tn.rand[3, 4])  # Result has shape [2]
    tn.rand([3, 4]).dot(tn.rand[3, 4])  # Result is a scalar tensor

    :param t1: a tensor
    :param t2: a tensor
    :return: a scalar (if `t1` and `t2` have the same number of dimensions) or tensor otherwise

    """

    t1, t2 = _process(t1, t2)
    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
        return t1.flatten().dot(t2.flatten())

    if t1.dim() < t2.dim():
        return tn.dot(t2, t1)
    Lprod = torch.ones([t1.cores[-1].shape[-1], t2.cores[-1].shape[-1]])
    k = min(t1.dim(), t2.dim())
    if not np.array_equal(t1.shape[-k:], t2.shape[-k:]):
        raise ValueError('Dot product requires trailing dimensions to be equal, but they are {} and {}'.format(t1.shape[-k:], t2.shape[-k:]))
    for mu in range(k):
        mu1 = t1.dim()-1-mu
        mu2 = t2.dim()-1-mu
        core1 = t1.cores[mu1]
        core2 = t2.cores[mu2]
        if t1.Us[mu1] is None:
            if t2.Us[mu2] is not None:
                if core1.dim() == 3:
                    core1 = torch.einsum('iak,aj->ijk', (core1, t2.Us[mu2]))
                else:
                    core1 = torch.einsum('ak,aj->jk', (core1, t2.Us[mu2]))
        elif t2.Us[mu2] is None:
            if core2.dim() == 3:
                core2 = torch.einsum('iak,aj->ijk', (core2, t1.Us[mu1]))
            else:
                core2 = torch.einsum('ak,aj->jk', (core2, t1.Us[mu1]))
        else:  # Both have Tucker factors
            if core2.dim() == 3:
                core2 = torch.einsum('ar,aj,ijk->irk', (t1.Us[mu1], t2.Us[mu2], core2))
            else:
                core2 = torch.einsum('ar,aj,jk->rk', (t1.Us[mu1], t2.Us[mu2], core2))
        if core1.dim() == 3:
            Ucore = torch.einsum('ijk,ka->ija', (core1, Lprod))
        else:
            Ucore = torch.einsum('ji,ik->ijk', (core1, Lprod))
        Vcore = core2
        if Vcore.dim() == 3:
            Lprod = torch.mm(Ucore.reshape([Ucore.shape[0], -1]), torch.t(Vcore.reshape([Vcore.shape[0], -1])))
        else:
            Lprod = torch.einsum('ijs,js->is', (Ucore, Vcore))
    if k < t1.dim():
        result = tn.Tensor(t1.cores[:t1.dim()-k], t1.Us[:t1.dim()-k]).clone()
        if result.cores[-1].dim() == 3:
            result.cores[-1] = torch.einsum('iaj,jk->iak', (result.cores[-1], Lprod))
        else:
            result.cores[-1] = torch.einsum('aj,jk->jak', (result.cores[-1], Lprod))
        return result
    else:
        return torch.sum(Lprod)


def dist(t1, t2):
    """
    Computes the Euclidean distance between two tensors. Generally faster than tn.norm(t1-t2).

    :param t1: a tensor
    :param t2: a tensor
    :return: a scalar >= 0

    """

    t1, t2 = _process(t1, t2)
    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
        return torch.dist(t1, t2)
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
        return torch.dist(gt, approx) / torch.norm(gt)
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
        return torch.dist(gt, approx) / np.sqrt(gt.numel())
    return tn.dist(gt, approx) / torch.sqrt(gt.size)


def r_squared(gt, approx):
    """
    Computes the R^2 score between two tensors (torch or tntorch).

    :param gt: a torch or tntorch tensor
    :param approx: a torch or tntorch tensor
    :return: a scalar <= 1

    """

    gt, approx = _process(gt, approx)
    if isinstance(gt, torch.Tensor) and isinstance(approx, torch.Tensor):
        return 1 - torch.dist(gt, approx)**2 / torch.dist(gt, torch.mean(gt))**2
    return 1 - tn.dist(gt, approx)**2 / tn.normsq(gt-tn.mean(gt))


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

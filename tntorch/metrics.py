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
        gt = gt.torch()
    if is2:
        approx = approx.torch()
    return gt, approx


def dot(t1, t2, k=None):
    """
    Generalized tensor dot product: contracts the k leading dimensions of two tensors of dimension N1 and N2.

    - If k is None:
        - If N1 == N2, returns a scalar (dot product between the two tensors)
        - If N1 < N2, the result will have dimension N2 - N1
        - If N2 < N1, the result will have dimension N1 - N2

        Example: suppose t1 has shape 3 x 4 and t2 has shape 3 x 4 x 5 x 6. Then, tn.dot(t1, t2) will have shape
        5 x 6.

    - If k is given:
        The trailing (N1-k) dimensions from the 1st tensor will be sorted backwards, and then the trailing (N2-k)
        dimensions from the 2nd tensor will be appended to them.

        Example: suppose t1 has shape 3 x 4 x 5 x 6 and t2 has shape 3 x 4 x 10 x 11. Then, tn.dot(t1, t2, k=2) will
        have shape 6 x 5 x 10 x 11.

    :param t1: a :class:`Tensor` (or a PyTorch tensor)
    :param t2: a :class:`Tensor` (or a PyTorch tensor)
    :param k: an int (default: None)

    :return: a scalar (if k is None and t1.dim() == t2.dim()), a tensor otherwise
    """

    def _project_spatial(core, M):
        if core.dim() == 3:
            return torch.einsum('iak,aj->ijk', (core, M))
        else:
            return torch.einsum('ak,aj->jk', (core, M))

    def _project_left(core, M):
        if core.dim() == 3:
            return torch.einsum('sr,rai->sai', (M, core))
        else:
            return torch.einsum('sr,ar->sar', (M, core))

    t1, t2 = _process(t1, t2)
    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
        return t1.flatten().dot(t2.flatten())
    Lprod = torch.ones([t2.ranks_tt[0], t1.ranks_tt[0]], device=t1.cores[0].device)
    if k is None:
        k = min(t1.dim(), t2.dim())
    assert k <= t1.dim() and k <= t2.dim()
    if not np.array_equal(t1.shape[:k], t2.shape[:k]):
        raise ValueError('Dot product requires leading dimensions to be equal, but they are {} and {}'.format(t1.shape[:k], t2.shape[:k]))

    # Crunch first k dimensions of both tensors
    for mu in range(k):
        core1 = t1.cores[mu]
        core2 = t2.cores[mu]
        # First part: absorb Tucker factors
        if t1.Us[mu] is None:
            if t2.Us[mu] is not None:
                core1 = _project_spatial(core1, t2.Us[mu])
        elif t2.Us[mu] is None:
            core2 = _project_spatial(core2, t1.Us[mu])
        else:  # Both have Tucker factors
            core2 = _project_spatial(core2, torch.matmul(t2.Us[mu].t(), t1.Us[mu]))
        # Second part: advance running factor `Lprod`
        Ucore = _project_left(core1, Lprod)
        Vcore = core2
        if Vcore.dim() == 3:
            Lprod = torch.matmul(tn.left_unfolding(Vcore).t(), tn.left_unfolding(Ucore))
        else:
            Lprod = torch.einsum('as,sar->sr', (Vcore, Ucore))

    # Deal with unprocessed dimensions, if any
    if k < t1.dim():
        t1trail = tn.Tensor(t1.cores[k:], t1.Us[k:]).clone()
        t1trail.cores[0] = _project_left(t1trail.cores[0], Lprod)
        if k == t2.dim():
            return t1trail
        else:
            t2trail = tn.Tensor(t2.cores[k:], t2.Us[k:]).clone()
            t1trail = tn.transpose(t1trail)
            return tn.Tensor(t1trail.cores + t2trail.cores, Us=t1trail.Us + t2trail.Us)
    else:
        if k == t2.dim():
            return torch.sum(Lprod)
        else:
            t2trail = tn.Tensor(t2.cores[k:], t2.Us[k:])#.clone()
            t2trail.cores[0] = _project_left(t2trail.cores[0], Lprod.t())
            return t2trail


def dist(t1, t2):
    """
    Computes the Euclidean distance between two tensors. Generally faster than `tn.norm(t1-t2)`.

    :param t1: a :class:`Tensor` (or a PyTorch tensor)
    :param t2: a :class:`Tensor` (or a PyTorch tensor)

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
    return tn.dist(gt, approx) / torch.sqrt(gt.numel())


def r_squared(gt, approx):
    """
    Computes the :math:`R^2` score between two tensors (torch or tntorch).

    :param gt: a torch or tntorch tensor
    :param approx: a torch or tntorch tensor

    :return: a scalar <= 1
    """

    gt, approx = _process(gt, approx)
    if isinstance(gt, torch.Tensor) and isinstance(approx, torch.Tensor):
        return 1 - torch.dist(gt, approx)**2 / torch.dist(gt, torch.mean(gt))**2
    return 1 - tn.dist(gt, approx)**2 / tn.normsq(gt-tn.mean(gt))


def mean(t, dim=None, keepdim=False):
    """
    Computes the mean of a tensor along all or some of its dimensions.

    :param t: a :class:`Tensor`
    :param dim: an int or list of ints (default: all)
    :param keepdim: whether to keep the same number of dimensions

    :return: a scalar
    """

    denom = t.shape[dim] if dim is not None else t.numel()
    summed = tn.sum(t, dim, keepdim)
    return summed / denom


def var(t):
    """
    Computes the variance of a tensor.

    :param t: a :class:`Tensor`

    :return: a scalar >= 0
    """

    return tn.normsq(t-tn.mean(t)) / t.numel()


def std(t):
    """
    Computes the standard deviation of a tensor.

    :param t: a :class:`Tensor`

    :return: a scalar >= 0
    """

    return torch.sqrt(tn.var(t))


def normsq(t):
    """
    Computes the squared norm of a tensor.

    :param t: a :class:`Tensor`

    :return: a scalar >= 0
    """

    return tn.dot(t, t)


def norm(t):
    """
    Computes the :math:`L^2` (Frobenius) norm of a tensor.

    :param t: a :class:`Tensor`

    :return: a scalar >= 0
    """

    return torch.sqrt(torch.clamp(tn.normsq(t), min=0))

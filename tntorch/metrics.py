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

    :return: a scalar :math:`\ge 0`
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

    :return: a scalar :math:`\ge 0`
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

    :return: a scalar :math:`\ge 0`
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


def sum(t, dim=None, keepdim=False, _normalize=False):
    """
    Compute the sum of a tensor along all (or some) of its dimensions.

    :param t: input :class:`Tensor`
    :param dim: an int or list of ints. By default, all dims will be summed
    :param keepdim: if True, summed dimensions will be kept as singletons. Default is False

    :return: a scalar (if keepdim is False and all dims were chosen) or :class:`Tensor` otherwise
    """
    if dim is None:
        dim = np.arange(t.dim())

    if not hasattr(dim, '__len__'):
        dim = [dim]

    device = t.cores[0].device

    if t.batch:
        if _normalize:
            us = [(1./t.shape[d])*torch.ones((t.shape[0], t.shape[d + 1])).to(device) for d in dim]
        else:
            us = [torch.ones((t.shape[0], t.shape[d + 1])).to(device) for d in dim]
    else:
        if _normalize:
            us = [(1./t.shape[d])*torch.ones(t.shape[d]).to(device) for d in dim]
        else:
            us = [torch.ones(t.shape[d]).to(device) for d in dim]

    result = tn.ttm(t, us, dim)
    if keepdim:
        return result
    else:
        if t.batch and t.shape[0] == 1:
            return tn.squeeze(result, np.arange(len(t.shape)))
        elif t.batch:
            return tn.squeeze(result, np.arange(1, len(t.shape)))
        else:
            return tn.squeeze(result)

def mean(t, dim=None, keepdim=False):
    """
    Computes the mean of a :class:`Tensor` along all or some of its dimensions.

    :param t: a :class:`Tensor`
    :param dim: an int or list of ints (default: all)
    :param keepdim: whether to keep the same number of dimensions

    :return: a scalar (if keepdim is False and all dims were chosen) or :class:`Tensor` otherwise
    """

    return tn.sum(t, dim, keepdim, _normalize=True)


def var(t):
    """
    Computes the variance of a :class:`Tensor`.

    :param t: a :class:`Tensor`

    :return: a scalar :math:`\ge 0`
    """

    return tn.normsq(t-tn.mean(t)) / t.numel()


def std(t):
    """
    Computes the standard deviation of a :class:`Tensor`.

    :param t: a :class:`Tensor`

    :return: a scalar :math:`\ge 0`
    """

    return torch.sqrt(tn.var(t))


def skew(t):
    """
    Computes the skewness of a :class:`Tensor`. Note: this function uses cross-approximation (:func:`tntorch.cross()`).

    :param t: a :class:`Tensor`

    :return: a scalar
    """

    return tn.mean(((t-tn.mean(t))/tn.std(t))**3)


def kurtosis(t, fisher=True):
    """
    Computes the kurtosis of a :class:`Tensor`. Note: this function uses cross-approximation (:func:`tntorch.cross()`).

    :param t: a :class:`Tensor`
    :param fisher: if True (default) Fisher's definition is used, otherwise Pearson's (aka excess)

    :return: a scalar
    """

    return tn.mean(((t-tn.mean(t))/tn.std(t))**4) - fisher*3


def raw_moment(t, k, eps=1e-6):
    """
    Compute a raw moment :math:`\\mathbb{E}[t^k]'.

    :param t: input :class:`Tensor`
    :param k: the desired moment order (integer :math:`\ge 1`)
    :param eps: relative error for rounding (default is 1e-6)

    :return: the :math:`k`-th order raw moment of `t`
    """

    return hadamard_sum([t]*k, eps=eps) / t.numel()


def normalized_moment(t, k, eps=1e-12):
    """
    Compute a normalized central moment :math:`\\mathbb{E}[(t - \\mathbb{E}[t])^k] / \\sigma^k'.

    :param t: input :class:`Tensor`
    :param k: the desired moment order (integer :math:`\ge 1`)
    :param eps: relative error for rounding (default is 1e-12)

    :return: the :math:`k`-th order normalized moment of `t`
    """

    return raw_moment(t-tn.mean(t), k=k, eps=eps) / tn.var(t)**(k/2.) / t.numel()


def hadamard_sum(ts, eps=1e-6):
    """
    Given tensors :math:`t_1, \\dots, t_M`, computes :math:'\\Sum (t_1 \\circ \\dots \\circ t_M)'.

    Reference: this is a variant of A. Novikov et al., "Putting MRFs on a Tensor Train" (2016), Alg. 1

    :param ts: a list of :class:`Tensor` (the algorithm will use temporary TT-format copies of those)
    :param eps: relative error used at each rounding step (default is 1e-6)

    :return: a scalar
    """

    def diag_core(c, m):
        """
        Takes a TT core of shape Rl x I x Rr and organizes it as a tensor of shape I x Rl x Rr x I
        """

        factor = c.permute(0, 2, 1)
        factor = torch.reshape(factor, [-1, factor.shape[-1]])
        core = torch.zeros(factor.shape[1], factor.shape[1] + 1, factor.shape[0])
        core[:, 0, :] = factor.t()
        core = core.reshape(factor.shape[1] + 1, factor.shape[1], factor.shape[0]).permute(0, 2, 1)[:-1, :, :]
        core = core.reshape([c.shape[1], c.shape[0], c.shape[2], c.shape[1]])
        if m == 0:
            core = torch.sum(core, dim=0, keepdim=True)
        if m == M-1:
            core = torch.sum(core, dim=-1, keepdim=True)
        return core

    def get_tensor(cores):
        M = len(cores)
        cs = []
        for m in range(M):
            c = diag_core(cores[m], m)
            cs.append(c.reshape(c.shape[0], c.shape[1]*c.shape[2], c.shape[3]))
        t = tn.Tensor(cs)
        t.round_tt(eps)
        cs = t.cores
        cs = [cs[m].reshape([cs[m].shape[0], cores[m].shape[0], cores[m].shape[2], cs[m].shape[-1]]) for m in range(M)]
        return cs

    M = len(ts)
    tstt = []
    for m in range(M):  # Convert everything to the TT format
        t = ts[m].decompress_tucker_factors()
        t._cp_to_tt()
        tstt.append(t)
    ts = tstt
    N = ts[0].dim()
    thiscores = get_tensor([t.cores[0] for t in ts])

    for n in range(1, N):
        nextcores = get_tensor([t.cores[n] for t in ts])
        newcores = []
        for m in range(M):
            c = torch.einsum('ijkl,akbc->iajblc', (thiscores[m], nextcores[m]))  # vecmat product
            c = torch.reshape(c, [c.shape[0]*c.shape[1]*c.shape[2], c.shape[3], c.shape[4]*c.shape[5]])
            newcores.append(c)
        thiscores = tn.round_tt(tn.Tensor(newcores), eps=eps).cores

        if n < N-1:
            for m in range(M):  # Cast the vector as a TT-matrix for the next iteration
                thiscores[m] = thiscores[m].reshape(thiscores[m].shape[0], 1, thiscores[m].shape[1], -1)
        else:
            return tn.Tensor(thiscores).torch().item()


def normsq(t):
    """
    Computes the squared norm of a :class:`Tensor`.

    :param t: a :class:`Tensor`

    :return: a scalar :math:`\ge 0`
    """

    return tn.dot(t, t)


def norm(t):
    """
    Computes the :math:`L^2` (Frobenius) norm of a tensor.

    :param t: a :class:`Tensor`

    :return: a scalar :math:`\ge 0`
    """

    return torch.sqrt(torch.clamp(tn.normsq(t), min=0))

import tntorch as tn
import torch
import numpy as np


def partialset(t, order=1, mask=None, bounds=None):
    """
    Given a tensor, compute another one that contains all partial derivatives of certain order(s) and according
    to some optional mask.

    :param t: a tensor
    :param order: an int or list of ints. Default is 1
    :param mask: an optional mask to select only a subset of partials
    :param bounds: a list of pairs [lower bound, upper bound] specifying parameter ranges (used to compute
    derivative steps). If None (default), all steps will be 1
    :return: a tensor

    Examples:

    t = tn.rand([10, 10, 10])  # A 3D tensor
    x, y, z = tn.symbols(3)
    partialset(t, 1, x)  # x
    partialset(t, 2, x)  # xx, xy, xz
    partialset(t, 2, tn.only(y | z))  # yy, yz, zz

    """

    if bounds is None:
        bounds = [[0, sh-1] for sh in t.shape]
    if not hasattr(order, '__len__'):
        order = [order]

    max_order = max(order)
    def diff(core, n):
        if core.dim() == 3:
            pad = torch.zeros(core.shape[0], 1, core.shape[2])
        else:
            pad = torch.zeros(1, core.shape[1])
        if core.shape[1] == 1:
            return pad
        step = (bounds[n][1] - bounds[n][0]) / (core.shape[-2] - 1)
        return torch.cat(((core[..., 1:, :] - core[..., :-1, :]) / step, pad), dim=-2)
    cores = []
    idxs = []
    for n in range(t.dim()):
        if t.Us[n] is None:
            stack = [t.cores[n]]
        else:
            stack = [torch.einsum('ijk,aj->iak', (t.cores[n], t.Us[n]))]
        idx = torch.zeros([t.shape[n]])
        for o in range(1, max_order+1):
            stack.append(diff(stack[-1], n))
            idx = torch.cat((idx, torch.ones(stack[-1].shape[-2])*o))
            if o == max_order:
                break
        cores.append(torch.cat(stack, dim=-2))
        idxs.append(idx)
    d = tn.Tensor(cores, idxs=idxs)
    wm = tn.automata.weight_mask(t.dim(), order, nsymbols=max_order+1)
    if mask is not None:
        wm = tn.mask(wm, mask)
    result = tn.mask(d, wm)
    result.idxs = idxs
    return result


def partial(t, dim, order=1, bounds=None):
    """
    Compute a single partial derivative.

    :param t: a tensor
    :param dim: int or list of ints
    :param order: how many times to derive. Default is 1
    :param bounds: variable(s) range bounds (to compute the derivative step). If None (default), step 1 will be assumed
    :return: a tensor

    """

    if not hasattr(dim, '__len__'):
        dim = [dim]
    if bounds is None:
        bounds = [[0, t.shape[n]-1] for n in range(t.dim())]
    if not hasattr(bounds[0], '__len__'):
        bounds = [bounds]

    t2 = t.clone()
    for i, d in enumerate(dim):
        for o in range(1, order+1):
            step = (bounds[i][1] - bounds[i][0]) / (t.shape[d]-1)
            if t2.Us[d] is None:
                t2.cores[d] = (t2.cores[d][..., 1:, :] - t2.cores[d][..., :-1, :]) / step
                if t2.cores[d].dim() == 3:
                    pad = torch.zeros(t2.cores[d].shape[0], 1, t2.cores[d].shape[2])
                else:
                    pad = torch.zeros(1, t2.cores[d].shape[1])
                t2.cores[d] = torch.cat((t2.cores[d], pad), dim=-2)
            else:
                t2.Us[d] = (t2.Us[d][1:, :] - t2.Us[d][:-1, :]) / step
                t2.Us[d] = torch.cat((t2.Us[d], torch.zeros(1, t2.cores[d].shape[-2])), dim=0)
    return t2


def gradient(t, dim='all', bounds=None):
    """
    Compute the gradient of a tensor

    :param t: a tensor
    :param dim: an integer (or list of integers). Default is all
    :param bounds: a pair (or list of pairs) of reals, or None. The bounds for each variable

    :return: a tensor (or a list of tensors)

    """

    if dim == 'all':
        dim = range(t.dim())
    if bounds is None:
        bounds = [[0, t.shape[d]-1] for d in dim]
    if not hasattr(bounds, '__len__'):
        bounds = [bounds]*len(dim)

    if not hasattr(dim, '__len__'):
        return partial(t, dim, bounds)
    else:
        return [partial(t, d, order=1, bounds=b) for d, b in zip(dim, bounds)]


def active_subspace(t):
    """
    Compute the main variational directions of a TT

    Reference: P. Constantine et al., "Discovering an Active Subspace in a
Single-Diode Solar Cell Model" (2017). Available: https://arxiv.org/pdf/1406.7607.pdf

    See also: https://github.com/paulcon/as-data-sets/blob/master/README.md

    :param t: a TT
    :return: (eigvals, eigvecs): an array and a matrix, encoding the eigenpairs in descending order

    """

    grad = tn.gradient(t, dim='all')

    M = torch.zeros(t.dim(), t.dim())
    for i in range(t.dim()):
        for j in range(i, t.dim()):
            M[i, j] = tn.dot(grad[i], grad[j]) / t.size
            M[j, i] = M[i, j]

    w, v = torch.symeig(M, eigenvectors=True)
    idx = range(t.dim()-1, -1, -1)
    w = w[idx]
    v = v[:, idx]
    return w, v


def divergence(ts, bounds=None):
    """
    Computes the divergence (scalar field) out of a vector field encoded in a tensor.

    :param ts: an N-D vector field, encoded as a list of N N-D tensors
    :param bounds:
    :return: a scalar field

    """

    assert ts[0].dim() == len(ts)
    assert all([t.shape == ts[0].shape for t in ts[1:]])
    if bounds is None:
        bounds = [None]*len(ts)
    elif not hasattr(bounds[0], '__len__'):
        bounds = [bounds for n in range(len(ts))]
    assert len(bounds) == len(ts)

    return sum([tn.partial(ts[n], n, order=1, bounds=bounds[n]) for n in range(len(ts))])


def curl(ts, bounds=None):
    """
    Compute the curl of a vector field

    :param ts: three 3D encoding the x, y, and z vector coordinates respectively
    :param bounds:
    :return: three tensors of the same shape

    """

    assert [t.dim() == 3 for t in ts]
    assert len(ts) == 3
    if bounds is None:
        bounds = [None for n in range(3)]
    elif not hasattr(bounds[0], '__len__'):
        bounds = [bounds for n in range(3)]
    assert len(bounds) == 3

    return [tn.partial(ts[2], 1, bounds=bounds[1]) - tn.partial(ts[1], 2, bounds=bounds[2]),
            tn.partial(ts[0], 2, bounds=bounds[2]) - tn.partial(ts[2], 0, bounds=bounds[0]),
            tn.partial(ts[1], 0, bounds=bounds[0]) - tn.partial(ts[0], 1, bounds=bounds[1])]


def laplacian(t, bounds=None):
    """
    Computes the Laplacian of a scalar field.

    :param t: a tensor
    :param bounds:
    :return: a tensor

    """

    if bounds is None:
        bounds = [None]*t.dim()
    elif not hasattr(bounds[0], '__len__'):
        bounds = [bounds for n in range(t.dim())]
    assert len(bounds) == t.dim()

    return sum([tn.partial(t, n, order=2, bounds=bounds[n]) for n in range(t.dim())])

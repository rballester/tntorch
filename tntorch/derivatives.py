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
        zero_slice = torch.zeros([core.shape[0], 1, core.shape[2]])
        if core.shape[1] == 1:
            return zero_slice
        step = (bounds[n][1] - bounds[n][0]) / (core.shape[1] - 1)
        return torch.cat(((core[:, 1:, :] - core[:, :-1, :]) / step, zero_slice), dim=1)
    cores = []
    idxs = []
    for n in range(t.ndim):
        if t.Us[n] is None:
            stack = [t.cores[n]]
        else:
            stack = [torch.einsum('ijk,aj->iak', (t.cores[n], t.Us[n]))]
        idx = torch.zeros([t.shape[n]])
        for o in range(1, max_order+1):
            stack.append(diff(stack[-1], n))
            idx = torch.cat((idx, torch.ones(stack[-1].shape[1])*o))
            if o == max_order:
                break
        cores.append(torch.cat(stack, dim=1))
        idxs.append(idx)
    d = tn.Tensor(cores, idxs=idxs)
    wm = tn.automata.weight_mask(t.ndim, order, nsymbols=max_order+1)
    if mask is not None:
        wm = tn.mask(wm, mask)
    result = tn.mask(d, wm)
    result.idxs = idxs
    return result


def partial(t, modes, order=1, bounds=None):
    """
    Compute a single partial derivative.

    :param t: a tensor
    :param modes: int or list of ints
    :param order: how many times to derive. Default is 1
    :param bounds: variable(s) range bounds (to compute the derivative step). If None (default), step 1 will be assumed
    :return: a tensor

    """

    if not hasattr(modes, '__len__'):
        modes = [modes]
    if bounds is None:
        bounds = [[0, t.shape[n]-1] for n in range(t.ndim)]
    if not hasattr(bounds[0], '__len__'):
        bounds = [bounds]

    t2 = t.clone()
    for i, mode in enumerate(modes):
        for o in range(1, order+1):
            step = (bounds[i][1] - bounds[i][0]) / (t.shape[mode]-1)
            if t2.Us[mode] is None:
                t2.cores[mode] = (t2.cores[mode][:, 1:, :] - t2.cores[mode][:, :-1, :]) / step
                t2.cores[mode] = torch.cat((t2.cores[mode], torch.zeros(t2.cores[mode].shape[0],
                                                                        1, t2.cores[mode].shape[2])), dim=1)
            else:
                t2.Us[mode] = (t2.Us[mode][1:, :] - t2.Us[mode][:-1, :]) / step
                t2.Us[mode] = torch.cat((t2.Us[mode], torch.zeros(1, t2.cores[mode].shape[1])), dim=0)
    return t2


def gradient(t, modes='all', bounds=None):
    """
    Compute the gradient of a tensor

    :param t: a tensor
    :param modes: an integer (or list of integers). Default is all modes
    :param bounds: a pair (or list of pairs) of reals, or None. The bounds for each variable

    :return: a tensor (or a list of tensors)

    """

    if modes == 'all':
        modes = range(t.ndim)
    if bounds is None:
        bounds = [[0, t.shape[mode]-1] for mode in modes]
    if not hasattr(bounds, '__len__'):
        bounds = [bounds]*len(modes)

    if not hasattr(modes, '__len__'):
        return partial(t, modes, bounds)
    else:
        return [partial(t, mode, order=1, bounds=b) for mode, b in zip(modes, bounds)]


def active_subspace(t):
    """
    Compute the main variational directions of a TT

    Reference: P. Constantine et al., "Discovering an Active Subspace in a
Single-Diode Solar Cell Model" (2017). Available: https://arxiv.org/pdf/1406.7607.pdf

    See also: https://github.com/paulcon/as-data-sets/blob/master/README.md

    :param t: a TT
    :return: (eigvals, eigvecs): an array and a matrix, encoding the eigenpairs in descending order

    """

    grad = tn.gradient(t, modes='all')

    M = torch.zeros(t.ndim, t.ndim)
    for i in range(t.ndim):
        for j in range(i, t.ndim):
            M[i, j] = tn.dot(grad[i], grad[j]) / t.size
            M[j, i] = M[i, j]

    w, v = torch.symeig(M, eigenvectors=True)
    idx = range(t.ndim-1, -1, -1)
    w = w[idx]
    v = v[:, idx]
    return w, v


def divergence(t, vector_mode=-1, bounds=None):
    """

    :param t: a tensor vector field
    :param vector_mode: which mode contains the vectors in the field ((x, y, z), for 3D). By default, the last mode
    :param bounds:
    :return: a scala

    """

    if bounds is None:
        bounds = [None for n in range(t.ndim)]
    elif not hasattr(bounds[0], '__len__'):
        bounds = [bounds for n in range(t.ndim)]
    assert len(bounds) == t.ndim
    if vector_mode < 0:
        vector_mode = t.ndim + vector_mode

    addends = []
    for m in range(t.shape[vector_mode]):
        idx = [slice(None)]*vector_mode + [m] + [slice(None)]*(t.ndim-vector_mode-1)
        addends.append(tn.partial(t[idx], m, order=2))
    return sum(addends)
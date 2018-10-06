import tntorch as tn
import torch
import numpy as np


def rand(*shape, **kwargs):
    """
    Generate a TT with random cores (and optionally factors), whose entries are uniform in [0, 1]

    :param shape: N ints
    :param ranks_tt: an integer or list of N-1 ints
    :param ranks_cp: an int or list. If a list, will be interleaved with ranks_tt
    :param ranks_tucker: an int or list
    :param requires_grad:
    :param device:

    :return:

    """

    return _create(torch.rand, *shape, **kwargs)


def rand_like(tensor, **kwargs):
    return _create(torch.rand, tensor.shape, **kwargs)


def randn(*shape, **kwargs):
    """
    Like `rand()`, but entries are distributed as a normal with mu=0, sigma=1
    """

    return _create(torch.randn, *shape, **kwargs)


def randn_like(tensor, **kwargs):
    return _create(torch.randn, tensor.shape, **kwargs)


def ones(*shape, **kwargs):
    return _create(torch.ones, *shape, ranks_tt=1, **kwargs)


def ones_like(tensor, **kwargs):
    return ones(tensor.shape, **kwargs)


def full(*shape, fill_value, **kwargs):
    return fill_value*tn.ones(*shape, **kwargs)


def full_like(tensor, fill_value, **kwargs):
    return tn.full(tensor.shape, fill_value=fill_value, **kwargs)


def zeros(*shape, **kwargs):
    return _create(torch.zeros, *shape, ranks_tt=1, **kwargs)


def zeros_like(tensor, **kwargs):
    return zeros(tensor.shape, **kwargs)


def _create(function, *shape, ranks_tt=None, ranks_cp=None, ranks_tucker=None, requires_grad=False, device=None):
    if hasattr(shape[0], '__len__'):
        shape = shape[0]
    N = len(shape)
    if not hasattr(ranks_tucker, "__len__"):
        ranks_tucker = [ranks_tucker for n in range(len(shape))]
    corespatials = []
    for n in range(len(shape)):
        if ranks_tucker[n] is None:
            corespatials.append(shape[n])
        else:
            corespatials.append(ranks_tucker[n])
    if ranks_tt is None and ranks_cp is None:
        if ranks_tucker is None:
            raise ValueError('Specify at least one of: ranks_tt ranks_cp, ranks_tucker')
        # We imitate a Tucker decomposition: we set full TT-ranks
        datashape = [corespatials[0], np.prod(corespatials) // corespatials[0]]
        ranks_tt = []
        for n in range(1, N):
            ranks_tt.append(min(datashape))
            datashape = [datashape[0] * corespatials[n], datashape[1] // corespatials[n]]
    if not hasattr(ranks_tt, "__len__"):
        ranks_tt = [ranks_tt]*(N-1)
    ranks_tt = [None] + list(ranks_tt) + [None]
    if not hasattr(ranks_cp, '__len__'):
        ranks_cp = [ranks_cp]*N
    coreranks = [r for r in ranks_tt]
    for n in range(N):
        if ranks_cp[n] is not None:
            if ranks_tt[n] is not None or ranks_tt[n+1] is not None:
                raise ValueError('The ranks_tt and ranks_cp provided are incompatible')
            coreranks[n] = ranks_cp[n]
            coreranks[n+1] = ranks_cp[n]
    assert len(coreranks) == N+1
    if coreranks[0] is None:
        coreranks[0] = 1
    if coreranks[-1] is None:
        coreranks[-1] = 1
    if coreranks.count(None) > 0:
        raise ValueError('One or more TT/CP ranks were not specified')
    assert len(ranks_tucker) == N

    cores = []
    Us = []
    for n in range(len(shape)):
        if ranks_tucker[n] is None:
            Us.append(None)
        else:
            Us.append(function([shape[n], ranks_tucker[n]], requires_grad=requires_grad, device=device))
        if ranks_cp[n] is None:
            cores.append(function([coreranks[n], corespatials[n], coreranks[n+1]], requires_grad=requires_grad, device=device))
        else:
            cores.append(function([corespatials[n], ranks_cp[n]], requires_grad=requires_grad, device=device))
    return tn.Tensor(cores, Us=Us)


def linspace(**kwargs):
    return tn.Tensor([torch.linspace(**kwargs)[None, :, None]])


def logspace(**kwargs):
    return tn.Tensor([torch.logspace(**kwargs)[None, :, None]])

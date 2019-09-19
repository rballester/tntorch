import tntorch as tn
import torch
import numpy as np


def eye(n, m=None, device=None, requires_grad=None):
    """
    Generates identity matrix like PyTorch's `eye()`.

    :param n: number of rows
    :param m: number of columns (default is n)

    :return: a 2D :class:`Tensor`
    """

    c1 = torch.eye(n, m)
    c2 = torch.eye(m, m)
    return tn.Tensor([c1[None, :, :], c2[:, :, None]], device=device, requires_grad=requires_grad)


def rand(*shape, **kwargs):
    """
    Generate a :class:`Tensor` with random cores (and optionally factors), whose entries are uniform in :math:`[0, 1]`.

    :Example:

    >>> tn.rand([10, 10], ranks_tt=3)  # Rank-3 TT tensor of shape 10x10

    :param shape: N ints (or a list of ints)
    :param ranks_tt: an integer or list of N-1 ints
    :param ranks_cp: an int or list. If a list, will be interleaved with ranks_tt
    :param ranks_tucker: an int or list
    :param requires_grad: default is False
    :param device:

    :return: a random tensor
    """

    return _create(torch.rand, *shape, **kwargs)


def rand_like(t, **kwargs):
    """
    Calls :meth:`rand()` with the shape of a given tensor.

    :param t: a tensor
    :param kwargs:

    :return: a :class:`Tensor`
    """

    return _create(torch.rand, t.shape, **kwargs)


def randn(*shape, **kwargs):
    """
    Like :meth:`rand()`, but entries are normally distributed with :math:`\\mu=0, \\sigma=1`.
    """

    return _create(torch.randn, *shape, **kwargs)


def randn_like(t, **kwargs):
    """
    Calls :meth:`randn()` with the shape of a given tensor.

    :param t: a tensor
    :param kwargs:

    :return: a :class:`Tensor`
    """

    return _create(torch.randn, t.shape, **kwargs)


def ones(*shape, **kwargs):
    """
    Generate a :class:`Tensor` filled with ones.

    :Example:

    >>> tn.ones(10)  # Vector of ones

    :param shape: N ints (or a list of ints)
    :param requires_grad:
    :param device:

    :return: a TT :class:`Tensor` of rank 1
    """

    return _create(torch.ones, *shape, ranks_tt=1, **kwargs)


def ones_like(t, **kwargs):
    """
    Calls :meth:`ones()` with the shape of a given tensor.

    :param t: a tensor
    :param kwargs:

    :return: a :class:`Tensor`
    """

    return ones(t.shape, **kwargs)


def full(shape, fill_value, **kwargs):
    """
    Generate a :class:`Tensor` filled with a constant.

    :param shape: list of ints
    :param fill_value: constant to fill the tensor with
    :param requires_grad:
    :param device:

    :return: a TT :class:`Tensor` of rank 1
    """

    return fill_value*tn.ones(*shape, **kwargs)


def full_like(t, fill_value, **kwargs):
    """
    Calls :meth:`full()` with the shape of a given tensor.

    :param t: a tensor
    :param kwargs:

    :return: a :class:`Tensor`
    """

    return tn.full(t.shape, fill_value=fill_value, **kwargs)


def zeros(*shape, **kwargs):
    """
    Generate a :class:`Tensor` filled with zeros.

    :param shape: N ints (or a list of ints)
    :param requires_grad:
    :param device:

    :return: a TT :class:`Tensor` of rank 1
    """

    return _create(torch.zeros, *shape, ranks_tt=1, **kwargs)


def zeros_like(t, **kwargs):
    """
    Calls :meth:`zeros()` with the shape of a given tensor.

    :param t: a tensor
    :param kwargs:

    :return: a :class:`Tensor`
    """

    return zeros(t.shape, **kwargs)


def gaussian(*shape, sigma_factor=0.2):
    """
    Create a multivariate Gaussian that is axis-aligned (i.e. with diagonal covariance matrix).

    :param shape: list of ints
    :param sigma_factor: a real (or list of reals) encoding the ratio sigma / shape. Default is 0.2, i.e. one fifth along each dimension

    :return: a :class:`Tensor` that sums to 1
    """

    if hasattr(shape[0], '__len__'):
        shape = shape[0]
    N = len(shape)
    if not hasattr(sigma_factor, '__len__'):
        sigma_factor = [sigma_factor]*N

    cores = [torch.ones(1, 1, 1) for n in range(N)]
    Us = []
    for n in range(N):
        sigma = sigma_factor[n] * shape[n]
        if shape[n] == 1:
            x = torch.tensor([0])
        else:
            x = torch.linspace(-shape[n] / 2, shape[n] / 2, shape[n])
        U = torch.exp(-x**2 / (2*sigma**2))
        U = U[:, None] / torch.sum(U)
        Us.append(U)
    return tn.Tensor(cores, Us)


def gaussian_like(tensor, **kwargs):
    """
    Calls :meth:`gaussian()` with the shape of a given tensor.

    :param t: a tensor
    :param kwargs:

    :return: a :class:`Tensor`
    """

    return gaussian(tensor.shape, **kwargs)


def _create(function, *shape, ranks_tt=None, ranks_cp=None, ranks_tucker=None, requires_grad=False, device=None, batch=False):
    if hasattr(shape[0], '__len__'):
        shape = shape[0]

    if batch:
        N = len(shape) - 1
    else:
        N = len(shape)
    if not hasattr(ranks_tucker, "__len__"):
        ranks_tucker = [ranks_tucker for n in range(N)]
    corespatials = []
    if batch:
        corespatials.append(shape[0])
    for n in range(N):
        if ranks_tucker[n] is None:
            if batch:
                corespatials.append(shape[n + 1])
            else:
                corespatials.append(shape[n])
        else:
            corespatials.append(ranks_tucker[n])
    if ranks_tt is None and ranks_cp is None:
        if ranks_tucker is None:
            raise ValueError('Specify at least one of: ranks_tt ranks_cp, ranks_tucker')

        # We imitate a Tucker decomposition: we set full TT-ranks
        if batch:
            datashape = [corespatials[0], corespatials[1], np.prod(corespatials[1:]) // corespatials[1]]
        else:
            datashape = [corespatials[0], np.prod(corespatials) // corespatials[0]]

        ranks_tt = []

        for n in range(1, N):
            if batch:
                ranks_tt.append(min(datashape[1:]))
                datashape = [datashape[0], datashape[1] * corespatials[n + 1], datashape[2] // corespatials[n + 1]]
            else:
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
    for n in range(N):
        if ranks_tucker[n] is None:
            Us.append(None)
        else:
            if batch:
                Us.append(function([shape[0], shape[n + 1], ranks_tucker[n]], requires_grad=requires_grad, device=device))
            else:
                Us.append(function([shape[n], ranks_tucker[n]], requires_grad=requires_grad, device=device))
        if ranks_cp[n] is None:
            if batch:
                cores.append(function([shape[0], coreranks[n], corespatials[n + 1], coreranks[n+1]], requires_grad=requires_grad, device=device))
            else:
                cores.append(function([coreranks[n], corespatials[n], coreranks[n+1]], requires_grad=requires_grad, device=device))
        else:
            if batch:
                cores.append(function([shape[0], corespatials[n + 1], ranks_cp[n]], requires_grad=requires_grad, device=device))
            else:
                cores.append(function([corespatials[n], ranks_cp[n]], requires_grad=requires_grad, device=device))
    return tn.Tensor(cores, Us=Us, batch=batch)


def arange(*args, **kwargs):
    """
    Creates a 1D :class:`Tensor` (see PyTorch's `arange`).

    :param args:
    :param kwargs:

    :return: a 1D :class:`Tensor`
    """

    return tn.Tensor([torch.arange(*args, dtype=torch.get_default_dtype(), **kwargs)[None, :, None]])


def linspace(*args, **kwargs):
    """
    Creates a 1D :class:`Tensor` with evenly spaced values (see PyTorch's `linspace`).

    :param args:
    :param kwargs:

    :return: a 1D :class:`Tensor`
    """

    return tn.Tensor([torch.linspace(*args, **kwargs)[None, :, None]])


def logspace(*args, **kwargs):
    """
    Creates a 1D :class:`Tensor` with logarithmically spaced values (see PyTorch's `logspace`).

    :param args:
    :param kwargs:

    :return: a 1D :class:`Tensor`
    """

    return tn.Tensor([torch.logspace(*args, **kwargs)[None, :, None]])

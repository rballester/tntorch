import copy
import numpy as np
import torch
import tntorch as tn


def anova_decomposition(t, marginals=None):
    """
    Compute an extended tensor that contains all terms of the ANOVA decomposition for a given tensor.

    Reference: R. Ballester-Ripoll, E. G. Paredes, and R. Pajarola: `"Sobol Tensor Trains for Global Sensitivity Analysis" (2017) <https://www.sciencedirect.com/science/article/pii/S0951832018303132?dgcid=rss_sd_all>`_

    :param t: ND input tensor
    :param marginals: list of N vectors, each containing the PMF for each variable (use None for uniform distributions)
    :return: a :class:`Tensor`
    """

    marginals = copy.deepcopy(marginals)
    if marginals is None:
        marginals = [None] * t.dim()
    for n in range(t.dim()):
        if marginals[n] is None:
            marginals[n] = torch.ones([t.shape[n]]) / float(t.shape[n])
    cores = [c.clone() for c in t.cores]
    Us = []
    idxs = []
    for n in range(t.dim()):
        if t.Us[n] is None:
            U = torch.eye(t.shape[n])
        else:
            U = t.Us[n]
        expected = torch.sum(U * (marginals[n][:, None] / torch.sum(marginals[n])), dim=0, keepdim=True)
        Us.append(torch.cat((expected, U-expected), dim=0))
        idxs.append([0] + [1]*t.shape[n])
    return tn.Tensor(cores, Us, idxs=idxs)


def undo_anova_decomposition(a):
    """
    Undo the transformation done by :func:`anova_decomposition()`.

    :param a: a :class:`Tensor` obtained with :func:`anova_decomposition()`

    :return: a :class:`Tensor` t that has `a` as its ANOVA tensor
    """

    cores = []
    Us = []
    for n in range(a.dim()):
        if a.Us[n] is None:
            cores.append(a.cores[n][..., 1:, :] + a.cores[n][..., 0:1, :])
            Us.append(None)
        else:
            cores.append(a.cores[n].clone())
            Us.append(a.Us[n][1:, :] + a.Us[n][0:1, :])
    return tn.Tensor(cores, Us=Us)


def truncate_anova(t, mask, keepdim=False, marginals=None):
    """
    Given a tensor and a mask, return the function that results after deleting all ANOVA terms that do not satisfy the
    mask.

    :Example:

    >>> t = ...  # an ND tensor
    >>> x = tn.symbols(t.dim())[0]
    >>> t2 = tn.truncate_anova(t, mask=tn.only(x), keepdim=False)  # This tensor will depend on one variable only

    :param t: an N-dimensional :class:`Tensor`
    :param mask: an N-dimensional mask
    :param keepdim: if True, all dummy dimensions will be preserved, otherwise they will disappear. Default is False
    :param marginals: see :func:`anova_decomposition()`. Defaults to uniform marginals

    :return: a :class:`Tensor`
    """

    t = tn.undo_anova_decomposition(tn.mask(tn.anova_decomposition(t, marginals=marginals), mask=mask))
    if not keepdim:
        N = t.dim()
        affecting = torch.sum(tn.accepted_inputs(mask).double(), dim=0)
        slices = [0 for n in range(N)]
        for i in np.where(affecting)[0]:
            slices[int(i)] = slice(None)
        t = t[tuple(slices)]
    return t


def sobol(t, mask, marginals=None, normalize=True):
    """
    Compute Sobol indices (as given by a certain mask) for a tensor and independently distributed input variables.

    Reference: R. Ballester-Ripoll, E. G. Paredes, and R. Pajarola: `"Sobol Tensor Trains for Global Sensitivity Analysis" (2017) <https://www.sciencedirect.com/science/article/pii/S0951832018303132?dgcid=rss_sd_all>`_

    :param t: an N-dimensional :class:`Tensor`
    :param mask: an N-dimensional mask
    :param marginals: a list of N vectors (will be normalized if not summing to 1). If None (default), uniform distributions are assumed for all variables
    :param normalize: whether to normalize indices by the total variance of the model (True by default)

    :return: a scalar >= 0
    """

    if marginals is None:
        marginals = [None] * t.dim()

    a = tn.anova_decomposition(t, marginals)
    a -= tn.Tensor([torch.cat((torch.ones(1, 1, 1),
                               torch.zeros(1, sh-1, 1)), dim=1)
                    for sh in a.shape])*a[(0,)*t.dim()]  # Set empty tuple to 0
    am = a.clone()
    for n in range(t.dim()):
        if marginals[n] is None:
            m = torch.ones([t.shape[n]])
        else:
            m = marginals[n]
        m /= torch.sum(m)  # Make sure each marginal sums to 1
        if am.Us[n] is None:
            if am.cores[n].dim() == 3:
                am.cores[n][:, 1:, :] *= m[None, :, None]
            else:
                am.cores[n][1:, :] *= m[:, None]
        else:
            am.Us[n][1:, :] *= m[:, None]
    am_masked = tn.mask(am, mask)
    if am_masked.cores[-1].shape[-1] > 1:
        am_masked.cores.append(torch.eye(am_masked.cores[-1].shape[-1])[:, :, None])
        am_masked.Us.append(None)

    if normalize:
        return tn.dot(a, am_masked) / tn.dot(a, am)
    else:
        return tn.dot(a, am_masked)


def mean_dimension(t, mask=None, marginals=None):
    """
    Computes the mean dimension of a given tensor with given marginal distributions. This quantity measures how well the
    represented function can be expressed as a sum of low-parametric functions. For example, mean dimension 1 (the
    lowest possible value) means that it is a purely additive function: :math:`f(x_1, ..., x_N) = f_1(x_1) + ... + f_N(x_N)`.

    Assumption: the input variables :math:`x_n` are independently distributed.

    References:

    - R. E. Caflisch, W. J. Morokoff, and A. B. Owen: `"Valuation of Mortgage Backed Securities Using Brownian Bridges to Reduce Effective Dimension" (1997) <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.36.3160>`_

    -  R. Ballester-Ripoll, E. G. Paredes, and R. Pajarola: `"Tensor Algorithms for Advanced Sensitivity Metrics" (2017) <https://epubs.siam.org/doi/10.1137/17M1160252>`_

    :param t: an N-dimensional :class:`Tensor`
    :param marginals: a list of N vectors (will be normalized if not summing to 1). If None (default), uniform distributions are assumed for all variables

    :return: a scalar >= 1
    """

    if mask is None:
        return tn.sobol(t, tn.weight(t.dim()), marginals=marginals)
    else:
        return tn.sobol(t, tn.mask(tn.weight(t.dim()), mask), marginals=marginals) / tn.sobol(t, mask, marginals=marginals)


def dimension_distribution(t, mask=None, order=None, marginals=None):
    """
    Computes the dimension distribution of an ND tensor.

    :param t: ND input :class:`Tensor`
    :param mask: an optional mask :class:`Tensor` to restrict to
    :param order: int, compute only this many order contributions. By default, all N are returned
    :param marginals: PMFs for input variables. By default, uniform distributions

    :return: a PyTorch vector containing N elements
    """

    if order is None:
        order = t.dim()
    if mask is None:
        return tn.sobol(t, tn.weight_one_hot(t.dim(), order+1), marginals=marginals).torch()[1:]
    else:
        mask2 = tn.mask(tn.weight_one_hot(t.dim(), order+1), mask)
        return tn.sobol(t, mask2, marginals=marginals).torch()[1:] / tn.sobol(t, mask, marginals=marginals)

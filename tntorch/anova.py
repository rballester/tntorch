import copy
import torch
import tntorch as tn


def anova_decomposition(t, marginals=None):
    """
    Compute an extended tensor that contains all terms of the ANOVA decomposition for a given tensor T.

    Reference: "Sobol Tensor Trains for Global Sensitivity Analysis", Ballester-Ripoll, Paredes and Pajarola (2017)

    :param t:
    :param marginals:
    :return:
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
    Undo the transformation done by `anova_decomposition()`

    :param a: a tensor obtained with `ànova_decomposition()`
    :return: a tensor t that has `a` as its ANOVA tensor

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


def sobol(t, mask, marginals=None):
    """
    Compute Sobol indices as given by a certain mask

    :param t: an N-dimensional tensor
    :param mask: an N-dimensional mask
    :param marginals: a list of N vector tensors (will be normalized if not summing to 1). If None (default), uniform
    distributions are assumed for all variables
    :return: a scalar

    """

    if marginals is None:
        marginals = [None] * t.dim()

    a = tn.anova_decomposition(t, marginals)
    a -= tn.Tensor([torch.cat((torch.ones(1, 1, 1),
                               torch.zeros(1, sh-1, 1)), dim=1)
                    for sh in a.shape])*a[[0]*t.dim()]  # Set empty tuple to 0
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
    return tn.dot(a, am_masked) / tn.dot(a, am)


def mean_dimension(t, marginals=None):
    """
    Computes the mean dimension of a given tensor with given marginal distributions. This quantity measures how well the
    represented function can be expressed as a sum of low-parametric functions. For example, mean dimension 1 (the
    lowest possible value) means that it is a purely additive function: f(x_1, ..., x_N) = f_1(x_1) + ... + f_N(x_N).

    Assumption: the input variables x_n are independently distributed.

    References:
    - R. E. Caflisch, W. J. Morokoff and A. B. Owen: "Valuation of Mortgage Backed Securities
        Using Brownian Bridges to Reduce Effective Dimension (1997)
    -  R. Ballester-Ripoll, E. G. Paredes and R. Pajarola: "Tensor Approximation of Advanced Metrics
     for Sensitivity Analysis" (2017)

    :param t: an N-dimensional tensor
    :param marginals: a list of N vector tensors (will be normalized if not summing to 1). If None (default), uniform
    distributions are assumed for all variables
    :return: a scalar >= 1

    """

    return tn.sobol(t, tn.weight(t.dim()), marginals=marginals)


def dimension_distribution(t, order=None, marginals=None):
    """
    Computes the dimension distribution of an N-D tensor.

    :param t: input tensor
    :param order: compute only this many order contributions. By default, all N are returned
    :param marginals: PMFs for input variables. By default, uniform distributions
    :return: a torch vector containing N elements

    """

    if order is None:
        order = t.dim()
    return tn.sobol(t, tn.weight_one_hot(t.dim(), order+1), marginals=marginals).full()[1:]

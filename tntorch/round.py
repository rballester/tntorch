import tntorch as tn
import torch
import numpy as np
import time


def round_tt(t, **kwargs):
    """
    Copies and rounds a tensor (see :meth:`tensor.Tensor.round_tt()`.

    :param t:
    :param kwargs:

    :return: a rounded copy of `t`
    """

    t2 = t.clone()
    t2.round_tt(**kwargs)
    return t2


def round_tucker(t, **kwargs):
    """
    Copies and rounds a tensor (see :meth:`tensor.Tensor.round_tucker()`.

    :param t:
    :param kwargs:

    :return: a rounded copy of `t`
    """

    t2 = t.clone()
    t2.round_tucker(**kwargs)
    return t2


def round(t, **kwargs):
    """
    Copies and rounds a tensor (see :meth:`tensor.Tensor.round()`.

    :param t:
    :param kwargs:

    :return: a rounded copy of `t`
    """

    t2 = t.clone()
    t2.round(**kwargs)
    return t2


def truncated_svd(M, delta=None, eps=None, rmax=None, left_ortho=True, algorithm='svd', verbose=False):
    """
    Decomposes a matrix M (size (m x n) in two factors U and V (sizes m x r and r x n) with bounded error (or given r).

    :param M: a matrix
    :param delta: if provided, maximum error norm
    :param eps: if provided, maximum relative error
    :param rmax: optionally, maximum r
    :param left_ortho: if True (default), U will be orthonormal. If False, V will
    :param algorithm: 'svd' (default) or 'eig'. The latter is often faster, but less accurate
    :param verbose:

    :return: U, V
    """

    if delta is not None and eps is not None:
        raise ValueError('Provide either `delta` or `eps`')
    if delta is None and eps is not None:
        delta = eps*torch.norm(M).item()
    if delta is None and eps is None:
        delta = 0
    if rmax is None:
        rmax = np.iinfo(np.int32).max
    assert rmax >= 1
    assert algorithm in ('svd', 'eig')

    if algorithm == 'svd':
        start = time.time()
        svd = torch.svd(M)[:2]
        singular_vectors = 'left'
        if verbose:
            print('Time (SVD):', time.time() - start)
    else:
        start = time.time()
        if M.shape[0] <= M.shape[1]:
            gram = torch.mm(M, M.permute(1, 0))
            singular_vectors = 'left'
        else:
            gram = torch.mm(M.permute(1, 0), M)
            singular_vectors = 'right'
        if verbose:
            print('Time (gram):', time.time() - start)
        start = time.time()
        w, v = torch.symeig(gram, eigenvectors=True)
        if verbose:
            print('Time (symmetric EIG):', time.time() - start)
        w[w < 0] = 0
        w = torch.sqrt(w)
        svd = [v, w]
        # Sort eigenvalues and eigenvectors in decreasing importance
        reverse = np.arange(len(svd[1])-1, -1, -1)
        idx = np.argsort(svd[1])[reverse]
        svd[0] = svd[0][:, idx]
        svd[1] = svd[1][idx]

    if svd[1][0] < 1e-13:  # Special case: M = zero -> rank is 1
        return torch.zeros([M.shape[0], 1]), torch.zeros([1, M.shape[1]])

    S = svd[1]**2
    reverse = np.arange(len(S)-1, -1, -1)
    where = np.where((torch.cumsum(S[reverse], dim=0) <= delta**2).to('cpu'))[0]
    if len(where) == 0:
        rank = max(1, int(min(rmax, len(S))))
    else:
        rank = max(1, int(min(rmax, len(S) - 1 - where[-1])))
    left = svd[0]
    left = left[:, :rank]

    start = time.time()
    if singular_vectors == 'left':
        if left_ortho:
            M2 = torch.mm(left.permute(1, 0), M)
        else:
            M2 = torch.mm((1. / svd[1][:rank])[:, None]*left.permute(1, 0), M)
            left = left*svd[1][:rank]
    else:
        if left_ortho:
            M2 = torch.mm(M, (left * (1. / svd[1][:rank])[None, :]))
            left, M2 = M2, torch.mm(left, (torch.diag(svd[1][:rank]))).permute(1, 0)
        else:
            M2 = torch.mm(M, left)
            left, M2 = M2, left.permute(1, 0)
    if verbose:
        print('Time (product):', time.time() - start)

    return left, M2

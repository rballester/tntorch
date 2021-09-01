import torch
import time
from typing import Optional

def round_tt(t, **kwargs):
    """
    Copies and rounds a tensor (see :meth:`tensor.Tensor.round_tt()`.

    :param t: input :class:`Tensor`
    :param kwargs:

    :return: a rounded copy of `t`
    """

    t2 = t.clone()
    t2.round_tt(**kwargs)
    return t2


def round_tucker(t, **kwargs):
    """
    Copies and rounds a tensor (see :meth:`tensor.Tensor.round_tucker()`.

    :param t: input :class:`Tensor`
    :param kwargs:

    :return: a rounded copy of `t`
    """

    t2 = t.clone()
    t2.round_tucker(**kwargs)
    return t2


def round(t, **kwargs):
    """
    Copies and rounds a tensor (see :meth:`tensor.Tensor.round()`.

    :param t: input :class:`Tensor`
    :param kwargs:

    :return: a rounded copy of `t`
    """

    t2 = t.clone()
    t2.round(**kwargs)
    return t2


def truncated_svd(
    M: torch.tensor,
    delta: Optional[float] = None,
    eps: Optional[float] = None,
    rmax: Optional[int] = None,
    left_ortho: Optional[bool] = True,
    algorithm: Optional[str] = 'svd',
    verbose: Optional[bool] = False,
    batch: Optional[bool] = False):

    """
    Decomposes a matrix M (size (m x n) in two factors U and V (sizes m x r and r x n) with bounded error (or given r).

    :param M: a matrix
    :param delta: if provided, maximum error norm
    :param eps: if provided, maximum relative error
    :param rmax: optionally, maximum r
    :param left_ortho: if True (default), U will be orthonormal. If False, V will
    :param algorithm: 'svd' (default) or 'eig'. The latter is often faster, but less accurate
    :param verbose: Boolean
    :param batch: Boolean

    :return: U, V
    """
    
    if delta is not None and eps is not None:
        raise ValueError('Provide either `delta` or `eps`')
    if delta is None and eps is not None:
        delta = eps*torch.norm(M).item()
    if delta is None and eps is None:
        delta = 0
    if rmax is None:
        rmax = torch.iinfo(torch.int32).max
    assert rmax >= 1
    assert algorithm in ('svd', 'eig')

    if batch:
        batch_size = M.shape[0]
        dims_permute = [0, 2, 1]
    else:
        dims_permute = [1, 0]

    if algorithm == 'svd':
        start = time.time()
        svd = torch.svd(M)[:2]

        singular_vectors = 'left'
        if verbose:
            print('Time (SVD):', time.time() - start)
    else:
        start = time.time()

        if M.shape[-2] <= M.shape[-1]:
            gram = M @ M.permute(dims_permute)
            singular_vectors = 'left'
        else:
            gram = M.permute(dims_permute) @ M
            singular_vectors = 'right'

        if verbose:
            print('Time (gram):', time.time() - start)

        start = time.time()
        w, v = torch.linalg.eigh(gram)
        if verbose:
            print('Time (symmetric EIG):', time.time() - start)
        w = torch.where(w < 0, torch.zeros_like(w) + 1e-8, w)
        w = torch.sqrt(w)
        svd = [v, w]
        # Sort eigenvalues and eigenvectors in decreasing importance
        if batch:
            reverse = torch.arange(len(svd[1][0]) - 1, -1, -1)
            idx = torch.argsort(svd[1])[:, reverse]
            svd[0] = torch.cat([svd[0][i, ..., idx[i]][None, ...] for i in range(len(idx))])
            svd[1] = torch.cat([svd[1][i, ..., idx[i]][None, ...] for i in range(len(idx))])
        else:
            reverse = torch.arange(len(svd[1]) - 1, -1, -1)
            idx = torch.argsort(svd[1])[reverse]
            svd[0] = svd[0][..., idx]
            svd[1] = svd[1][..., idx]

     # NOTE: Special case: M = zero -> rank is 1
    if batch:
        if svd[1].max() < 1e-13:
            return torch.zeros([batch_size, M.shape[1], 1]), torch.zeros([batch_size, 1, M.shape[2]])
    else:
        if svd[1][0] < 1e-13:
            return torch.zeros([M.shape[0], 1]), torch.zeros([1, M.shape[1]])

    S = svd[1]**2

    if batch:
        rank = max(1, int(min(rmax, len(S[0]))))
    else:
        reverse = torch.arange(len(S) - 1, -1, -1)
        where = torch.where((torch.cumsum(S[reverse], dim=0) <= delta**2))[0]

        if len(where) == 0:
            rank = max(1, int(min(rmax, len(S))))
        else:
            rank = max(1, int(min(rmax, len(S) - 1 - where[-1])))

    left = svd[0]
    left = left[..., :rank]

    start = time.time()
    if singular_vectors == 'left':
        if left_ortho:
            M2 = left.permute(dims_permute) @ M
        else:
            M2 = (1. / svd[1][..., :rank])[..., None] * left.permute(dims_permute) @ M
            if batch:
                left = torch.einsum('bij,bj->bij', left, svd[1][..., :rank])
            else:
                left = left * svd[1][:rank]
    else:
        if left_ortho:
            M2 = M @ (left * (1. / svd[1][..., :rank])[..., None, :])
            left, M2 = M2, (left @ (torch.diag(svd[1][..., :rank]))).permute(dims_permute)
        else:
            M2 = M @ left

            left, M2 = M2, left.permute(dims_permute)
    
    if verbose:
        print('Time (product):', time.time() - start)

    return left, M2

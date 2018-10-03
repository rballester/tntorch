import tntorch as tn
import torch
import numpy as np
import time
import scipy.fftpack


"""
Initializations
"""


def rand(shape, **kwargs):
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

    return _create(torch.rand, shape, **kwargs)


def rand_like(tensor, **kwargs):
    return _create(torch.rand, tensor.shape, **kwargs)


def randn(shape, **kwargs):
    """
    Like `rand()`, but entries are distributed as a normal with mu=0, sigma=1
    """

    return _create(torch.randn, shape, **kwargs)


def randn_like(tensor, **kwargs):
    return _create(torch.randn, tensor.shape, **kwargs)


def ones(shape, **kwargs):
    return _create(torch.ones, shape, ranks_tt=1, **kwargs)


def ones_like(tensor, **kwargs):
    return ones(tensor.shape, **kwargs)


def zeros(shape, **kwargs):
    return _create(torch.zeros, shape, ranks_tt=1, **kwargs)


def zeros_like(tensor, **kwargs):
    return zeros(tensor.shape, **kwargs)


def _create(function, shape, ranks_tt=None, ranks_cp=None, ranks_tucker=None, requires_grad=False, device=None):
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


"""
Array-like manipulations
"""


def squeeze(t, modes=None):
    """
    Removes singleton dimensions

    :param modes: which modes to delete. By default, all that have size 1
    :return: Another TT tensor, without dummy (singleton) indices
    """

    if modes is None:
        modes = np.where(t.shape == 1)[0]
    assert np.all(np.array(t.shape)[modes] == 1)

    idx = [slice(None) for n in range(t.dim())]
    for m in modes:
        idx[m] = 0
    return t[idx]


def cat(ts, mode):
    """
    Concatenate two or more tensors along a given mode, similarly to PyTorch's `cat()`.

    :param ts: a list of tensors
    :param mode: an int
    :return: a tensor of the same shape as all tensors in the list, except along `mode` where it has the sum of shapes

    """

    if len(ts) == 1:
        return ts[0].clone()
    if any([any([t.shape[n] != ts[0].shape[n] for n in np.delete(range(ts[0].dim()), mode)]) for t in ts[1:]]):
        raise ValueError('To concatenate tensors, all must have the same shape along all but the given mode')

    shapes = np.array([t.shape[mode] for t in ts])
    sumshapes = np.concatenate([np.array([0]), np.cumsum(shapes)])
    for i in range(len(ts)):
        t = ts[i].clone()
        if t.Us[mode] is None:
            if t.cores[mode].dim() == 2:
                t.cores[mode] = torch.zeros(sumshapes[-1], t.cores[mode].shape[-1])
            else:
                t.cores[mode] = torch.zeros(t.cores[mode].shape[0], sumshapes[-1], t.cores[mode].shape[-1])
            t.cores[mode][..., sumshapes[i]:sumshapes[i+1], :] += ts[i].cores[mode]
        else:
            t.Us[mode] = torch.zeros(sumshapes[-1], t.Us[mode].shape[-1])
            t.Us[mode][sumshapes[i]:sumshapes[i+1], :] += ts[i].Us[mode]
        if i == 0:
            result = t
        else:
            result += t
    return result


def transpose(t):
    """
    Inverts the dimension order of a tensor, e.g. I1 x I2 x I3 becomes I3 x I2 x I1.

    :param t: a tensor
    :return: another tensor, indexed by dimensions in inverse order

    """

    cores = []
    Us = []
    idxs = []
    for n in range(t.dim()-1, -1, -1):
        if t.cores[n].dim() == 3:
            cores.append(t.cores[n].permute(2, 1, 0))
        else:
            cores.append(t.cores[n])
        if t.Us[n] is None:
            Us.append(None)
        else:
            Us.append(t.Us[n].clone())
        if t.idxs[n] is None:
            idxs.append(None)
        else:
            idxs.append(t.idxs[n].clone())
    return tn.Tensor(cores, Us, idxs)


def meshgrid(axes):
    """
    See NumPy's or PyTorch's `meshgrid()`.

    :param axes: a list of N ints or torch vectors
    :return: N tensors

    """

    if not hasattr(axes, '__len__'):
        axes = [axes]
    axes = axes.copy()
    N = len(axes)
    for n in range(N):
        if not hasattr(axes[n], '__len__'):
            axes[n] = torch.arange(axes[n]).double()

    tensors = []
    for n in range(N):
        cores = [torch.ones(1, len(ax), 1) for ax in axes]
        cores[n] = torch.Tensor(axes[n])[None, :, None]
        tensors.append(tn.Tensor(cores))
    return tensors


def flip(t, dims):
    """
    Reverses the order of a tensor along one or several dimensions; see NumPy's or PyTorch's `flip()`.

    :param t: a tensor
    :param dims: an int or list of ints
    :return: another tensor of the same shape

    """

    if not hasattr(dims, '__len__'):
        dims = [dims]

    shape = t.shape
    result = t.clone()
    for d in dims:
        print(d, shape)
        idx = np.arange(shape[d]-1, -1, -1)
        if result.Us[d] is not None:
            result.Us[d] = result.Us[d][idx, :]
        else:
            print(idx, result.cores[d].shape)
            result.cores[d] = result.cores[d][..., idx, :]
    return result


def unbind(t, dim):
    """
    Slices a tensor along a dimension and returns the slices as a sequence, like PyTorch's `unbind()`.

    :param t: a tensor
    :param dim: an int
    :return: a list of tensors, as many as `t.shape[dim]`

    """

    if dim < 0:
        dim += t.dim()
    return [t[[slice(None)]*dim + [sl] + [slice(None)]*(t.dim()-1-dim)] for sl in range(t.shape[dim])]


def unfolding(data, n):
    return data.permute([n] + list(range(n)) + list(range(n + 1, data.dim()))).reshape([data.shape[n], -1])


def right_unfolding(core):
    return core.reshape([core.shape[0], -1])


def left_unfolding(core):
    return core.reshape([-1, core.shape[-1]])


"""
Rounding
"""


def round_tt(t, **kwargs):
    t2 = t.clone()
    t2.round_tt(**kwargs)
    return t2


def round_tucker(t, **kwargs):
    t2 = t.clone()
    t2.round_tucker(**kwargs)
    return t2


def round(t, **kwargs):
    t2 = t.clone()
    t2.round(**kwargs)
    return t2


def truncated_svd(M, delta=None, eps=None, rmax=None, left_ortho=True, algorithm='svd', verbose=False):
    """
    Decompose a matrix M (size (m x n) in two factors U and V (sizes m x r and r x n) with bounded error (or given r)

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
        delta = eps*torch.norm(M)
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
    where = np.where(np.cumsum(S[reverse]) <= delta**2)[0]
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


"""
Multilinear algebra
"""


def sum(t, modes=None, keepdims=False):
    """
    Compute the sum of a tensor along all (or some) dimensions.

    :param t: a tensor
    :param modes: an int or list of ints. By default, all modes will be summed
    :param keepdims: if True, summed dimensions will be kept as singletons. Default is False
    :return: a scalar

    """

    if modes is None:
        modes = np.arange(t.dim())
    if not hasattr(modes, '__len__'):
        modes = [modes]
    cores = []
    Us = []
    for n in range(t.dim()):
        if n in modes:
            if t.Us[n] is None:
                cores.append(torch.sum(t.cores[n], dim=-2, keepdim=True))
                Us.append(None)
            else:
                cores.append(t.cores[n].clone())
                Us.append(torch.sum(t.Us[n], dim=0, keepdim=True))
        else:
            cores.append(t.cores[n].clone())
            Us.append(t.Us[n].clone())
    result = tn.Tensor(cores, Us=Us)
    if keepdims:
        return result
    else:
        return tn.squeeze(result, modes)


def ttm(t, U, mode, transpose=False):
    """
    Tensor-times-matrix (TTM) along one or several modes

    :param U: one or several factors
    :param mode: one or several modes (may be vectors or matrices)
    :param transpose: if False (default) the contraction is performed
     along U's rows, else along its columns
    :return: transformed TT

    """

    if not isinstance(U, (list, tuple)):
        U = [U]
    if not hasattr(mode, '__len__'):
        mode = [mode]

    cores = []
    Us = []
    for n in range(t.dim()):
        if n in mode:
            if transpose:
                factor = U[mode.index(n)].t
            else:
                factor = U[mode.index(n)]
            if factor.dim() == 1:
                factor = factor[None, :]
            if t.Us[mode[n]] is None:
                if t.cores[mode[n]].dim() == 3:
                    cores.append(torch.einsum('iak,ja->ijk', (t.cores[mode[n]], factor)))
                else:
                    cores.append(torch.einsum('ai,ja->ji', (t.cores[mode[n]], factor)))
                Us.append(None)
            else:
                cores.append(t.cores[n].clone())
                Us.append(torch.matmul(factor, t.Us[mode[n]]))
        else:
            cores.append(t.cores[n].clone())
            if t.Us[n] is None:
                Us.append(None)
            else:
                Us.append(t.Us[n].clone())
    return tn.Tensor(cores, Us=Us, idxs=t.idxs)


def cumsum(t, modes):
    """
    Computes the cumulative sum of a tensor along one or several modes, similarly to PyTorch's `cumsum()`.

    :param t: a tensor
    :param modes: an int or list of ints
    :return: a tensor of the same shape

    """

    if not hasattr(modes, '__len__'):
        modes = [modes]

    t = t.clone()
    for n in modes:
        if t.Us[n] is None:
            t.cores[n] = torch.cumsum(t.cores[n], dim=-2)
        else:
            t.Us[n] = torch.cumsum(t.Us[n], dim=0)
    return t


"""
Miscellaneous
"""


def mask(t, mask):
    """
    Masks a tensor.

    It's basically an element-wise product, but this function makes sure slices are matched according to their "meaning"
    (as annotated by the tensor's `idx` field, if available)

    :param t: a tensor
    :param mask: a mask tensor
    :return: a tensor

    """

    if not hasattr(t, 'idxs'):
        idxs = [np.arange(sh) for sh in t.shape]
    else:
        idxs = t.idxs
    cores = []
    Us = []
    for n in range(t.dim()):
        idx = np.array(idxs[n])
        idx[idx >= mask.shape[n]] = mask.shape[n]-1  # Clamp
        if mask.Us[n] is None:
            cores.append(mask.cores[n][..., idx, :])
            Us.append(None)
        else:
            cores.append(mask.cores[n])
            Us.append(mask.Us[n][idx, :])
    mask = tn.Tensor(cores, Us)
    return t*mask


def sample(t, P=1):  # TODO
    """
    Generate P points (with replacement) from a joint PDF distribution represented by this tensor.

    The tensor does not have to sum 1.

    :param P: how many samples to draw (default: 1)
    :return Xs: a matrix of size P x N

    """

    def from_matrix(M):
        """
        Treat each row of a matrix M as a pdf and select a column per row according to it
        """

        M /= torch.sum(M, dim=1)[:, None]  # Normalize row-wise
        M = np.hstack([np.zeros([M.shape[0], 1]), M])
        M = np.cumsum(M, axis=1)
        thresh = np.random.rand(M.shape[0])
        M -= thresh[:, np.newaxis]
        shiftand = np.logical_and(M[:, :-1] <= 0, M[:, 1:] > 0)  # Find where the sign switches
        return np.where(shiftand)[1]

    Xs = torch.zeros([P, t.dim()])
    rights = [torch.ones(1)]
    for core in t.cores[::-1]:
        rights.append(torch.matmul(torch.sum(core, dim=1), rights[-1]))
    rights = rights[::-1]
    lefts = torch.ones([P, 1])

    for mu in range(t.dim()):
        fiber = torch.einsum('ijk,k->ij', (t.cores[mu], rights[mu + 1]))
        per_point = torch.einsum('ij,jk->ik', (lefts, fiber))
        rows = from_matrix(per_point)
        Xs[:, mu] = rows
        lefts = torch.einsum('ij,jik->ik', (lefts, t.cores[mu][:, rows, :]))

    return Xs


def hash(t):
    """
    Obtains an integer number that depends on the tensor entries (not on its internal compressed representation).

    We compute it as <T, W>, where W is a rank-1 tensor of weights selected at random (always the same seed).

    :return: an integer

    """

    gen = torch.Generator()
    gen.manual_seed(0)
    cores = [torch.ones(1, 1, 1) for n in range(t.dim())]
    Us = [torch.rand([sh, 1], generator=gen) for sh in t.shape]
    w = tn.Tensor(cores, Us)
    return t.dot(w)


def generate_basis(name, shape, orthonormal=False):
    """
    Generate factor matrices that

    :param name: 'dct', 'legendre', 'chebyshev' or 'hermite'
    :param shape: two integers
    :param orthonormal: whether to orthonormalize the basis
    :return: a matrix of `shape`

    """

    if name == "dct":
        U = scipy.fftpack.dct(np.eye(shape[0]), norm="ortho")[:, :shape[1]]
    else:
        eval_points = np.linspace(-1, 1, shape[0])
        if name == "legendre":
            U = np.polynomial.legendre.legval(eval_points, np.eye(shape[0], shape[1])).T
        elif name == "chebyshev":
            U = np.polynomial.chebyshev.chebval(eval_points, np.eye(shape[0], shape[1])).T
        elif name == "hermite":
            U = np.polynomial.hermite.hermval(eval_points, np.eye(shape[0], shape[1])).T
        else:
            raise ValueError("Unsupported basis function")
    if orthonormal:
        U / np.sqrt(np.sum(U*U, axis=0))
    return torch.from_numpy(U)

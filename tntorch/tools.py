import tntorch as tn
import torch
import numpy as np
import time
import scipy.fftpack


"""
Array-like manipulations
"""


def squeeze(t, dim=None):
    """
    Removes singleton dimensions

    :param dim: which dim to delete. By default, all that have size 1
    :return: Another TT tensor, without dummy (singleton) indices
    """

    if dim is None:
        dim = np.where([s == 1 for s in t.shape])[0]
    assert np.all(np.array(t.shape)[dim] == 1)

    idx = [slice(None) for n in range(t.dim())]
    for m in dim:
        idx[m] = 0
    return t[tuple(idx)]


def cat(*ts, dim):
    """
    Concatenate two or more tensors along a given dim, similarly to PyTorch's `cat()`.

    :param ts: a list of tensors
    :param dim: an int
    :return: a tensor of the same shape as all tensors in the list, except along `dim` where it has the sum of shapes

    """

    if hasattr(ts[0], '__len__'):
        ts = ts[0]
    if len(ts) == 1:
        return ts[0].clone()
    if any([any([t.shape[n] != ts[0].shape[n] for n in np.delete(range(ts[0].dim()), dim)]) for t in ts[1:]]):
        raise ValueError('To concatenate tensors, all must have the same shape along all but the given dim')

    shapes = np.array([t.shape[dim] for t in ts])
    sumshapes = np.concatenate([np.array([0]), np.cumsum(shapes)])
    for i in range(len(ts)):
        t = ts[i].clone()
        if t.Us[dim] is None:
            if t.cores[dim].dim() == 2:
                t.cores[dim] = torch.zeros(sumshapes[-1], t.cores[dim].shape[-1])
            else:
                t.cores[dim] = torch.zeros(t.cores[dim].shape[0], sumshapes[-1], t.cores[dim].shape[-1])
            t.cores[dim][..., sumshapes[i]:sumshapes[i+1], :] += ts[i].cores[dim]
        else:
            t.Us[dim] = torch.zeros(sumshapes[-1], t.Us[dim].shape[-1])
            t.Us[dim][sumshapes[i]:sumshapes[i+1], :] += ts[i].Us[dim]
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
        try:
            idxs.append(t.idxs[n].clone())
        except Exception:
            idxs.append(None)
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
        idx = np.arange(shape[d]-1, -1, -1)
        if result.Us[d] is not None:
            result.Us[d] = result.Us[d][idx, :]
        else:
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
Multilinear algebra
"""


def sum(t, dim=None, keepdim=False):
    """
    Compute the sum of a tensor along all (or some) dimensions.

    :param t: a tensor
    :param dim: an int or list of ints. By default, all dims will be summed
    :param keepdim: if True, summed dimensions will be kept as singletons. Default is False
    :return: a scalar (if keepdim is False and all dims were chosen) or tensor otherwise

    """

    if dim is None:
        dim = np.arange(t.dim())
    if not hasattr(dim, '__len__'):
        dim = [dim]
    us = [torch.ones(t.shape[d]) for d in dim]
    result = tn.ttm(t, us, dim)
    if keepdim:
        return result
    else:
        return tn.squeeze(result, dim)


def ttm(t, U, dim=None, transpose=False):
    """
    Tensor-times-matrix (TTM) along one or several dimensions

    :param U: one or several factors
    :param dim: one or several dimensions (may be vectors or matrices). If None, the first len(U) dims are assumed
    :param transpose: if False (default) the contraction is performed
     along U's rows, else along its columns
    :return: transformed TT

    """

    if not isinstance(U, (list, tuple)):
        U = [U]
    if dim is None:
        dim = range(len(U))
    if not hasattr(dim, '__len__'):
        dim = [dim]
    dim = list(dim)
    for i in range(len(dim)):
        if dim[i] < 0:
            dim[i] += t.dim()

    cores = []
    Us = []
    for n in range(t.dim()):
        if n in dim:
            if transpose:
                factor = U[dim.index(n)].t().double()
            else:
                factor = U[dim.index(n)].double()
            if factor.dim() == 1:
                factor = factor[None, :]
            if t.Us[n] is None:
                if t.cores[n].dim() == 3:
                    cores.append(torch.einsum('iak,ja->ijk', (t.cores[n], factor)))
                else:
                    cores.append(torch.einsum('ai,ja->ji', (t.cores[n], factor)))
                Us.append(None)
            else:
                cores.append(t.cores[n].clone())
                Us.append(torch.matmul(factor, t.Us[n]))
        else:
            cores.append(t.cores[n].clone())
            if t.Us[n] is None:
                Us.append(None)
            else:
                Us.append(t.Us[n].clone())
    return tn.Tensor(cores, Us=Us, idxs=t.idxs)


def cumsum(t, dim):
    """
    Computes the cumulative sum of a tensor along one or several dims, similarly to PyTorch's `cumsum()`.

    :param t: a tensor
    :param dim: an int or list of ints
    :return: a tensor of the same shape

    """

    if not hasattr(dim, '__len__'):
        dim = [dim]

    t = t.clone()
    for n in dim:
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


def sample(t, P=1):
    """
    Generate P points (with replacement) from a joint PDF distribution represented by this tensor.

    The tensor does not have to sum 1.

    :param P: how many samples to draw (default: 1)
    :return Xs: a matrix of size P x N

    """

    def from_matrix(M):
        """
        Treat each row of a matrix M as a PMF and select a column per row according to it
        """

        M /= torch.sum(M, dim=1)[:, None]  # Normalize row-wise
        M = np.hstack([np.zeros([M.shape[0], 1]), M])
        M = np.cumsum(M, axis=1)
        thresh = np.random.rand(M.shape[0])
        M -= thresh[:, np.newaxis]
        shiftand = np.logical_and(M[:, :-1] <= 0, M[:, 1:] > 0)  # Find where the sign switches
        return np.where(shiftand)[1]

    N = t.dim()
    tsum = tn.sum(t, dim=np.arange(N), keepdim=True).decompress_tucker_factors()
    Xs = torch.zeros([P, N])
    rights = [torch.ones(1)]
    for core in tsum.cores[::-1]:
        rights.append(torch.matmul(torch.sum(core, dim=1), rights[-1]))
    rights = rights[::-1]
    lefts = torch.ones([P, 1])
    t = t.decompress_tucker_factors()
    for mu in range(t.dim()):
        fiber = torch.einsum('ijk,k->ij', (t.cores[mu], rights[mu + 1]))
        per_point = torch.einsum('ij,jk->ik', (lefts, fiber))
        rows = from_matrix(per_point)
        Xs[:, mu] = torch.Tensor(rows)
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
    Generate a factor matrix whose columns are functions of a truncated basis.

    :param name: 'dct', 'legendre', 'chebyshev' or 'hermite'
    :param shape: two integers
    :param orthonormal: whether to orthonormalize the basis
    :return: a matrix of `shape`

    """

    if name == "dct":
        U = scipy.fftpack.dct(np.eye(shape[0]), norm="ortho")[:, :shape[1]]
    elif name == 'identity':
        U = np.eye(shape[0], shape[1])
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


def reduce(ts, function, eps=0, rmax=np.iinfo(np.int32).max, verbose=False, **kwargs):
    """
    Compute a tensor as a function to all tensors in a sequence.

    Example 1 (addition):

    > import operator
    > tn.reduce([t1, t2], operator.add)

    Example 2 (cat with bounded rank):

    > tn.reduce([t1, t2], tn.cat, rmax=10)

    :param ts: A generator (or list) of tensors
    :param eps: intermediate tensors will be rounded at this error when climbing up the hierarchy
    :param rmax: no node should exceed this number of ranks
    :return: the reduced result

    """

    d = dict()
    start = time.time()
    for i, elem in enumerate(ts):
        if verbose and i % 100 == 0:
            print("reduce: element {}, time={:g}".format(i, time.time()-start))
        climb = 0  # For going up the tree
        while climb in d:
            elem = tn.round(function(d[climb], elem, **kwargs), eps=eps, rmax=rmax)
            d.pop(climb)
            climb += 1
        d[climb] = elem
    keys = list(d.keys())
    result = keys[0]
    for key in keys[1:]:
        result = tn.round(function(result, d[key], **kwargs), eps=eps, rmax=rmax)
    return result

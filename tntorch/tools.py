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
    Removes singleton dimensions.

    :param t: input :class:`Tensor`
    :param dim: which dim to delete. By default, all that have size 1

    :return: another :class:`Tensor`, without dummy (singleton) indices
    """

    if dim is None:
        dim = np.where([s == 1 for s in t.shape])[0]
    if not hasattr(dim, '__len__'):
        dim = [dim]

    assert np.all(np.array(t.shape)[dim] == 1)

    idx = [slice(None) for n in range(len(t.shape))]
    for m in dim:
        idx[m] = 0
    return t[tuple(idx)]


def unsqueeze(t, dim):
    """
    Inserts singleton dimensions at specified positions.

    :param t: input :class:`Tensor`
    :param dim: int or list of int

    :return: a :class:`Tensor` with dummy (singleton) dimensions inserted at the positions given by `dim`
    """

    if not hasattr(dim, '__len__'):
        dim = [dim]

    idx = [slice(None) for n in range(t.dim()+len(dim))]
    for d in dim:
        idx[d] = None
    return t[tuple(idx)]


def cat(*ts, dim):
    """
    Concatenate two or more tensors along a given dim, similarly to PyTorch's `cat()`.

    :param ts: a list of :class:`Tensor`
    :param dim: an int

    :return: a :class:`Tensor` of the same shape as all tensors in the list, except along `dim` where it has the sum of shapes
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
    Inverts the dimension order of a tensor, e.g. :math:`I_1 \\times I_2 \\times I_3` becomes :math:`I_3 \\times I_2 \\times I_1`.

    :param t: input tensor

    :return: another :class:`Tensor`, indexed by dimensions in inverse order
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


def meshgrid(*axes, batch=False):
    """
    See NumPy's or PyTorch's `meshgrid()`.

    :param axes: a list of N ints or torch vectors

    :return: a list of N :class:`Tensor`, of N dimensions each
    """

    device = None
    if not hasattr(axes, '__len__'):
        axes = [axes]
    if hasattr(axes[0], '__len__'):
        axes = axes[0]
    if hasattr(axes[0], 'device'):
        device = axes[0].device
    axes = list(axes)
    N = len(axes)
    for n in range(N):
        if not hasattr(axes[n], '__len__'):
            axes[n] = torch.arange(axes[n], dtype=torch.get_default_dtype())

    tensors = []
    for n in range(N):
        cores = [torch.ones(1, len(ax), 1).to(device) for ax in axes]
        if isinstance(axes[n], torch.Tensor):
            cores[n] = axes[n].type(torch.get_default_dtype())
        else:
            cores[n] = torch.tensor(axes[n].type(torch.get_default_dtype()))
        cores[n] = cores[n][None, :, None].to(device)
        tensors.append(tn.Tensor(cores, device=device, batch=batch))
    return tensors


def flip(t, dim):
    """
    Reverses the order of a tensor along one or several dimensions; see NumPy's or PyTorch's `flip()`.

    :param t: input :class:`Tensor`
    :param dims: an int or list of ints

    :return: another :class:`Tensor` of the same shape
    """

    if not hasattr(dim, '__len__'):
        dim = [dim]

    shape = t.shape
    result = t.clone()
    for d in dim:
        idx = np.arange(shape[d]-1, -1, -1)
        if result.Us[d] is not None:
            result.Us[d] = result.Us[d][idx, :]
        else:
            result.cores[d] = result.cores[d][..., idx, :]
    return result


def unbind(t, dim):
    """
    Slices a tensor along a dimension and returns the slices as a sequence, like PyTorch's `unbind()`.

    :param t: input :class:`Tensor`
    :param dim: an int

    :return: a list of :class:`Tensor`, as many as `t.shape[dim]`
    """

    if dim < 0:
        dim += t.dim()
    return [t[[slice(None)]*dim + [sl] + [slice(None)]*(t.dim()-1-dim)] for sl in range(t.shape[dim])]


def unfolding(data, n, batch=False):
    """
    Computes the `n-th mode unfolding <https://epubs.siam.org/doi/pdf/10.1137/07070111X>`_ of a PyTorch tensor.

    :param data: a PyTorch tensor
    :param n: unfolding mode
    :param batch: Boolean

    :return: a PyTorch matrix
    """
    if batch:
        return data.permute(
            [0, n + 1] + \
            list(range(1, n + 1)) + \
            list(range(n + 2, data.dim()))
        ).reshape([data.shape[0], data.shape[n + 1], -1])
    else:
        return data.permute([n] + list(range(n)) + list(range(n + 1, data.dim()))).reshape([data.shape[n], -1])


def right_unfolding(core, batch=False):
    """
    Computes the `right unfolding <https://epubs.siam.org/doi/pdf/10.1137/090752286>`_ of a 3D PyTorch tensor.

    :param core: a PyTorch tensor of shape :math:`I_1 \\times I_2 \\times I_3`
    :param batch: Boolean

    :return: a PyTorch matrix of shape :math:`I_1 \\times I_2 I_3`
    """
    if batch:
        return core.reshape([core.shape[0], core.shape[1], -1])
    else:
        return core.reshape([core.shape[0], -1])


def left_unfolding(core, batch=False):
    """
    Computes the `left unfolding <https://epubs.siam.org/doi/pdf/10.1137/090752286>`_ of a 3D PyTorch tensor.

    :param core: a PyTorch tensor of shape :math:`I_1 \\times I_2 \\times I_3`

    :return: a PyTorch matrix of shape :math:`I_1 I_2 \\times I_3`
    """

    if batch:
        return core.reshape([core.shape[0], -1, core.shape[-1]])
    else:
        return core.reshape([-1, core.shape[-1]])


"""
Multilinear algebra
"""


def ttm(t, U, dim=None, transpose=False):
    """
    `Tensor-times-matrix (TTM) <https://epubs.siam.org/doi/pdf/10.1137/07070111X>`_ along one or several dimensions.

    :param t: input :class:`Tensor`
    :param U: one or several factors
    :param dim: one or several dimensions (may be vectors or matrices). If None, the first len(U) dims are assumed
    :param transpose: if False (default) the contraction is performed
     along U's rows, else along its columns

    :return: transformed :class:`Tensor`
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
                factor = U[dim.index(n)].t()
            else:
                factor = U[dim.index(n)]
            if factor.dim() == 1 and not t.batch:
                factor = factor[None, ...]
            if factor.dim() == 2 and t.batch:
                factor = factor[:,  None, ...]
            if t.Us[n] is None:
                if t.batch:
                    if t.cores[n].dim() == 4:
                        cores.append(torch.einsum('biak,bja->bijk', (t.cores[n], factor)))
                    else:
                        cores.append(torch.einsum('bai,bja->bji', (t.cores[n], factor)))
                else:
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
    return tn.Tensor(cores, Us=Us, idxs=t.idxs, batch=t.batch)


"""
Miscellaneous
"""


def mask(t, mask):
    """
    Masks a tensor. Basically an element-wise product, but this function makes sure slices are matched according to their "meaning" (as annotated by the tensor's `idx` field, if available)

    :param t: input :class:`Tensor`
    :param mask: a mask :class:`Tensor`

    :return: masked :class:`Tensor`
    """
    device = t.cores[0].device
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
            cores.append(mask.cores[n][..., idx, :].to(device))
            Us.append(None)
        else:
            cores.append(mask.cores[n].to(device))
            Us.append(mask.Us[n][idx, :])
    mask = tn.Tensor(cores, Us, device=device)
    return t*mask


def sample(t, P=1):
    """
    Generate P points (with replacement) from a joint PDF distribution represented by a tensor.

    The tensor does not have to sum 1 (will be handled in a normalized form).

    :param t: a :class:`Tensor`
    :param P: how many samples to draw (default: 1)

    :return Xs: an integer matrix of size :math:`P \\times N`
    """

    def from_matrix(M):
        """
        Treat each row of a matrix M as a PMF and select a column per row according to it
        """

        M = np.abs(M)
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
        Xs[:, mu] = torch.tensor(rows)
        lefts = torch.einsum('ij,jik->ik', (lefts, t.cores[mu][:, rows, :]))

    return Xs


def hash(t):
    """
    Computes an integer number that depends on the tensor entries (not on its internal compressed representation).

    We obtain it as :math:`\\langle T, W \\rangle`, where :math:`W` is a rank-1 tensor of weights selected at random (always the same seed).

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
    :param batch: Boolean

    :return: a PyTorch matrix of `shape`
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


def reduce(ts, function, eps=0, rmax=np.iinfo(np.int32).max, algorithm='svd', verbose=False, **kwargs):
    """
    Compute a tensor as a function to all tensors in a sequence.

    :Example 1 (addition):

    >>> import operator
    >>> tn.reduce([t1, t2], operator.add)

    :Example 2 (cat with bounded rank):

    >>> tn.reduce([t1, t2], tn.cat, rmax=10)

    :param ts: A generator (or list) of :class:`Tensor`
    :param eps: intermediate tensors will be rounded at this error when climbing up the hierarchy
    :param rmax: no node should exceed this number of ranks
    :param algorithm: passed to :func:`round.round()`
    :param verbose: Boolean

    :return: the reduced result
    """

    d = dict()
    start = time.time()
    for i, elem in enumerate(ts):
        if verbose and i % 100 == 0:
            print("reduce: element {}, time={:g}".format(i, time.time()-start))
        climb = 0  # For going up the tree
        while climb in d:
            elem = tn.round(function(d[climb], elem, **kwargs), eps=eps, rmax=rmax, algorithm=algorithm)
            d.pop(climb)
            climb += 1
        d[climb] = elem
    keys = list(d.keys())
    result = d[keys[0]]
    for key in keys[1:]:
        result = tn.round(function(result, d[key], **kwargs), eps=eps, rmax=rmax, algorithm=algorithm)
    return result


def pad(t, shape, dim=None, fill_value=0):
    """
    Pad a tensor with a constant value.

    :param t: N-dim input :class:`Tensor`
    :param shape: int or list of ints
    :param dim: int or list of ints (default: all modes)
    :param fill_value: default is 0

    :return: a :class:`Tensor` of size `shape` along the indicated modes
    """

    if dim is None:
        dim = range(t.dim())
    if not hasattr(dim, '__len__'):
        dim = [dim]
    if not hasattr(shape, '__len__'):
        shape = [shape]*len(dim)

    t = t.clone()
    for i in range(len(dim)):
        mult = 0
        if i == 0:
            mult = fill_value
        if t.Us[dim[i]] is None:
            if t.cores[dim[i]].dim() == 2:
                t.cores[dim[i]] = torch.cat([t.cores[dim[i]],
                                             mult*torch.ones(shape[i] - t.cores[dim[i]].shape[0],
                                                         t.cores[dim[i]].shape[1])], dim=0)
            else:
                t.cores[dim[i]] = torch.cat([t.cores[dim[i]],
                                             mult*torch.ones(t.cores[dim[i]].shape[0],
                                                         shape[i] - t.cores[dim[i]].shape[1],
                                                         t.cores[dim[i]].shape[2])], dim=1)
        else:
            t.Us[dim[i]] = torch.cat([t.Us[dim[i]],
                                         mult*torch.ones(shape[i] - t.Us[dim[i]].shape[0],
                                                     t.Us[dim[i]].shape[1])], dim=0)
    return t

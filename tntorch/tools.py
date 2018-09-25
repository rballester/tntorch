import tntorch as tn
import torch
import numpy as np
import time
import scipy.fftpack
from functools import reduce


def core_kron(a, b):
    c = a[:, None, :, :, None] * b[None, :, :, None, :]
    c = c.reshape([a.shape[0] * b.shape[0], -1, a.shape[-1] * b.shape[-1]])
    return c


def dot(a, b, k=None):
    """
    Computes the dot product between two tensors.

    :param a: a tensor
    :param b: a tensor
    :return: a scalar

    """

    return a.dot(b, k)


def mean(t):
    """
    Computes the mean of a tensor.

    :param t: a tensor
    :return: a scalar

    """

    return tn.sum(t) / t.size


def var(t):
    """
    Computes the variance of a tensor.

    :param t: a tensor
    :return: a scalar >= 0

    """

    return tn.normsq(t-tn.mean(t)) / t.size


def std(t):
    """
    Computes the standard deviation of a tensor.

    :param t: a tensor
    :return: a scalar >= 0

    """

    return torch.sqrt(tn.var(t))


def normsq(t):
    """
    Computes the squared norm of a tensor.

    :param t: a tensor
    :return: a scalar >= 0

    """

    return tn.dot(t, t)


def norm(t):
    """
    Computes the L^2 (Frobenius) norm of a tensor.

    :param t: a tensor
    :return: a scalar >= 0

    """

    return torch.sqrt(torch.clamp(tn.normsq(t), min=0))


def sum(t, modes=None, keepdims=False):
    """
    Compute the sum of a tensor along all (or some) dimensions.

    :param t: a tensor
    :param modes: an int or list of ints. By default, all modes will be summed
    :param keepdims: if True, summed dimensions will be kept as singletons. Default is False
    :return: a scalar

    """

    # return self.ttm([torch.arange(sh, dtype=torch.double) for sh in self.shape], mode=range(self.ndim)).item()
    if modes is None:
        modes = np.arange(t.ndim)
    if not hasattr(modes, '__len__'):
        modes = [modes]
    cores = []
    Us = []
    for n in range(t.ndim):
        if n in modes:
            if t.Us[n] is None:
                cores.append(torch.sum(t.cores[n], dim=1, keepdim=True))
                Us.append(None)
            else:
                cores.append(t.cores[n].clone())
                Us.append(torch.sum(t.Us[n], dim=0, keepdim=True))
        else:
            cores.append(t.cores[n].clone())
            Us.append(t.Us[n].clone())
    # return tn.Tensor(cores, Us=Us)
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
    for n in range(t.ndim):
        if n in mode:
            if transpose:
                factor = U[mode.index(n)].t
            else:
                factor = U[mode.index(n)]
            if factor.dim() == 1:
                factor = factor[None, :]
            if t.Us[mode[n]] is None:
                cores.append(torch.einsum('iak,ja->ijk', (t.cores[mode[n]], factor)))
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


def round(t, **kwargs):
    t2 = t.clone()
    t2.round(**kwargs)
    return t.round()


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


def rand(shape, ranks_tt=1, ranks_tucker=None, requires_grad=False, device=None):
    """
    Generate a TT with random cores (and optionally factors), whose entries are uniform in [0, 1]

    :param shape:
    :param ranks_tt: an integer or list
    :param ranks_tucker: an integer or list
    :param requires_grad:
    :param device:
    :return:

    """

    return _random(torch.rand, shape, ranks_tt, ranks_tucker, requires_grad, device)


def randn(shape, ranks_tt=1, ranks_tucker=None, requires_grad=False, device=None):
    """
    Like `rand()`, but entries are distributed as a normal with mu=0, sigma=1
    """

    return _random(torch.randn, shape, ranks_tt, ranks_tucker, requires_grad, device)


def _random(function, shape, ranks_tt=1, ranks_tucker=None, requires_grad=False, device=None):
    if not hasattr(ranks_tt, "__len__"):
        ranks_tt = [1, ] + [ranks_tt, ] * (len(shape) - 1) + [1, ]
    if not hasattr(ranks_tucker, "__len__"):
        ranks_tucker = [ranks_tucker for n in range(len(shape))]
    assert len(ranks_tt) == len(shape) + 1
    assert len(ranks_tucker) == len(shape)
    assert np.min(ranks_tt) >= 1

    cores = []
    Us = []
    for n in range(len(shape)):
        if ranks_tucker[n] is None:
            cores.append(function([ranks_tt[n], shape[n], ranks_tt[n+1]], requires_grad=requires_grad, device=device))
            Us.append(None)
        else:
            cores.append(function([ranks_tt[n], ranks_tucker[n], ranks_tt[n+1]], requires_grad=requires_grad, device=device))
            Us.append(function([shape[n], ranks_tucker[n]], requires_grad=requires_grad, device=device))
    return tn.Tensor(cores, Us=Us)


def ones(shape):
    return tn.Tensor([torch.ones(1, sh, 1) for sh in shape])


def ones_like(tensor):
    return ones(tensor.shape)


def zeros(shape):
    return tn.Tensor([torch.zeros(1, sh, 1) for sh in shape])


def zeros_like(tensor):
    return zeros(tensor.shape)


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
    for n in range(t.ndim):
        idx = np.array(idxs[n])
        idx[idx >= mask.shape[n]] = mask.shape[n]-1  # Clamp
        if mask.Us[n] is None:
            cores.append(mask.cores[n][:, idx, :])
            Us.append(None)
        else:
            cores.append(mask.cores[n])
            Us.append(mask.Us[n][idx, :])
    return t*tn.Tensor(cores, Us)


def right_unfolding(core):
    return core.reshape([core.shape[0], -1])


def left_unfolding(core):
    return core.reshape([-1, core.shape[-1]])


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


def squeeze(t, modes=None):
    """
    Removes singleton dimensions

    :param modes: which modes to delete. By default, all that have size 1
    :return: Another TT tensor, without dummy (singleton) indices
    """

    if modes is None:
        modes = np.where(t.shape == 1)[0]
    assert np.all(np.array(t.shape)[modes] == 1)

    idx = [slice(None) for n in range(t.ndim)]
    for m in modes:
        idx[m] = 0
    return t[idx]


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

    Xs = torch.zeros([P, t.ndim])
    rights = [torch.ones(1)]
    for core in t.cores[::-1]:
        rights.append(torch.matmul(torch.sum(core, dim=1), rights[-1]))
    rights = rights[::-1]
    lefts = torch.ones([P, 1])

    for mu in range(t.ndim):
        fiber = torch.einsum('ijk,k->ij', (t.cores[mu], rights[mu + 1]))
        per_point = torch.einsum('ij,jk->ik', (lefts, fiber))
        rows = from_matrix(per_point)
        Xs[:, mu] = rows
        lefts = torch.einsum('ij,jik->ik', (lefts, t.cores[mu][:, rows, :]))

    return Xs


def optimize(tensors, loss_function, tol=1e-4, max_iter=10000, print_freq=500, verbose=True):
    """
    High-level wrapper for iterative learning.
    
    Default stopping criterion: either the absolute (or relative) loss improvement must fall below `tol`.
    In addition, the rate loss improvement must be slowing down.

    :param tensors: one or several tensors; will be fed to `loss_function`
    :param loss_function: must take `tensors` and return a scalar (or tuple thereof)
    :param tol: stopping criterion
    :param max_iter:
    :param print_freq: progress will be printed every this many iterations
    :param verbose:

    """

    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    parameters = []
    for t in tensors:
        parameters.extend([c for c in t.cores if c.requires_grad])
        parameters.extend([U for U in t.Us if U is not None and U.requires_grad])
    if len(parameters) == 0:
        raise ValueError("There are no parameters to optimize. Did you forget a requires_grad=True somewhere?")

    optimizer = torch.optim.Adam(parameters)
    losses = []
    converged = False
    start = time.time()
    iter = 0
    while True:
        optimizer.zero_grad()
        loss = loss_function(*tensors)
        if not isinstance(loss, (tuple, list)):
            loss = [loss]
        losses.append(reduce(lambda x, y: (x + y) / 2, loss))
        # losses.append(sum(loss))
        if len(losses) >= 2:
            delta_loss = (losses[-1] - losses[-2])
        else:
            delta_loss = float('-inf')
        if iter >= 2 and tol is not None and (losses[-1] <= tol or -delta_loss / losses[-1] <= tol) and losses[-2]-losses[-1] < losses[-3]-losses[-2]:
            converged = True
            break
        if iter == max_iter:
            break
        if verbose and iter % print_freq == 0:
            print('iter: {: <6} | loss: '.format(iter), end='')
            print(' + '.join(['{:10.6f}'.format(l.item()) for l in loss]), end='')
            if len(loss) > 1:
                print(' = {:10.4}'.format(losses[-1].item()), end='')
            print(' | total time: {:9.4f}'.format(time.time() - start))
        losses[-1].backward()
        optimizer.step()
        iter += 1
    if verbose:
        print('iter: {: <6} | loss: '.format(iter), end='')
        print(' + '.join(['{:10.6f}'.format(l.item()) for l in loss]), end='')
        if len(loss) > 1:
            print(' = {:10.4}'.format(losses[-1].item()), end='')
        print(' | total time: {:9.4f}'.format(time.time() - start), end='')
        if converged:
            print(' <- converged (tol={})'.format(tol))
        else:
            print(' <- max_iter was reached')


def hash(t):
    """
    Obtains an integer number that depends on the tensor entries (not on its internal compressed representation).

    We compute it as <T, W>, where W is a rank-1 tensor of weights selected at random (always the same seed).

    :return: an integer

    """

    gen = torch.Generator()
    gen.manual_seed(0)
    cores = [torch.ones(1, 1, 1) for n in range(t.ndim)]
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
        eval_points = np.linspace(-1+(1./shape[0]), 1-(1./shape[0]), shape[0])
        if name == "legendre":
            U = np.polynomial.legendre.legval(eval_points, np.eye(shape[0], shape[1])).T
        elif name == "chebyshev":
            U = np.polynomial.chebyshev.chebval(eval_points, np.eye(shape[0], shape[1])).T
        elif name == "hermite":
            U = np.polynomial.hermite.hermval(eval_points, np.eye(shape[0], shape[1])).T
        else:
            raise ValueError("Unrecognized basis function")
    if orthonormal:
        U / np.sqrt(np.sum(U*U, axis=0))
    return torch.from_numpy(U)


def dof(t):
    """
    Compute the number of degrees of freedom of a tensor network.

    It is the sum of sizes of all its tensor nodes that have the requires_grad=True flag.

    :return: an integer

    """

    result = 0
    for n in range(t.ndim):
        if t.cores[n].requires_grad:
            result += t.cores[n].shape[0] * t.cores[n].shape[1] * t.cores[n].shape[2]
        if t.Us[n] is not None and t.Us[n].requires_grad:
            result += t.Us[n].shape[0] * t.Us[n].shape[1]
    return result


def cat(ts, mode):
    """
    Concatenate two or more tensors along a given mode, similarly to PyTorch's `cat()`.

    :param ts: a list of tensors
    :param mode: an int
    :return: a tensor of the same shape as all tensors in the list, except along `mode` where it has the sum of shapes

    """

    if len(ts) == 1:
        return ts[0].clone()
    if any([any([t.shape[n] != ts[0].shape[n] for n in np.delete(range(ts[0].ndim), mode)]) for t in ts[1:]]):
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

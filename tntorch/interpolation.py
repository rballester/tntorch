import numpy as np
import tntorch as tn
import torch
import time


def als_completion(train_x, train_y, ranks_tt, shape=None, ws=None, x0=None, niter=10, verbose=True):
    """
    Complete an N-dimensional TT from P samples using alternating least squares (ALS).
    We assume only low-rank structure, and no smoothness/spatial locality. Such assumption requires that there is at
    least one sample for each tensor hyperslice. This is usually the case for categorical variables, since it is
    meaningless to have a class or label for which no instances exist.

    Note that this method may not converge (or be extremely slow to do so) if the number of available samples is below
    or near a certain proportion of the overall tensor. Such proportion, unfortunately, depends on the data set and its
    true rank structure
    ("Riemannian optimization for high-dimensional tensor completion", M. Steinlechner, 2015)

    :param Xs: a P x N matrix of integers (tensor indices)
    :param ys: a vector with P elements
    :param ranks_tt: an integer (or list). Ignored if x0 is given
    :param shape: list of N integers. If None, the smallest shape that accommodates `Xs` will be chosen
    :param ws: a vector with P elements, with the weight of each sample (if None, 1 is assumed)
    :param x0: initial solution (a TT tensor). If None, a random tensor will be used
    :param niter: number of ALS sweeps. Default is 10
    :param verbose:
    :return: a `tntorch.Tensor'
    """

    assert train_x.dim() == 2
    assert train_y.dim() == 1
    if ws is None:
        ws = torch.ones(len(train_y))
    train_x = train_x.long()
    if shape is None:
        shape = [val.item() for val in torch.max(train_x, dim=0)[0]+1]
    train_y = train_y.to(torch.get_default_dtype())
    P = train_x.shape[0]
    N = train_x.shape[1]
    if x0 is None:
        x0 = tn.rand(shape, ranks_tt=ranks_tt)
    # All tensor slices must contain at least one sample point
    for dim in range(N):
        if torch.unique(train_x[:, dim]).numel() != x0.shape[dim]:
            raise ValueError('One groundtruth sample is needed for every tensor slice')

    if verbose:
        print('Completing a {}D tensor of size {} using {} samples...'.format(N, list(shape), P))

    normy = torch.norm(train_y)
    x0.orthogonalize(0)
    cores = x0.cores

    # Memoized product chains for all groundtruth points
    # lefts will be initialized on the go
    lefts = [torch.ones(1, P, x0.cores[n].shape[0]) for n in range(N)]
    # rights, however, needs to be initialized now
    rights = [None] * N
    rights[-1] = torch.ones(1, P, 1)
    for dim in range(N-2, -1, -1):
        rights[dim] = torch.einsum('ijk,kjl->ijl', (cores[dim+1][:, train_x[:, dim+1], :], rights[dim+1]))

    def optimize_core(cores, mu, direction):
        sse = 0
        for index in range(cores[mu].shape[1]):
            idx = torch.where(train_x[:, mu] == index)[0]
            leftside = lefts[mu][0, idx, :]
            rightside = rights[mu][:, idx, 0]
            lhs = rightside.t()[:, :, None]
            rhs = leftside[:, None, :]
            A = torch.reshape(lhs*rhs, [len(idx), -1])*ws[idx, None]
            b = train_y[idx]*ws[idx]
            sol = torch.lstsq(b, A)[0][:A.shape[1], :]
            residuals = torch.norm(A.matmul(sol)[:, 0] - b) ** 2
            cores[mu][:, index, :] = torch.reshape(sol, cores[mu][:, index, :].shape)#.t()
            sse += residuals
        # Update product chains for next core
        if direction == 'right':
            x0.left_orthogonalize(mu)
            lefts[mu+1] = torch.einsum('ijk,kjl->ijl', (lefts[mu], cores[mu][:, train_x[:, mu], :]))
        else:
            x0.right_orthogonalize(mu)
            rights[mu-1] = torch.einsum('ijk,kjl->ijl', (cores[mu][:, train_x[:, mu], :], rights[mu]))
        return sse

    start = time.time()
    for swp in range(niter):

        # Sweep: left-to-right
        for mu in range(N-1):
            optimize_core(cores, mu, direction="right")

        # Sweep: right-to-left
        for mu in range(N-1, 0, -1):
            sse = optimize_core(cores, mu, direction="left")
        eps = torch.sqrt(sse) / normy

        if verbose:
            print('iter: {: <{}}'.format(swp, len('{}'.format(niter)) + 1), end='')
            print('| eps: {:.3e}'.format(eps), end='')
            print(' | time: {:8.4f}'.format(time.time() - start))

    return x0


def sparse_tt_svd(Xs, ys, eps, shape=None, rmax=None):
    """
    TT-SVD for sparse tensors.

    :param Xs: matrix P x N of sample coordinates (integers)
    :param ys: P-sized vector of sample values
    :param eps: prescribed accuracy (resulting relative error is guaranteed to be not larger than this)
    :param shape: input tensor shape. If not specified, a tensor will be chosen such that `Xs` fits in
    :param rmax: optionally, cap all ranks above this value
    :param verbose:
    :return: a TT
    """

    def sparse_covariance(Xs, ys, nrows):
        u, v = torch.unique(Xs[:, 1:], dim=0, return_inverse=True)
        D = torch.zeros(nrows, len(u))
        D[Xs[:, 0], v] = ys
        return D.mm(D.t())

    def full_times_sparse(F, Xs, ys):
        F = torch.Tensor(F)
        u, v = torch.unique(Xs[:, 1:], dim=0, return_inverse=True)
        idx = np.unique(v, return_index=True)[1]

        D = torch.zeros(F.shape[1], len(u))
        D[Xs[:, 0], v] = ys
        FD = F.mm(D)
        new_row = torch.remainder(torch.arange(FD.numel()), FD.shape[0])
        newcols = Xs[idx, 1:][:, None, :].repeat(1, FD.shape[0], 1)  # , FD.shape[0], axis=1)
        newcols = newcols.reshape(len(idx) * FD.shape[0], -1)
        return torch.cat([new_row[:, None], newcols], dim=1), FD.t().flatten()

    def sparse_truncate_svd(Xs, ys, nrows, delta, rmax):

        cov = sparse_covariance(Xs, ys, nrows)

        w, v = torch.symeig(cov, eigenvectors=True)
        w[w < 0] = 0
        w = torch.sqrt(w)
        svd = [v, w]

        # Sort eigenvalues and eigenvectors in decreasing importance
        idx = np.argsort(svd[1])[torch.arange(len(svd[1])-1, -1, -1)]
        svd[0] = svd[0][:, idx]
        svd[1] = svd[1][idx]

        S = svd[1]**2
        where = torch.where(np.cumsum(S[torch.arange(len(S)-1, -1, -1)]) <= delta**2)[0]
        if len(where) == 0:
            rank = max(1, int(np.min([rmax, len(S)])))
        else:
            rank = max(1, int(np.min([rmax, len(S) - 1 - where[-1]])))
        left = svd[0]
        left = left[:, :rank]

        Xs, ys = full_times_sparse(left.T, Xs, ys)
        return left, Xs, ys

    if isinstance(Xs, np.ndarray):
        Xs = torch.Tensor(Xs)
    if isinstance(ys, np.ndarray):
        ys = torch.Tensor(ys)
    assert Xs.dim() == 2
    assert ys.dim() == 1
    N = Xs.shape[1]
    if shape is None:
        shape = [val.item() for val in torch.max(Xs, dim=0)[0]+1]
    assert N == len(shape)
    if rmax is None:
        rmax = np.iinfo(np.int32).max

    delta = eps / np.sqrt(N - 1) * torch.norm(ys).item()

    # TT-SVD iteration
    cores = []
    curshape = shape.copy()
    for n in range(1, N):
        left, Xs, ys = sparse_truncate_svd(Xs, ys, curshape[0], delta=delta, rmax=rmax)
        cores.append(left.reshape(left.shape[0]//shape[n-1], shape[n-1], left.shape[1]))
        curshape[0] = left.shape[1]  # Rank of this unfolding

        if n < N-1:  # Merge the two first indices (sparse reshape)
            Xs = torch.cat([Xs[:, 0:1]*curshape[1] + Xs[:, 1:2], Xs[:, 2:]], dim=1)
            curshape[1] *= curshape[0]
            curshape = curshape[1:]

    lastcore = torch.zeros(list(curshape))
    lastcore[list(Xs.t())] = ys
    cores.append(lastcore[:, :, None])

    return tn.Tensor(cores)


def pce_interpolation(x, y, p=5, q=0.75, nnz=None, I=1024, rmax=500, eps=1e-3, verbose=True):
    """
    Polynomial chaos expansion (PCE) interpolation. This function requires scikit-learn to run.
    Parameter distributions are approximated as histograms of the training data, and orthogonal polynomial is learned using Gram-Schmidt [2].

    The main hyperparameters are q, p, and nnz.

    Coefficient selection follows [1]. A sparse regularization is enforced: only entries x whose coordinates satisfy ||x||_q < p are considered, where the
     Lq norm is used and `p` is a threshold. Additionally, among that space of candidates, at most `nnz` can be non-zero
      (they are selected using Least Angle Regression, LARS).

    We assume parameters are independently distributed; however, the regressor will often work well
    even if this doesn't hold [1].

    [1] "Data-driven polynomial chaos expansion for machine learning regression", Torre et al. 2020
    [2] "Modeling Arbitrary Uncertainties Using Gram-Schmidt Polynomial Chaos", Witteveen and Bijl, 2012

    :param x: a matrix of shape P x N (floats; input features)
    :param y: a vector of size P
    :param p: threshold for the hyperbolic truncation norm. Default is 5
    :param q: the truncation will use norm Lq. Default is 0.75
    :param nnz: how many non-zero coefficients to search at most. Default is number of samples / 3
    :param I: the resulting tensor will have this spatial resolution. Default is 1024
    :param rmax: the TT cores will be capped at this rank. Default is 500
    :param eps: rounding error to cast PCE into TT. Default is 1e-3
    :param verbose: Boolean; default is True
    :return: a `tntorch.Tensor` in the TT-Tucker format
    """

    import sklearn.linear_model

    assert x.dim() == 2
    P = x.shape[0]
    N = x.shape[1]
    assert y.shape[0] == P
    assert y.dim() == 1
    assert 0 <= q <= 1
    if nnz is None:
        nnz = P//3

    if verbose:
        start = time.time()
        print('Time: {:.3f}s | '.format(time.time() - start), end='')
        print('PCE interpolation of {} training points in {} dimensions...'.format(P, N))

    # Normalize and discretize all features to integers in [0, I-1]
    x -= torch.min(x, dim=0, keepdim=True)[0]
    x /= torch.max(x, dim=0, keepdim=True)[0]
    x = torch.round(x*(I-1)).long()

    # Find the coordinates of all coefficient candidates (i.e. all that satisfy ||x||_q < p)
    idx = np.zeros(N, dtype=np.int32)

    def find_candidates(p):
        # Traverse the whole hypercube of possible coefficients (size S**N)
        # Since hyperbolic truncation selects a contiguous region, this is efficient
        S = int(np.ceil(p))
        coords = []
        while True:
            pos = N-1
            while pos >= 0 and (max(idx) >= S or np.sum(idx ** q) >= p ** q):
                idx[pos] = 0
                idx[pos - 1] += 1
                pos -= 1
            if pos < 0:  # Traversed the entire hypercube
                break
            coords.append(idx.copy())
            idx[-1] += 1
            if len(coords) > 1e3:
                raise ValueError('Space of non-zero coefficients is too large')
        return torch.Tensor(coords).long()

    coords = find_candidates(p)
    nnz = min(nnz, len(coords))
    S = int(np.ceil(p))

    if verbose:
        ncandidates = len(coords)
        print('Time: {:.3f}s | '.format(time.time() - start), end='')
        print('Candidates after hyperbolic truncation = {} out of {} ({:.3g}%)'.format(ncandidates, S ** N, ncandidates / (S ** N) * 100))

    def gram_schmidt(weights, S):
        """
        Create a truncated polynomial basis (with S elements) that is orthogonal
        with respect to a given measure.

        The elements are computed using Gram-Schmidt's orthonormalization process,
        and lead to a PCE family that can be used for any probability distribution of
        the inputs.

        :param weights: a vector containing the measure values
        :param S: an integer: how many basis elements to form.
        :return: a matrix of shape len(weight) x S (one column per basis element)
        """

        assert weights.dim() == 1

        I = len(weights)
        e = torch.linspace(-1, 1, I)[:, None] ** torch.arange(S)[None, :]
        psipsi = torch.zeros(S)
        psi = torch.zeros(I, S)
        psi[:, 0] = 1
        for s in range(1, S):
            psipsi[s-1] = torch.sum(psi[:, s-1]**2*weights)
            epsi = torch.einsum('i,is,i->s', e[:, s], psi[:, :s], weights)
            cjk = epsi/psipsi[:s]
            psi[:, s] = e[:, s] - torch.sum(psi[:, :s]*cjk[None, :], dim=1)
        psi /= torch.sqrt(torch.sum(psi**2*weights[:, None], dim=0, keepdim=True))
        return psi

    # Build orthogonal polynomial bases based on the
    # empirical marginals (frequencies) from the input features
    Ms = []
    for n in range(N):
        unique, counts = torch.unique(x[:, n], return_counts=True)
        marginal = torch.zeros(I)
        marginal[unique] = counts.double()/P
        Ms.append(gram_schmidt(marginal, S))

    # Assemble the design matrix
    M = torch.cat([Ms[n][x[:, n], :][:, None, :] for n in range(N)], dim=1)
    idx = torch.arange(N)[None, :].repeat(len(coords), 1)
    M = M[:, idx.flatten(), coords.flatten()].reshape(-1, coords.shape[0], coords.shape[1])
    M = torch.prod(M, dim=2)

    # Solve the sparse regression problem using LARS
    lars = sklearn.linear_model.Lars(n_nonzero_coefs=nnz, fit_intercept=False)
    lars.fit(M, y)

    if verbose:
        print('Time: {:.3f}s | '.format(time.time() - start), end='')
        print('LARS fitted {} nnz out of the {} candidates ({:.3g}%)'.format(nnz, ncandidates, nnz/ncandidates*100))

    # Assemble a TT-Tucker tensor:
    # The core is retrieved from the set of PCE coefficients found,
    # and is computed using sparse TT-SVD
    nonzeros = np.where(lars.coef_)[0]
    t = tn.sparse_tt_svd(coords[nonzeros, :], lars.coef_[nonzeros], rmax=rmax, eps=eps)
    # The factors are simply the polynomial bases
    t.Us = [Ms[n][:, :t.shape[n]] for n in range(N)]

    if verbose:
        print('Time: {:.3f}s | '.format(time.time() - start), end='')
        print('PCE finished, training relative error = {:.5g}'.format(tn.relative_error(y, t[x]).item()))

    return t

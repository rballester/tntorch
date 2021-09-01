import numpy as np
import tntorch as tn
import torch
import time


def als_completion(X, y, ranks_tt, shape=None, ws=None, x0=None, niter=10, verbose=True):
    """
    Complete an N-dimensional TT from P samples using alternating least squares (ALS).
    We assume only low-rank structure, and no smoothness/spatial locality. Such assumption requires that there is at
    least one sample for each tensor hyperslice. This is usually the case for categorical variables, since it is
    meaningless to have a class or label for which no instances exist.

    Note that this method may not converge (or be extremely slow to do so) if the number of available samples is below
    or near a certain proportion of the overall tensor. Such proportion, unfortunately, depends on the data set and its
    true rank structure
    ("Riemannian optimization for high-dimensional tensor completion", M. Steinlechner, 2015)

    :param X: a P X N matrix of integers (tensor indices)
    :param y: a vector with P elements
    :param ranks_tt: an integer (or list). Ignored if x0 is given
    :param shape: list of N integers. If None, the smallest shape that accommodates `X` will be chosen
    :param ws: a vector with P elements, with the weight of each sample (if None, 1 is assumed)
    :param x0: initial solution (a TT tensor). If None, a random tensor will be used
    :param niter: number of ALS sweeps. Default is 10
    :param verbose:
    :return: a `tntorch.Tensor'
    """

    assert not X.dtype.is_floating_point
    assert X.dim() == 2
    assert y.dim() == 1
    if ws is None:
        ws = torch.ones(len(y))
    X = X.long()
    if shape is None:
        shape = [val.item() for val in torch.max(X, dim=0)[0] + 1]
    y = y.to(torch.get_default_dtype())
    P = X.shape[0]
    N = X.shape[1]
    if x0 is None:
        x0 = tn.rand(shape, ranks_tt=ranks_tt)
    # All tensor slices must contain at least one sample point
    for dim in range(N):
        if torch.unique(X[:, dim]).numel() != x0.shape[dim]:
            raise ValueError('One groundtruth sample is needed for every tensor slice')

    if verbose:
        print('Completing a {}D tensor of size {} using {} samples...'.format(N, list(shape), P))

    normy = torch.norm(y)
    x0.orthogonalize(0)
    cores = x0.cores

    # Memoized product chains for all groundtruth points
    # lefts will be initialized on the go
    lefts = [torch.ones(1, P, x0.cores[n].shape[0]) for n in range(N)]
    # rights, however, needs to be initialized now
    rights = [None] * N
    rights[-1] = torch.ones(1, P, 1)
    for dim in range(N - 2, -1, -1):
        rights[dim] = torch.einsum('ijk,kjl->ijl', (cores[dim + 1][:, X[:, dim + 1], :], rights[dim + 1]))

    def optimize_core(cores, mu, direction):
        sse = 0
        for index in range(cores[mu].shape[1]):
            idx = torch.where(X[:, mu] == index)[0]
            leftside = lefts[mu][0, idx, :]
            rightside = rights[mu][:, idx, 0]
            lhs = rightside.t()[:, :, None]
            rhs = leftside[:, None, :]
            A = torch.reshape(lhs*rhs, [len(idx), -1])*ws[idx, None]
            b = y[idx] * ws[idx]
            sol = torch.linalg.lstsq(A, b).solution[:A.shape[1]]
            residuals = torch.norm(A.matmul(sol)[0] - b) ** 2
            cores[mu][:, index, :] = torch.reshape(sol, cores[mu][:, index, :].shape)#.t()
            sse += residuals
        # Update product chains for next core
        if direction == 'right':
            x0.left_orthogonalize(mu)
            lefts[mu+1] = torch.einsum('ijk,kjl->ijl', (lefts[mu], cores[mu][:, X[:, mu], :]))
        else:
            x0.right_orthogonalize(mu)
            rights[mu-1] = torch.einsum('ijk,kjl->ijl', (cores[mu][:, X[:, mu], :], rights[mu]))
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


def sparse_tt_svd(X, y, eps, shape=None, rmax=None):
    """
    TT-SVD for sparse tensors.

    :param X: matrix P X N of sample coordinates (integers)
    :param y: P-sized vector of sample values
    :param eps: prescribed accuracy (resulting relative error is guaranteed to be not larger than this)
    :param shape: input tensor shape. If not specified, a tensor will be chosen such that `X` fits in
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

        w, v = torch.linalg.eigh(cov)
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

    if isinstance(X, np.ndarray):
        X = torch.Tensor(X)
    if isinstance(y, np.ndarray):
        y = torch.Tensor(y)
    assert not X.dtype.is_floating_point
    assert X.dim() == 2
    assert y.dim() == 1
    N = X.shape[1]
    if shape is None:
        shape = [val.item() for val in torch.max(X, dim=0)[0] + 1]
    assert N == len(shape)
    if rmax is None:
        rmax = np.iinfo(np.int32).max

    delta = eps / np.sqrt(N - 1) * torch.norm(y).item()

    # TT-SVD iteration
    cores = []
    curshape = shape.copy()
    for n in range(1, N):
        left, X, y = sparse_truncate_svd(X, y, curshape[0], delta=delta, rmax=rmax)
        cores.append(left.reshape(left.shape[0]//shape[n-1], shape[n-1], left.shape[1]))
        curshape[0] = left.shape[1]  # Rank of this unfolding

        if n < N-1:  # Merge the two first indices (sparse reshape)
            X = torch.cat([X[:, 0:1] * curshape[1] + X[:, 1:2], X[:, 2:]], dim=1)
            curshape[1] *= curshape[0]
            curshape = curshape[1:]

    lastcore = torch.zeros(list(curshape))
    lastcore[list(X.t())] = y
    cores.append(lastcore[:, :, None])

    return tn.Tensor(cores)


def get_bounding_box(X):
    """
    Compute the bounding box of a set of points.

    :param X: a tensor of shape ... X N, where N is the number of features
    :return: a list of N pairs [bottom, top]
    """

    return [(torch.min(X[..., n]).item(), torch.max(X[..., n]).item()) for n in range(X.shape[-1])]


def features2indices(X, bbox=None, I=512, domain=None):
    """
    Convert floating point features into dicrete tensor indices.

    :param X: a float tensor of shape ... X N, where N is the number of features
    :param bbox: a list of N pairs [bottom, top]. If None (default), X's own bounding box will be used.
        Values falling outside the box will be clamped.
    :param I: grid resolution. Default is 512
    :param domain: optional list of N vectors to specify the grid. Overrides `bbox` and `I`
    :return: an integer tensor of shape ... X N
    """

    assert X.dtype.is_floating_point
    X = X.clone()
    if domain is not None:
        for n in range(X.shape[1]):
            X[:, n] = torch.Tensor(np.interp(X[:, n].numpy(), domain[n].numpy(), np.arange(len(domain[n]))))
        return torch.round(X).long()
    if bbox is None:
        bbox = tn.get_bounding_box(X)
    assert len(bbox) == X.shape[-1]
    bbox = torch.Tensor(bbox)
    X -= bbox[:, 0][[None] * (X.dim() - 1) + [slice(None)]]
    X /= (bbox[:, 1] - bbox[:, 0])[[None] * (X.dim() - 1) + [slice(None)]]
    X = torch.round(X * (I - 1)).long()
    X[X < 0] = 0
    X[X > I - 1] = I - 1
    return X


def indices2features(X, bbox=None, I=512, domain=None):

    assert not X.dtype.is_floating_point
    assert X.dim() == 2

    result = torch.zeros_like(X).to(torch.get_default_dtype())
    if domain is None:
        domain = [torch.linspace(
            b[0] + (b[1]-b[0])/(2*I),
            b[1] - (b[1]-b[0])/(2*I),
            I) for b in bbox]
    for n in range(X.shape[1]):
        result[:, n] = domain[n][X[:, n]]
    return result



def empirical_marginals(X, domain):
    """
    Given a matrix of sample points, get its discrete marginal distribution over a
    specified domain.

    :param X: a P x N matrix
    :param domain: a list of N vectors
    :return: a list of N vectors
    """

    assert X.dim() == 2
    assert X.shape[1] == len(domain)
    P = X.shape[0]
    N = X.shape[1]

    X_discrete = tn.discretize(X, domain=domain)
    result = [torch.zeros(len(domain[n])) for n in range(N)]
    for n in range(N):
        unique, counts = torch.unique(X_discrete[:, n], return_counts=True)
        result[n][unique.numpy()] = counts.double() / P
    return result


def gram_schmidt(x, S):
    """
    Create a truncated polynomial basis with S elements that is orthogonal
    with respect to a given measure.

    The elements are computed using a modified [1] version of Gram-Schmidt's orthonormalization
    process, and lead to a PCE family that can be used for any probability distribution of
    the inputs [2].

    [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Numerical_stability
    [2] "Modeling Arbitrary Uncertainties Using Gram-Schmidt Polynomial Chaos", Witteveen and Bijl, 2012

    :param x: list of observed inputs
    :param S: an integer: how many basis elements to form.
    :return: a matrix of shape S X S (one column per basis element)
    """

    assert x.dim() == 1

    xpowers = x[:, None] ** torch.arange(S, device=x.device)[None, :]

    def proj(u, v):
        xu = xpowers.matmul(u)
        xv = xpowers.matmul(v)
        return torch.mean(xu * xv) / torch.mean(xu * xu) * u

    def norm(u):
        xu = xpowers.matmul(u)
        return torch.sqrt(torch.mean(xu * xu))

    Psi = torch.eye(S, S, device=x.device)
    for s in range(1, S):
        u = Psi[:, s]
        for k in range(s):
            u = u - proj(Psi[:, k], u)
        Psi[:, s] = u / norm(u)

    return Psi


class PCEInterpolator:
    """
    Polynomial chaos expansion (PCE) interpolator. This class requires scikit-learn to run, and
    therefore is CPU-only. Parameter distributions are fully empirical. Orthogonal polynomial families
    are learned using Gram-Schmidt [1].

    Coefficient selection follows [2]. A sparse regularization is enforced: only entries X whose coordinates satisfy
    ||X||_q < p are considered, where the Lq norm is used and `p` is a threshold. Additionally, among that space of
    candidates, at most `nnz` can be non-zero (they are selected using Least Angle Regression, LARS).

    We assume parameters are independently distributed; however, the regressor will often work well
    even if this doesn't hold [2].

    [1] "Modeling Arbitrary Uncertainties Using Gram-Schmidt Polynomial Chaos", Witteveen and Bijl, 2012
    [2] "Data-driven Polynomial Chaos Expansion for Machine Learning Regression", Torre et al. 2020
    """

    def __init__(self):
        pass

    def _design_matrix(self, x):
        N = len(self.Psis)
        S = self.Psis[0].shape[0]
        M = torch.cat([(x[:, n:n+1]**torch.arange(S)[None, :]).matmul(self.Psis[n])[:, None, :] for n in range(N)], dim=1)
        idx = torch.arange(N)[None, :].repeat(len(self.coords), 1)
        M = M[:, idx.flatten(), self.coords.flatten()]
        M = M.reshape(-1, self.coords.shape[0], self.coords.shape[1])
        M = torch.prod(M, dim=2)
        return M

    def fit(self, X, y, p=5, q=0.75, val_split=0.1, seed=0, matrix_size_limit=5e7, retrain=True, verbose=True):
        """
        Fit the model to a training dataset (X, y) using LARS. The optimal number of non-zero
        coefficients to select is automatically found via a validation set.

        :param X: a matrix of shape P X N floats (input features)
        :param y: a vector of size P
        :param p: threshold for the hyperbolic truncation norm. Default is 5
        :param q: the truncation will use norm Lq. Default is 0.75
            the optimal will be selected using a validation set
        :param val_split: the fraction of elements to use for validation (default is 0.1)
        :param seed: integer, the seed used to select the train/validation split. Default is 0
        :param matrix_size_limit: abort if the design matrix exceeds this number of elements. Default is 5e7
        :param retrain: Boolean. If True (default), retrain on the entire input data after selecting the best nnz
            This improves interpolation quality, but setting retrain=False is faster and good for hyperparameter tuning.
        :param verbose: Boolean; default is True
        """

        import sklearn.linear_model
        assert X.dim() == 2
        assert X.dtype.is_floating_point
        P = X.shape[0]
        N = X.shape[1]
        assert y.shape[0] == P
        assert y.dim() == 1
        assert 0 <= q <= 1

        # Save features' bounding box
        self.bbox = tn.get_bounding_box(X)

        # Center each feature to improve stability of polynomial orthogonalization
        self.X_mean = torch.mean(X, dim=0)
        self.X_std = torch.std(X, dim=0)
        X = (X.clone() - self.X_mean[None, :])/self.X_std[None, :]

        # Split into train and validation sets
        n_val = int(P*val_split)
        rng = np.random.default_rng(seed=seed)
        idx_val = rng.choice(P, n_val)
        idx_train = np.delete(np.arange(P), idx_val)
        y_train = y[idx_train]
        y_val = y[idx_val]

        if verbose:
            start = time.time()
            print('PCE interpolation (p={}, q={}) of {} points ({} train + {} val) in {}D'.format(p, q, P, P-n_val, n_val, N))

        if verbose:
            print('{:.3f}s | '.format(time.time() - start), end='')
            print('Hyperbolic truncation...', end='')

        # Find the coordinates of all coefficient candidates (i.e. all that satisfy ||X||_q < p)
        idx = np.zeros(N, dtype=np.int)

        def find_candidates(p, q):
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
                if len(coords)*P > matrix_size_limit:
                    raise ValueError('Design matrix exceeds matrix_size_limit ({:g} elements). Decrease p or q, or increase matrix_size_limit'.format(matrix_size_limit))
            return torch.Tensor(coords).long()

        self.coords = find_candidates(p, q)
        S = int(np.ceil(p))

        if verbose:
            ncandidates = len(self.coords)
            print(' done, we kept {} / {} candidates'.format(ncandidates, S**N))
            print('{:.3f}s | '.format(time.time() - start), end='')
            print('Assembling a {} X {} design matrix...'.format(P, len(self.coords)), end='', flush=True)

        # Build orthogonal polynomial bases based on the
        # empirical marginals of the input features
        self.Psis = [gram_schmidt(X[:, n], S) for n in range(N)]

        # Assemble the design matrix from the training features
        M = self._design_matrix(X)
        M_train = M[idx_train, ...]
        M_val = M[idx_val, ...]

        if verbose:
            print(' done')
            print('{:.3f}s | '.format(time.time() - start), end='')
            print('Finding best nnz in LARS...', end='', flush=True)

        # Solve the sparse regression problem using LARS
        lars = sklearn.linear_model.Lars(n_nonzero_coefs=M_train.shape[1], fit_intercept=False, fit_path=True)
        lars.fit(M_train, y_train)

        # Find the validation eps for every choice of nnz (LARS' solution path)
        reco_path = torch.einsum('pc,cd->pd', M_val, torch.Tensor(lars.coef_path_))
        error_path = torch.sqrt(torch.sum((reco_path - y_val[:, None])**2, dim=0))/torch.norm(y_val)
        argmin = torch.argmin(error_path)  # Pick the best nnz
        nnz = len(np.where(lars.coef_path_[:, argmin])[0])

        if verbose:
            print(' done, val eps={:.5g}'.format(error_path[argmin]))
            print('{:.3f}s | '.format(time.time() - start), end='')

        if retrain:

            if verbose:
                print('Retraining at nnz={}...'.format(nnz), end='', flush=True)

            # Retrain now on the entire input dataset and optimal nnz
            lars = sklearn.linear_model.Lars(n_nonzero_coefs=nnz, fit_intercept=False, fit_path=False)
            lars.fit(M, y)

            # Remove features not selected by LARS
            lars.coef_ = lars.coef_[0, :]
            nonzeros = np.where(lars.coef_)[0]
            self.allcoords = self.coords
            self.allcoef = torch.Tensor(lars.coef_)
            self.coef = torch.Tensor(lars.coef_[nonzeros])
            self.coords = self.coords[nonzeros, :]

            if verbose:
                reco = M[:, nonzeros].matmul(self.coef)
                print(' done, training eps={:.5g}'.format(torch.norm(y-reco)/torch.norm(y)))
                print('{:.3f}s'.format(time.time() - start), flush=True)
                print()

        else:
            nonzeros = np.where(lars.coef_path_[:, argmin])[0]
            self.coef = torch.Tensor(lars.coef_path_[nonzeros, argmin])
            self.coords = self.coords[nonzeros, :]
            print()

    def predict(self, X):
        """
        Predict regression values for an input.

        :param X: a matrix of shape P X N floats (input features)
        :return: a vector of size P
        """
        return self._design_matrix((X - self.X_mean[None, :])/self.X_std[None, :]).matmul(self.coef)

    def to_tensor(self, domain=512, rmax=200, eps=1e-3, verbose=True):
        """
        Convert the interpolator to a TT-Tucker tensor.

        :param domain: one of the following:
            - list of N vectors to specify the tensor grid
            - integer I: the training data set's bounding box will be used, at resolution I. Default is 512
        :param rmax: the TT cores will be capped at this rank. Default is 500
        :param eps: rounding error to cast PCE into TT. Default is 1e-3
        :param verbose: Boolean; default is True
        :return: a `tntorch.Tensor` in the TT-Tucker format
        """

        N = len(self.Psis)
        S = self.Psis[0].shape[0]
        if not isinstance(domain, (list, tuple)):
            domain = [torch.linspace(
                self.bbox[n][0]+(self.bbox[n][1]-self.bbox[n][0])/(2*domain),
                self.bbox[n][1]-(self.bbox[n][1]-self.bbox[n][0])/(2*domain),
                domain) for n in range(N)]
        assert len(domain) == N
        domain_centered = [(domain[n] - self.X_mean[n])/self.X_std[n] for n in range(N)]

        if verbose:
            start = time.time()
            print('Conversion to TT-Tucker format (rmax={}, eps={:.5g})'.format(rmax, eps))
            print('{:.3f}s | '.format(time.time() - start), end='')
            print('Sparse TT-SVD...', end='', flush=True)

        # Assemble a TT-Tucker tensor:
        # The core is retrieved from the set of PCE coefficients found,
        # and is formed in the TT format using sparse TT-SVD
        t = tn.sparse_tt_svd(self.coords, self.coef, rmax=rmax, eps=eps)

        if verbose:
            eps = torch.norm(t[self.coords].torch()-self.coef)/torch.norm(self.coef)
            print(' done, rmax={}, eps={:.5g}'.format(max(t.ranks_tt), eps.item()))

        # The factors are the polynomial bases evaluated on the grid vectors
        Us = []
        for n in range(N):
            Us.append(
                (domain_centered[n][:, None]**torch.arange(S)).to(torch.get_default_dtype()).matmul(
                    self.Psis[n][:, :t.shape[n]])
            )
        t.Us = Us

        if verbose:
            print('{:.3f}s'.format(time.time() - start), flush=True)
            print()

        return t

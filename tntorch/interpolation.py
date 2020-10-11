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

    :param Xs: a P x N matrix
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

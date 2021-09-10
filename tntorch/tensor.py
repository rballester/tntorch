import numpy as np
import torch
import tntorch as tn
import time
from typing import Any, Optional, Sequence, Union

# The derivative of lstsq() is not implemented as of PyTorch 1.9.0,
# so we use cvxpylayer solver
def lstsq(
    b: torch.Tensor,
    A: torch.Tensor,
    method: Optional[str] = 'lstsq',
    lam: Optional[float] = 1e-3,
    eps: Optional[float] = 1e-3):

    if method == 'qr':
        Q, R = torch.linalg.qr(A.transpose(-1, -2))
        X = b.transpose(-1, -2) @ torch.linalg.pinv(R) @ Q.transpose(-1, -2)

        return X
    elif method == 'cvxpylayers':
        try:
            import cvxpy as cp
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Selected method requires optional cvxpy package, which can be installed by 'pip install cvxpy'.")
        try:
            from cvxpylayers.torch import CvxpyLayer
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Selected method requires optional cvxpylayers package, which can be installed by 'pip install cvxpylayers'.")

        Acp = cp.Parameter(A.shape)
        bcp = cp.Parameter(b.shape)
        x = cp.Variable((A.shape[-1], b.shape[-1]))

        obj = cp.sum_squares(Acp @ x - bcp) + lam * cp.pnorm(x, p=2)**2
        prob = cp.Problem(cp.Minimize(obj))
        prob_th = CvxpyLayer(prob, [Acp, bcp], [x])

        return prob_th(A, b, solver_args={'eps': eps})[0].transpose(-1, -2)
    elif method == 'lstsq':
        X = torch.linalg.lstsq(A, b)[0]
        return X.transpose(-1, -2)
    else:
        raise ValueError('Wrong value of method parameter, only qr and cvxpylayers are supported')


def _full_rank_tt(
    data: Union[torch.Tensor, Any],
    batch: Optional[bool] = False): # Naive TT formatting, don't even attempt to compress

    data = data.to(torch.get_default_dtype())
    shape = data.shape
    result = []

    data = data if isinstance(data, torch.Tensor) else torch.tensor(data)
    device = data.device

    if batch:
        N = data.dim() - 1
        resh = data.reshape([*shape[:2], -1])
        d1 = 1
        d2 = 2
    else:
        N = data.dim()
        resh = data.reshape([shape[0], -1])
        d1 = 0
        d2 = 1

    for n in range(1, N):
        if resh.shape[d1] < resh.shape[d2]:
            if batch:
                I = torch.eye(resh.shape[1], device=device).repeat(resh.shape[0], 1, 1)
                result.append(I.reshape([resh.shape[0], resh.shape[1] // shape[n], shape[n], resh.shape[1]]))
                resh = torch.reshape(resh, (resh.shape[0], resh.shape[1] * shape[n + 1], resh.shape[2] // shape[n + 1]))
            else:
                I = torch.eye(resh.shape[0], device=device)
                result.append(I.reshape([resh.shape[0] // shape[n - 1], shape[n - 1], resh.shape[0]]).to(device))
                resh = torch.reshape(resh, (resh.shape[0] * shape[n], resh.shape[1] // shape[n]))
        else:
            if batch:
                result.append(resh.reshape([resh.shape[0], resh.shape[1] // shape[n], shape[n], resh.shape[2]]))
                I = torch.eye(resh.shape[2], device=device).repeat(resh.shape[0], 1, 1)
                resh = I.reshape((resh.shape[0], resh.shape[2] * shape[n + 1], resh.shape[2] // shape[n + 1]))
            else:
                result.append(resh.reshape([resh.shape[0] // shape[n - 1], shape[n - 1], resh.shape[1]]))
                I = torch.eye(resh.shape[1], device=device)
                resh = I.reshape(resh.shape[1] * shape[n], resh.shape[1] // shape[n]).to(device)

    if batch:
        result.append(resh.reshape([resh.shape[0], resh.shape[1] // shape[N], shape[N], 1]))
    else:
        result.append(resh.reshape([resh.shape[0] // shape[N - 1], shape[N - 1], 1]))
    return result


class Tensor(object):

    """
    Class for all tensor networks. Currently supported: `tensor train (TT) <https://epubs.siam.org/doi/pdf/10.1137/090752286>`_, `CANDECOMP/PARAFAC (CP) <https://epubs.siam.org/doi/pdf/10.1137/07070111X>`_, `Tucker <https://epubs.siam.org/doi/pdf/10.1137/S0895479898346995>`_, and hybrid formats.

    Internal representation: an ND tensor has N cores, with each core following one of four options:

    - Size :math:`R_{n-1} \\times I_n \\times R_n` (standard TT core)
    - Size :math:`R_{n-1} \\times S_n \\times R_n` (TT-Tucker core), accompanied by an :math:`I_n \\times S_n` factor matrix
    - Size :math:`I_n \\times R` (CP factor matrix)
    - Size :math:`S_n \\times R_n` (CP-Tucker core), accompanied by an :math:`I_n \\times S_n` factor matrix
    """

    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray, Sequence[torch.Tensor]],
        Us: Optional[Union[torch.Tensor, Any]] = None,
        idxs: Optional[Any] = None,
        device: Optional[Any] = None,
        requires_grad: Optional[bool] = None,
        ranks_cp: Optional[Sequence[int]] = None,
        ranks_tucker: Optional[Sequence[int]] = None,
        ranks_tt: Optional[Sequence[int]] = None,
        eps: Optional[float] = None,
        max_iter: Optional[int] = 25,
        tol: Optional[float] = 1e-4,
        verbose: Optional[bool] = False,
        batch: Optional[bool] = False,
        algorithm: Optional[str] = 'svd',
        lstsq_algorithm: Optional[str] = 'qr'):

        """
        The constructor can either:

        - Decompose an uncompressed tensor

        - Use an explicit list of tensor cores (and optionally, factors)

        See `this notebook <https://github.com/rballester/tntorch/blob/master/tutorials/decompositions.ipynb>`_ for examples of use.

        :param data: a NumPy ndarray, PyTorch tensor, or a list of cores (which can represent either CP factors or TT cores)
        :param Us: optional list of Tucker factors
        :param idxs: annotate maskable tensors (*advanced users*)
        :param device: PyTorch device
        :param requires_grad: Boolean
        :param ranks_cp: an integer (or list)
        :param ranks_tucker: an integer (or list)
        :param ranks_tt: an integer (or list)
        :param eps: maximal error
        :param max_iter: maximum number of iterations when computing a CP decomposition using ALS
        :param tol: stopping criterion (change in relative error) when computing a CP decomposition using ALS
        :param verbose: Boolean
        :param batch: Boolean
        :param algorithm: 'svd' (default) or 'eig'. The latter can be faster, but less accurate
        :param lstsq_algorithm: 'qr' (default), 'cvxpylayers' or 'lstsq'. The latter is more accurate but doesn't allow backpropagation

        :return: a :class:`Tensor`
        """

        assert lstsq_algorithm in ('qr', 'lstsq', 'cvxpylayers')
        self.batch = batch

        if isinstance(data, (list, tuple)):
            if batch:
                min_dim = 3
                max_dim = 4
                d1 = 1
                d2 = 2
            else:
                min_dim = 2
                max_dim = 3
                d1 = 0
                d2 = 1

            if not all([min_dim <= d.dim() <= max_dim for d in data]): # add one dim for batch
                raise ValueError('All tensor cores must have 2 (for CP) or 3 (for TT) dimensions')
            for n in range(len(data)-1):
                if (data[n+1].dim() == max_dim and data[n].shape[-1] != data[n+1].shape[d1]) or (data[n+1].dim() == min_dim and data[n].shape[-1] != data[n+1].shape[d2]):
                    raise ValueError('Core ranks do not match')
            self.cores = data
            N = len(data)
        else:
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, device=device)
            elif isinstance(data, torch.Tensor):
                data = data.to(device)
            else:
                raise ValueError('A tntorch.Tensor may be built either from a list of cores, one NumPy ndarray, or one PyTorch tensor')
            N = data.dim()
        if Us is None:
            Us = [None] * N
        self.Us = Us
        if isinstance(data, torch.Tensor):
            if data.dim() == 0:
                data = data * torch.ones(1, device=device)
            if ranks_cp is not None:  # Compute CP from full tensor: CP-ALS
                if ranks_tt is not None:
                    raise ValueError('ALS for CP-TT is not yet supported')
                assert not hasattr(ranks_cp, '__len__')
                start = time.time()
                if verbose:
                    print('ALS', end='')
                if ranks_tucker is None: # We initialize CP factor to HOSVD
                    if batch:
                        data_norms = torch.sqrt(torch.sum(data**2, dim=list(range(1, data.dim()))))
                        N = data.dim() - 1
                    else:
                        data_norm = tn.norm(data)
                        N = data.dim()

                    self.cores = []
                    for n in range(N):
                        gram = tn.unfolding(data, n, batch)
                        gram = gram @ gram.transpose(-1, -2)
                        eigvals, eigvecs = torch.linalg.eigh(gram)

                        # Sort eigenvectors in decreasing importance
                        if batch:
                            assert len(eigvals[0]) == len(eigvals[-1])
                            reverse = torch.arange(len(eigvals[0]) - 1, -1, -1)
                            idx = torch.argsort(eigvals)[:, reverse[:ranks_cp]]
                            self.cores.append(eigvecs[[[i] for i in range(len(idx))], :, idx].transpose(-1, -2))
                            if self.cores[-1].shape[2] < ranks_cp:  # Complete with random entries
                                self.cores[-1] = torch.cat((
                                    self.cores[-1],
                                    torch.randn(
                                        self.cores[-1].shape[0],
                                        self.cores[-1].shape[1],
                                        ranks_cp-self.cores[-1].shape[2],
                                        device=device)), dim=2)
                        else:
                            reverse = torch.arange(len(eigvals)-1, -1, -1)
                            idx = torch.argsort(eigvals)[reverse[:ranks_cp]]
                            self.cores.append(eigvecs[:, idx])
                            if self.cores[-1].shape[1] < ranks_cp:  # Complete with random entries
                                self.cores[-1] = torch.cat((self.cores[-1], torch.randn(self.cores[-1].shape[0], ranks_cp-self.cores[-1].shape[1], device=device)), dim=1)
                else: # CP on Tucker's core
                    self.cores = _full_rank_tt(data, batch)
                    self.round_tucker(rmax=ranks_tucker, algorithm=algorithm)
                    data = self.tucker_core()

                    if batch:
                        data_norms = torch.sqrt(torch.sum(data**2, dim=list(range(1, data.dim()))))
                        self.cores = [torch.randn(data.shape[0], sh, ranks_cp, device=device) for sh in data.shape[1:]]
                    else:
                        data_norm = tn.norm(data)
                        self.cores = [torch.randn(sh, ranks_cp, device=device) for sh in data.shape]

                if verbose:
                    print(' -- initialization time =', time.time() - start)

                grams = [None] + [self.cores[n].transpose(-1, -2) @ self.cores[n] for n in range(1, self.dim())]

                if batch:
                    assert all([len(self.cores[i]) == len(self.cores[i + 1]) for i in range(self.dim() - 1)])
                    batch_size = len(self.cores[0])

                errors = []
                converged = False
                for iter in range(max_iter):
                    for n in range(self.dim()):
                        if batch:
                            khatri = torch.ones(batch_size, 1, ranks_cp, device=device)
                            prod = torch.ones(batch_size, ranks_cp, ranks_cp, device=device)
                            idxs = 'bir,bjr->bijr'
                            shape = [batch_size, -1, ranks_cp]
                        else:
                            khatri = torch.ones(1, ranks_cp, device=device)
                            prod = torch.ones(ranks_cp, ranks_cp, device=device)
                            idxs = 'ir,jr->ijr'
                            shape = [-1, ranks_cp]

                        for m in range(self.dim()-1, -1, -1):
                            if m != n:
                                prod *= grams[m]
                                khatri = torch.reshape(torch.einsum(idxs, (self.cores[m], khatri)), shape)

                        unfolding = tn.unfolding(data, n, batch)

                        unf_khatri_t = (unfolding @ khatri).transpose(-1, -2)
                        self.cores[n] = lstsq(unf_khatri_t, prod, lstsq_algorithm)
                        grams[n] = self.cores[n].transpose(-1, -2) @ self.cores[n]

                    if batch:
                        err = data - tn.Tensor(self.cores, batch=self.batch).torch()
                        errors.append((torch.sqrt(torch.sum(err**2, dim=list(range(1, err.dim())))) / data_norms).mean())
                    else:
                        errors.append(torch.norm(data - tn.Tensor(self.cores, batch=self.batch).torch()) / data_norm)
                    if len(errors) >= 2 and errors[-2] - errors[-1] < tol:
                        converged = True
                    if verbose:
                        print('iter: {: <{}} | eps: '.format(iter, len('{}'.format(max_iter))), end='')
                        print('{:.8f}'.format(errors[-1]), end='')
                        print(' | total time: {:9.4f}'.format(time.time() - start), end='')
                        if converged:
                            print(' <- converged (tol={})'.format(tol))
                        elif iter == max_iter-1:
                            print(' <- max_iter was reached: {}'.format(max_iter))
                        else:
                            print()
                    if converged:
                        break
            else:
                self.cores = _full_rank_tt(data, batch)
                self.Us = [None] * self.dim()

                if ranks_tucker is not None:
                    self.round_tucker(rmax=ranks_tucker, algorithm=algorithm)
                if ranks_tt is not None:
                    self.round_tt(rmax=ranks_tt, algorithm=algorithm)

        # Check factor shapes
        if batch:
            N = self.dim() - 1
            d1 = 3
            d2 = 2
        else:
            N = self.dim()
            d1 = 2
            d2 = 1

        for n in range(N):
            if self.Us[n] is None:
                continue
            assert self.Us[n].dim() == d1
            assert self.cores[n].shape[-2] == self.Us[n].shape[d2]

        # Set cores/Us requires_grad, if needed
        if requires_grad:
            for n in range(self.dim()):
                if self.Us[n] is not None:
                    self.Us[n].requires_grad_()
                self.cores[n].requires_grad_()

        if idxs is None:
            idxs = [torch.arange(sh, device=device) for sh in self.shape]
        self.idxs = idxs
        if eps is not None: # TT-SVD (or TT-EIG) algorithm
            if ranks_cp is not None or ranks_tucker is not None or ranks_tt is not None:
                raise ValueError('Specify eps or ranks, but not both')
            self.round(eps)

    """
    Arithmetic operations
    """

    def __add__(
        self,
        other: Union[Any, int, float]):

        if not isinstance(other, Tensor): # A scalar
            factor = other

            if self.batch:
                other = Tensor([torch.ones([self.shape[0], 1, self.shape[n + 1], 1]) for n in range(self.dim())], batch=True)
            else:
                other = Tensor([torch.ones([1, self.shape[n], 1]) for n in range(self.dim())])

            other.cores[0].data *= factor
            other.to(self.cores[0].device)
        
        if self.batch != other.batch:
            raise ValueError('Tensors with the same batch mode are supported')
        if self.batch:
            assert self.shape[0] == other.shape[0], f'Batch dim must match, got {self.shape[0]} and {other.shape[0]}'

        if self.dim() == 1: # Special case
            return Tensor([self.decompress_tucker_factors().cores[0] + other.decompress_tucker_factors().cores[0]])

        if self.batch:
            idxs = 'bijk,baj->biak'
            m = 3
            d = 1
        else:
            idxs = 'ijk,aj->iak'
            m = 2
            d = 0

        this, other = _broadcast(self, other)
        cores = []
        Us = []
        for n in range(this.dim()):
            core1 = this.cores[n]
            core2 = other.cores[n]

            # CP + CP -> CP, other combinations -> TT
            if core1.dim() == m and core2.dim() == m:
                if self.batch:
                    core1 = core1[:, None]
                    core2 = core2[:, None]
                else:
                    core1 = core1[None]
                    core2 = core2[None]
            else:
                core1 = self._cp_to_tt(core1)
                core2 = self._cp_to_tt(core2)

            if this.Us[n] is not None and other.Us[n] is not None:
                if self.batch:
                    slice1 = torch.cat([core1, torch.zeros(core2.shape[0], core1.shape[1], core1.shape[2], core1.shape[3])], dim=1)
                    slice1 = torch.cat([slice1, torch.zeros(core1.shape[0], core1.shape[1] + core2.shape[1], core1.shape[2], core2.shape[3])], dim=3)
                    slice2 = torch.cat([torch.zeros(core1.shape[0], core1.shape[1], core2.shape[2], core2.shape[3]), core2], dim=1)
                    slice2 = torch.cat([torch.zeros(core1.shape[0], core1.shape[1]+core2.shape[1], core2.shape[2], core1.shape[3]), slice2], dim=3)
                    c = torch.cat([slice1, slice2], dim=2)
                    Us.append(torch.cat((self.Us[n], other.Us[n]), dim=2))
                else:
                    slice1 = torch.cat([core1, torch.zeros([core2.shape[0], core1.shape[1], core1.shape[2]])], dim=0)
                    slice1 = torch.cat([slice1, torch.zeros(core1.shape[0] + core2.shape[0], core1.shape[1], core2.shape[2])], dim=2)
                    slice2 = torch.cat([torch.zeros([core1.shape[0], core2.shape[1], core2.shape[2]]), core2], dim=0)
                    slice2 = torch.cat([torch.zeros(core1.shape[0]+core2.shape[0], core2.shape[1], core1.shape[2]), slice2], dim=2)
                    c = torch.cat([slice1, slice2], dim=1)
                    Us.append(torch.cat((self.Us[n], other.Us[n]), dim=1))

                cores.append(c)
                continue

            if this.Us[n] is not None:
                core1 = torch.einsum(idxs, (core1, self.Us[n]))
            if other.Us[n] is not None:
                core2 = torch.einsum(idxs, (core2, other.Us[n]))

            if self.batch:
                c1_st = torch.zeros([core2.shape[0], core2.shape[1], this.shape[n + 1], core1.shape[3]], device=core1.device)
                c2_st = torch.zeros([core1.shape[0], core1.shape[1], this.shape[n + 1], core2.shape[3]], device=core2.device)
            else:
                c1_st = torch.zeros([core2.shape[0], this.shape[n], core1.shape[2]], device=core1.device)
                c2_st = torch.zeros([core1.shape[0], this.shape[n], core2.shape[2]], device=core2.device)


            column1 = torch.cat([core1, c1_st], dim=d)
            column2 = torch.cat([c2_st, core2], dim=d)
            c = torch.cat([column1, column2], dim=m)
            cores.append(c)
            Us.append(None)

        # First core should have first size 1 (if it's TT)
        if not (this.cores[0].dim() == m and other.cores[0].dim() == m):
            cores[0] = cores[0].sum(dim=d, keepdim=True)
        # Similarly for the last core and last size
        if not (this.cores[-1].dim() == m and other.cores[-1].dim() == m):
            cores[-1] = cores[-1].sum(dim=m, keepdim=True)

        # Set up cores that should be CP cores
        for n in range(0, this.dim()):
            if this.cores[n].dim() == m and other.cores[n].dim() == m:
                cores[n] = cores[n].sum(dim=d, keepdim=False)

        return Tensor(cores, Us=Us, batch=self.batch)

    def __radd__(
        self,
        other: Union[Any, torch.Tensor]):

        if other is None:
            return self
        return self + other

    def __sub__(
        self,
        other: Union[Any, torch.Tensor]):

        return self + -1 * other

    def __rsub__(
        self,
        other: Union[Any, torch.Tensor]):

        return -1 * self + other

    def __neg__(self):
        return -1 * self

    def __mul__(
        self,
        other: Union[Any, int, float]):

        if not isinstance(other, Tensor):  # A scalar
            result = self.clone()
            result.cores[0].data *= other
            return result

        if self.batch:
            m = 3
            idx1 = 'gijk,gabc->giajbkc'
            idx2 = 'bij,bik->bijk'
            idx3 = 'bijk,baj->biak'
        else:
            idx1 = 'ijk,abc->iajbkc'
            idx2 = 'ij,ik->ijk'
            idx3 = 'ijk,aj->iak'
            m = 2

        this, other = _broadcast(self, other)
        cores = []
        Us = []
        for n in range(this.dim()):
            core1 = this.cores[n]
            core2 = other.cores[n]
            # CP + CP -> CP, other combinations -> TT
            if core1.dim() == m and core2.dim() == m:
                if self.batch:
                    core1 = core1[:, None]
                    core2 = core2[:, None]
                else:
                    core1 = core1[None]
                    core2 = core2[None]
            else:
                core1 = this._cp_to_tt(core1)
                core2 = this._cp_to_tt(core2)

            # We do the product core along 3 axes, unless it would blow up
            if self.batch:
                d1 = this.cores[n].shape[2] * other.cores[n].shape[2]
                
                shape1 = (
                    core1.shape[0],
                    core1.shape[1] * core2.shape[1],
                    core1.shape[2] * core2.shape[2],
                    core1.shape[3] * core2.shape[3])
                if this.Us[n] is not None:
                    shape2 = (this.Us[n].shape[0], this.Us[n].shape[1], -1)
            else:
                d1 = this.cores[n].shape[1] * other.cores[n].shape[1]
                
                shape1 = (
                    core1.shape[0] * core2.shape[0],
                    core1.shape[1] * core2.shape[1],
                    core1.shape[2] * core2.shape[2])
                if this.Us[n] is not None:
                    shape2 = (this.Us[n].shape[0], -1)

            if this.Us[n] is not None and other.Us[n] is not None and d1 < this.shape[n]:
                cores.append(torch.reshape(torch.einsum(idx1, (core1, core2)), shape1))
                Us.append(torch.reshape(torch.einsum(idx2, (this.Us[n], other.Us[n])), shape2))
            else: # Decompress spatially, then do normal TT-TT slice-wise kronecker product
                if this.Us[n] is not None:
                    core1 = torch.einsum(idx3, (core1, this.Us[n]))
                if other.Us[n] is not None:
                    core2 = torch.einsum(idx3, (core2, other.Us[n]))
                cores.append(_core_kron(core1, core2, self.batch))
                Us.append(None)

            if this.cores[n].dim() == m and other.cores[n].dim() == m:
                if self.batch:
                    cores[-1] = cores[-1][:, 0]
                else:
                    cores[-1] = cores[-1][0]
        return tn.Tensor(cores, Us=Us, batch=self.batch)

    def __truediv__(
        self,
        other: Union[Any, torch.Tensor]):

        return tn.cross(function=lambda x, y: x / y, tensors=[self, other], verbose=False)

    def __rtruediv__(
        self,
        other: Union[Any, torch.Tensor]):

        return tn.cross(function=lambda x, y: x / y, tensors=[tn.full_like(self, fill_value=other), self], verbose=False)

    def __pow__(
        self,
        other: Union[Any, torch.Tensor]):

        return tn.cross(function=lambda x, y: x**y, tensors=[self, tn.full_like(self, fill_value=power)], verbose=False)

    def __rmul__(
        self,
        other: Union[Any, torch.Tensor]):

        return self * other

    def __truediv__(
        self,
        other: Union[Any, torch.Tensor]):

        return self * (1./other)

    """
    Boolean logic
    """

    def __invert__(self):
        return 1 - self

    def __and__(
        self,
        other: Union[Any, torch.Tensor]):

        return self * other

    def __or__(
        self,
        other: Union[Any, torch.Tensor]):

        return self + other - self * other

    def __xor__(
        self,
        other: Union[Any, torch.Tensor]):

        return self + other - 2 * self * other

    def __eq__(
        self,
        other: Union[Any, torch.Tensor]):

        return tn.dist(self, other) <= 1e-14

    def __ne__(
        self,
        other: Union[Any, torch.Tensor]):

        return not self == other

    """
    Shapes and ranks
    """

    @property
    def shape(self):
        """
        Returns the shape of this tensor.

        :return: a PyTorch shape object
        """

        shape = []

        if self.batch:
            shape.append(len(self.cores[0]))

        for n in range(self.dim()):
            if self.Us[n] is None:
                shape.append(self.cores[n].shape[-2])
            else:
                shape.append(self.Us[n].shape[-2])
        return torch.Size(shape)

    def b(self):
        if not self.batch:
            raise ValueError
        return self.cores[0].shape[0]

    @property
    def ranks_tt(self):
        """
        Returns the TT ranks of this tensor.

        :return: a vector of integers
        """

        if self.batch:
            d1 = 3
            d2 = 2
            d3 = 1
        else:
            d1 = 2
            d2 = 1
            d3 = 0

        if self.cores[0].dim() == d1:
            first = self.cores[0].shape[d2]
        else:
            first = self.cores[0].shape[d3]

        return torch.tensor([first] + [c.shape[-1] for c in self.cores])

    @ranks_tt.setter
    def ranks_tt(
        self,
        value: int):

        self.round_tt(rmax=value)

    @property
    def ranks_tucker(self):
        """
        Returns the Tucker ranks of this tensor.

        :return: a vector of integers
        """

        return torch.tensor([c.shape[-2] for c in self.cores])

    @ranks_tucker.setter
    def ranks_tucker(
        self,
        value: int):

        self.round_tucker(rmax=value)

    def dim(self):
        """
        Returns the number of cores of this tensor.

        :return: an int
        """

        return len(self.cores)

    def size(self):
        """
        Alias for :meth:`shape` (as PyTorch does)
        """

        return self.shape

    def __repr__(self):
        format = []
        if any([(c.dim() == 3 and not self.batch) or (c.dim() == 4 and self.batch) for c in self.cores]):
            format.append('TT')
        if any([(c.dim() == 2 and not self.batch) or (c.dim() == 3 and self.batch) for c in self.cores]):
            format.append('CP')
        if any([U is not None for U in self.Us]):
            format.append('Tucker')
        format = '-'.join(format)
        s = '{}D {} tensor:\n'.format(self.dim(), format)
        s += '\n'
        ttr = self.ranks_tt
        tuckerr = self.ranks_tucker

        if self.batch:
            s+= 'with batch = {}\n'.format(self.cores[0].shape[0])

        if any([U is not None for U in self.Us]):
            # Shape
            row = [' ']*(4*self.dim()-1)
            shape = self.shape
            for n in range(self.dim()):
                if self.Us[n] is None:
                    continue
                lenn = len('{}'.format(shape[n]))
                row[n * 4-lenn // 2 + 2:n * 4 - lenn // 2 + lenn + 2] = '{}'.format(shape[n])
            s += ''.join(row)
            s += '\n'

        # Tucker ranks
        row = [' '] * (4 * self.dim() - 1)
        for n in range(self.dim()):
            if self.Us[n] is None:
                lenr = len('{}'.format(tuckerr[n]))
                row[n * 4 - lenr // 2 + 2:n * 4 - lenr // 2 + lenr + 2] = '{}'.format(tuckerr[n])
            else:
                row[n * 4 + 2:n * 4 + 3] = '|'
        s += ''.join(row)
        s += '\n'

        row = [' '] * (4 * self.dim() - 1)
        for n in range(self.dim()):
            if self.Us[n] is None:
                row[n * 4 + 2:n * 4 + 3] = '|'
            else:
                lenr = len('{}'.format(tuckerr[n]))
                row[n * 4 - lenr // 2 + 2:n * 4 - lenr // 2 + lenr + 2] = '{}'.format(tuckerr[n])
        s += ''.join(row)
        s += '\n'

        # Nodes
        row = [' '] * (4 * self.dim() - 1)
        for n in range(self.dim()):
            if self.cores[n].dim() == 2:
                nodestr = '<{}>'.format(n)
            else:
                nodestr = '({})'.format(n)
            lenn = len(nodestr)
            row[(n + 1) * 4 - (lenn - 1) // 2:(n + 1) * 4 - (lenn - 1) // 2 + lenn] = nodestr
        s += ''.join(row[2:])
        s += '\n'

        # TT rank bars
        s += ' / \\'*self.dim()
        s += '\n'

        # Bottom: TT/CP ranks
        row = [' '] * (4 * self.dim())
        for n in range(self.dim() + 1):
            lenr = len('{}'.format(ttr[n]))
            row[n * 4:n * 4 + lenr] = '{}'.format(ttr[n])
        s += ''.join(row)
        s += '\n'

        return s

    """
    Decompression
    """

    def _process_key(
        self,
        key: Union[Sequence[int], torch.Tensor, int, Any]):
        if not hasattr(key, '__len__'):
            key = (key,)
        fancy = False
        if isinstance(key, torch.Tensor):
            key = key.detach().numpy()
        if any([not np.isscalar(k) for k in key]):  # Fancy
            key = list(key)
            fancy = True
        if isinstance(key, tuple):
            key = list(key)
        elif not fancy:
            key = [key]

        # Process ellipsis, if any
        nonecount = sum(1 for k in key if k is None)
        for i in range(len(key)):
            if key[i] is Ellipsis:
                key = key[:i] + [slice(None)] * (len(self.shape) - (len(key) - nonecount) + 1) + key[i + 1:]
                break
        if any([k is Ellipsis for k in key]):
            raise IndexError('Only one ellipsis is allowed, at most')
        if len(self.shape) - (len(key) - nonecount) < 0:
            raise IndexError('Too many index entries {} vs {}'.format(len(self.shape), len(key) - nonecount))

        # Fill remaining unspecified dimensions with slice(None)
        key = key + [slice(None)] * (len(self.shape) - (len(key) - nonecount))
        return key

    def __getitem__(
        self,
        key: Union[Sequence[int], torch.Tensor, int, Any]):
        """
        NumPy-style indexing for compressed tensors. There are 5 accessors supported: slices, index arrays, integers,
        None, or another Tensor (selection via binary indexing)

        - Index arrays can be lists, tuples, or vectors
        - All index arrays must have the same length P
        - In NumPy, index arrays and slices can be interleaved. We do not admit that, as it requires expensive transpose operations

        """

        # Preprocessing
        if isinstance(key, Tensor):
            if torch.abs(tn.sum(key) - 1) > 1e-8:
                raise ValueError('When indexing via a mask tensor, that mask should have exactly 1 accepting string')
            s = tn.accepted_inputs(key)[0]
            slicing = []
            for n in range(len(self.shape)):
                idx = self.idxs[n].long()
                idx[idx > 1] = 1
                idx = torch.where(idx == s[n])[0]
                sl = slice(idx[0], idx[-1] + 1)
                lenidx = len(idx)
                if lenidx == 1:
                    sl = idx.item()
                slicing.append(sl)
            return self[slicing]

        if isinstance(key, torch.Tensor):
            key = key.clone().detach().cpu().long()
        if isinstance(key, np.ndarray) or isinstance(key, torch.Tensor) and key.ndim == 2:
            key = [key[:, col] for col in range(key.shape[1])]

        device = self.cores[0].device
        key = self._process_key(key)

        if self.batch:
            batch_dim_processed = False
            batch_dim_idx = slice(self.shape[0])
            batch_size = self.shape[0]

        last_mode = None
        factors = {'int': None, 'index': None, 'index_done': False}
        cores = []
        Us = []
        counter = 0
        first_index_dim = None

        def join_cores(c1, c2):
            if self.batch:
                if c1.dim() == 2 and c2.dim() == 3:
                    return torch.einsum('bi,bai->bai', (c1, c2))
                elif c1.dim() == 3 and c2.dim() == 3:
                    return torch.einsum('bij,baj->biaj', (c1, c2))
                elif c1.dim() == 2 and c2.dim() == 4:
                    return torch.einsum('bi,biaj->biaj', (c1, c2))
                elif c1.dim() == 3 and c2.dim() == 4:
                    return torch.einsum('bij,bjak->biak', (c1, c2))
                else:
                    raise ValueError
            else:
                if c1.dim() == 1 and c2.dim() == 2:
                    return torch.einsum('i,ai->ai', (c1, c2))
                elif c1.dim() == 2 and c2.dim() == 2:
                    return torch.einsum('ij,aj->iaj', (c1, c2))
                elif c1.dim() == 1 and c2.dim() == 3:
                    return torch.einsum('i,iaj->iaj', (c1, c2))
                elif c1.dim() == 2 and c2.dim() == 3:
                    return torch.einsum('ij,jak->iak', (c1, c2))
                else:
                    raise ValueError

        def insert_core(factors, core=None, key=None, U=None):
            if factors['index'] is not None:
                if factors['int'] is not None:
                    factors['index'] = join_cores(factors['int'], factors['index'])
                    factors['int'] = None
                cores.append(factors['index'])
                Us.append(None)
                factors['index'] = None
                factors['index_done'] = True
            if core is not None:
                if factors['int'] is not None: # There is a previous 1D/2D core (CP/Tucker) from an integer slicing
                    if U is None:
                        Us.append(None)
                        nCore = core[..., key, :]
                        if self.batch:
                            nCore = nCore[batch_dim_idx]
                            if isinstance(batch_dim_idx, (int, np.integer)):
                                nCore = nCore[None, ...]

                        cores.append(join_cores(factors['int'], nCore))
                    else:
                        nU = U[..., key, :]
                        nCore = core
                        if self.batch:
                            nCore = nCore[batch_dim_idx]
                            nU = nU[batch_dim_idx]
                            if isinstance(batch_dim_idx, (int, np.integer)):
                                nU = nU[None, ...]
                                nCore = nCore[None, ...]

                        cores.append(join_cores(factors['int'], nCore))
                        Us.append(nU)
                    factors['int'] = None
                else: # Easiest case
                    if U is None:
                        Us.append(None)
                        nCore = core[..., key, :]
                        if self.batch:
                            nCore = nCore[batch_dim_idx]
                            if isinstance(batch_dim_idx, (int, np.integer)):
                                nCore = nCore[None, ...]

                        cores.append(nCore)
                    else:
                        nU = U[..., key, :]
                        nCore = core
                        if self.batch:
                            nCore = nCore[batch_dim_idx]
                            nU = nU[batch_dim_idx]
                            if isinstance(batch_dim_idx, (int, np.integer)):
                                nU = nU[None, ...]
                                nCore = nCore[None, ...]

                        cores.append(nCore)
                        Us.append(nU)

        def get_key(
            counter: int,
            key: Union[Sequence[int], torch.Tensor, int, Any]):
            if self.Us[counter] is None:
                if self.batch:
                    nCore = self.cores[counter][..., key, :][batch_dim_idx]
                    if isinstance(batch_dim_idx, (int, np.integer)):
                        return nCore[None, ...]
                    else:
                        return nCore
                else:
                    return self.cores[counter][..., key, :]
            else:
                sl = self.Us[counter][..., key, :]

                if self.batch:
                    sl = sl[batch_dim_idx]
                    nCore = self.cores[counter][batch_dim_idx]
                    if isinstance(batch_dim_idx, (int, np.integer)):
                        sl = sl[None, ...]
                        nCore = nCore[None, ...]

                    if sl.dim() == 2:  # key is an int
                        if nCore.dim() == 4:
                            return torch.einsum('bijk,bj->bik', (nCore, sl))
                        else:
                            return torch.einsum('bji,bj->bi', (nCore, sl))
                    else:
                        if nCore.dim() == 4:
                            return torch.einsum('bijk,baj->biak', (nCore, sl))
                        else:
                            return torch.einsum('bji,baj->bai', (nCore, sl))
                else:
                    if sl.dim() == 1: # key is an int
                        if self.cores[counter].dim() == 3:
                            return torch.einsum('ijk,j->ik', (self.cores[counter], sl))
                        else:
                            return torch.einsum('ji,j->i', (self.cores[counter], sl))
                    else:
                        if self.cores[counter].dim() == 3:
                            return torch.einsum('ijk,aj->iak', (self.cores[counter], sl))
                        else:
                            return torch.einsum('ji,aj->ai', (self.cores[counter], sl))

        for i in range(len(key)):
            if hasattr(key[i], '__len__'):
                this_mode = 'index'
            elif key[i] is None:
                this_mode = 'none'
            elif isinstance(key[i], (int, np.integer)):
                this_mode = 'int'
            elif isinstance(key[i], slice):
                this_mode = 'slice'
            else:
                raise IndexError

            if this_mode == 'none':
                if self.batch:
                    if batch_dim_processed:
                        core = torch.cat([torch.eye(self.ranks_tt[counter - 1].item())[None, ...] for _ in range(batch_size)])
                        insert_core(
                            factors,
                            core[:, :, None, :],
                            key=slice(None),
                            U=None
                        )
                    else:
                        raise ValueError('Cannot change batch dimension')
                else:
                    insert_core(factors, torch.eye(self.ranks_tt[counter].item())[:, None, :], key=slice(None), U=None)
            elif this_mode == 'slice':
                if self.batch:
                    if batch_dim_processed:
                        insert_core(factors, self.cores[counter - 1], key=key[i], U=self.Us[counter - 1])
                    else:
                        batch_dim_processed = True
                        batch_dim_idx = key[i]
                else:
                    insert_core(factors, self.cores[counter], key=key[i], U=self.Us[counter])

                counter += 1
            elif this_mode == 'index':
                if self.batch and first_index_dim == 0:
                    raise ValueError('Advanced indexing is prohibited for batch dimension')
                if factors['index_done']:
                    raise IndexError("All index arrays must appear contiguously")
                if factors['index'] is None:
                    if self.batch:
                        if first_index_dim is None:
                            first_index_dim = i

                        if batch_dim_processed:
                            factors['index'] = get_key(counter - 1, key[i])
                        else:
                            batch_dim_processed = True
                            batch_dim_idx = key[i]
                    else:
                        factors['index'] = get_key(counter, key[i])
                else:
                    if factors['index'].shape[-2] != len(key[i]):
                            raise ValueError('Index arrays must have the same length')
                    a1 = factors['index']

                    if self.batch:
                        a2 = get_key(counter - 1, key[i]).to(device)
                        if a1.dim() == 3 and a2.dim() == 3:
                            factors['index'] = torch.einsum('bai,bai->bai', (a1, a2))
                        elif a1.dim() == 3 and a2.dim() == 4:
                            factors['index'] = torch.einsum('bai,biaj->biaj', (a1, a2))
                        elif a1.dim() == 4 and a2.dim() == 3:
                            factors['index'] = torch.einsum('biaj,baj->biaj', (a1, a2))
                        elif a1.dim() == 4 and a2.dim() == 4:
                            factors['index'] = torch.einsum('biaj,bjak->biak', (a1, a2))
                    else:
                        a2 = get_key(counter, key[i]).to(device)

                        if a1.dim() == 2 and a2.dim() == 2:
                            factors['index'] = torch.einsum('ai,ai->ai', (a1, a2))
                        elif a1.dim() == 2 and a2.dim() == 3:
                            factors['index'] = torch.einsum('ai,iaj->iaj', (a1, a2))
                        elif a1.dim() == 3 and a2.dim() == 2:
                            factors['index'] = torch.einsum('iaj,aj->iaj', (a1, a2))
                        elif a1.dim() == 3 and a2.dim() == 3:
                            factors['index'] = torch.einsum('iaj,jak->iak', (a1, a2))

                counter += 1
            elif this_mode == 'int':
                if self.batch:
                    if batch_dim_processed:
                        if last_mode == 'index':
                            insert_core(factors)
                        if factors['int'] is None:
                            factors['int'] = get_key(counter - 1, key[i])
                        else:
                            c1 = factors['int']
                            c2 = get_key(counter - 1, key[i])

                            if c1.dim() == 2 and c2.dim() == 2:
                                factors['int'] = torch.einsum('bi,bi->bi', (c1, c2))
                            elif c1.dim() == 2 and c2.dim() == 3:
                                factors['int'] = torch.einsum('bi,bij->bij', (c1, c2))
                            elif c1.dim() == 3 and c2.dim() == 2:
                                factors['int'] = torch.einsum('bij,bj->bij', (c1, c2))
                            elif c1.dim() == 3 and c2.dim() == 3:
                                factors['int'] = torch.einsum('bij,bjk->bik', (c1, c2))
                    else:
                        batch_dim_processed = True
                        batch_dim_idx = key[i]
                else:
                    if last_mode == 'index':
                        insert_core(factors)
                    if factors['int'] is None:
                        factors['int'] = get_key(counter, key[i])
                    else:
                        c1 = factors['int']
                        c2 = get_key(counter, key[i])

                        if c1.dim() == 1 and c2.dim() == 1:
                            factors['int'] = torch.einsum('i,i->i', (c1, c2))
                        elif c1.dim() == 1 and c2.dim() == 2:
                            factors['int'] = torch.einsum('i,ij->ij', (c1, c2))
                        elif c1.dim() == 2 and c2.dim() == 1:
                            factors['int'] = torch.einsum('ij,j->ij', (c1, c2))
                        elif c1.dim() == 2 and c2.dim() == 2:
                            factors['int'] = torch.einsum('ij,jk->ik', (c1, c2))
                counter += 1
            last_mode = this_mode

        # At the end: handle possibly pending factors
        if last_mode == 'index':
            insert_core(factors, core=None, key=None, U=None)
        elif last_mode == 'int':
            if len(cores) > 0:  # We return a tensor: absorb existing cores with int factor
                if self.batch:
                    nCore = cores[-1][batch_dim_idx]
                    if isinstance(batch_dim_idx, (int, np.integer)):
                        nCore = nCore[None, ...]

                    if nCore.dim() == 3 and factors['int'].dim() == 2:
                        cores[-1] = torch.einsum('bai,bi->bai', (nCore, factors['int']))
                    elif nCore.dim() == 3 and factors['int'].dim() == 3:
                        cores[-1] = torch.einsum('bai,bij->biaj', (nCore, factors['int']))
                    elif nCore.dim() == 4 and factors['int'].dim() == 2:
                        cores[-1] = torch.einsum('biaj,bj->bai', (nCore, factors['int']))
                    elif nCore.dim() == 4 and factors['int'].dim() == 3:
                        cores[-1] = torch.einsum('biaj,bjk->biak', (nCore, factors['int']))
                else:
                    if cores[-1].dim() == 2 and factors['int'].dim() == 1:
                        cores[-1] = torch.einsum('ai,i->ai', (cores[-1], factors['int']))
                    elif cores[-1].dim() == 2 and factors['int'].dim() == 2:
                        cores[-1] = torch.einsum('ai,ij->iaj', (cores[-1], factors['int']))
                    elif cores[-1].dim() == 3 and factors['int'].dim() == 1:
                        cores[-1] = torch.einsum('iaj,j->ai', (cores[-1], factors['int']))
                    elif cores[-1].dim() == 3 and factors['int'].dim() == 2:
                        cores[-1] = torch.einsum('iaj,jk->iak', (cores[-1], factors['int']))
            else:  # We return a scalar
                if not self.batch and factors['int'].numel() > 1:
                    return torch.sum(factors['int'])
                return torch.squeeze(factors['int'])

        if self.batch and (isinstance(batch_dim_idx, (int, np.integer))):
            nUs = []
            for U in Us:
                if U is None:
                    nUs.append(U)
                else:
                    nUs.append(U[0])

            return tn.Tensor([core[0] for core in cores], Us=nUs, batch=False)
        else:
            return tn.Tensor(cores, Us=Us, batch=self.batch)

    def __setitem__(
            self,
            key: Union[Sequence[int], torch.Tensor, int, Any],
            value: Any):

        key = self._process_key(key)
        scalar = False
        if isinstance(value, np.ndarray):
            value = tn.Tensor(torch.tensor(value), batch=self.batch)
        elif isinstance(value, torch.Tensor):
            if value.dim() == 0:
                value = value.item()
                scalar = True
            else:
                if self.batch:
                    if isinstance(key[0], int):
                        value = value[None]
                    if len(value.shape) == 1:
                        value = value[:, None]

                value = tn.Tensor(value, batch=self.batch)
        elif isinstance(value, tn.Tensor):
            pass
        else:  # It's a scalar
            scalar = True

        subtract_cores = []
        add_cores = []

        key_length = len(key)
        if self.batch:
            key_length -= 1

        for i in range(key_length):
            if not isinstance(key[i], slice) and not hasattr(key[i], '__len__'):
                key[i] = slice(key[i], key[i] + 1)

            subtract_core = torch.zeros_like(self.cores[i])
            if self.batch:
                chunk = self.cores[i][key[0], ..., key[i + 1], :]
                subtract_core[key[0], ..., key[i + 1], :] += chunk
                sh = chunk.shape[2]
                k = i + 1
            else:
                chunk = self.cores[i][..., key[i], :]
                subtract_core[..., key[i], :] += chunk
                sh = chunk.shape[1]
                k = i
            
            subtract_cores.append(subtract_core)
            if scalar:
                if self.batch:
                    if self.cores[i].dim() == 4:
                        add_core = torch.zeros(self.shape[0], 1, self.shape[i + 1], 1)
                    else:
                        add_core = torch.zeros(self.shape[0], self.shape[i + 1], 1)
                    
                    add_core[key[0], ..., key[i + 1], :] += 1
                    if i == 0:
                        add_core *= value
                else:
                    if self.cores[i].dim() == 3:
                        add_core = torch.zeros(1, self.shape[i], 1)
                    else:
                        add_core = torch.zeros(self.shape[i], 1)

                    add_core[..., key[i], :] += 1
                    if i == 0:
                        add_core *= value
            else:
                if len(value.shape) != len(key):
                    if k == len(value.shape) - 1:
                        value = value[..., None]
                    else:
                        if sh == 1:
                            if value.shape[k] == sh:
                                value = value[..., None]
                            else:
                                current_shape = list(value.shape)
                                new_shape = current_shape[:k] + [1] + current_shape[k:]
                                value = tn.Tensor(value.torch().reshape(new_shape), batch=self.batch)

                if self.batch:
                    if self.cores[i].dim() == 4:
                        add_core = torch.zeros(self.cores[i].shape[0], value.cores[i].shape[1], self.shape[i + 1], value.cores[i].shape[3])
                    else:
                        add_core = torch.zeros(self.cores[i].shape[0], self.shape[i + 1], value.cores[i].shape[2])

                    if isinstance(key[i + 1], int):
                        add_core[key[0], ..., key[i + 1], :] += value.cores[i][..., 0, :]
                    else:
                        add_core[key[0], ..., key[i + 1], :] += value.cores[i]
                else:
                    if chunk.shape[1] != value.shape[i]:
                        raise ValueError('{}-th dimension mismatch in tensor assignment: {} (lhs) != {} (rhs)'.format(i, chunk.shape[1], value.shape[i]))
                    if self.cores[i].dim() == 3:
                        add_core = torch.zeros(value.cores[i].shape[0], self.shape[i], value.cores[i].shape[2])
                    else:
                        add_core = torch.zeros(self.shape[i], value.cores[i].shape[1])

                    add_core[..., key[i], :] += value.cores[i]
            add_cores.append(add_core)
        result = self - tn.Tensor(subtract_cores, batch=self.batch) + tn.Tensor(add_cores, batch=self.batch)
        self.__init__(result.cores, result.Us, self.idxs, batch=self.batch)

    def tucker_core(self):
        """
        If this is a Tucker-like tensor, returns its Tucker core as an explicit PyTorch tensor.

        If this tensor does not have Tucker factors, then it returns the full decompressed tensor.

        :return: a PyTorch tensor
        """

        return tn.Tensor(self.cores, batch=self.batch).torch()

    def decompress_tucker_factors(
        self,
        dim: Optional[Union[str, Any]] = 'all',
        _clone: Optional[bool] = True):

        """
        Decompresses this tensor along the Tucker factors only.

        :param dim: int, list, or 'all' (default)

        :return: a :class:`Tensor` in CP/TT format, without Tucker factors
        """

        if dim == 'all':
            dim = range(self.dim())
        if not hasattr(dim, '__len__'):
            dim = [dim] * self.dim()

        cores = []
        Us = []
        for n in range(self.dim()):
            if n in dim and self.Us[n] is not None:
                if self.batch:
                    if self.cores[n].dim() == 3:
                        cores.append(torch.einsum('bjk,baj->bak', (self.cores[n], self.Us[n])))
                    else:
                        cores.append(torch.einsum('bijk,baj->biak', (self.cores[n], self.Us[n])))
                else:
                    if self.cores[n].dim() == 2:
                        cores.append(torch.einsum('jk,aj->ak', (self.cores[n], self.Us[n])))
                    else:
                        cores.append(torch.einsum('ijk,aj->iak', (self.cores[n], self.Us[n])))

                Us.append(None)
            else:
                if _clone:
                    cores.append(self.cores[n].clone())
                    if self.Us[n] is not None:
                        Us.append(self.Us[n].clone())
                    else:
                        Us.append(None)
                else:
                    cores.append(self.cores[n])
                    Us.append(self.Us[n])
        return tn.Tensor(cores, Us, idxs=self.idxs, batch=self.batch)

    def tt(self):
        """
        Casts this tensor as a pure TT format.

        :return: a :class:`Tensor` in the TT format
        """

        t = self.decompress_tucker_factors()
        t._cp_to_tt()
        return t

    def torch(self):
        """
        Decompresses this tensor into a PyTorch tensor.

        :return: a PyTorch tensor
        """

        t = self.decompress_tucker_factors(_clone=False)

        device = t.cores[0].device
        if self.batch:
            m = 3
            idx1 = 'gai,gbi->gabi'
            idx2 = 'gai,gbi->gab'
            idx3 = 'gai,gibj->gabj'
            batch_size = self.cores[0].shape[0]
            shape = [batch_size]
            factor = torch.ones(batch_size, 1, self.ranks_tt[0]).to(device)
        else:
            m = 2
            idx1 = 'ai,bi->abi'
            idx2 = 'ai,bi->ab'
            idx3 = 'ai,ibj->abj'
            factor = torch.ones(1, self.ranks_tt[0]).to(device)
            shape = []

        for n in range(t.dim()):
            shape.append(t.cores[n].shape[-2])

            if t.cores[n].dim() == m:  # CP core
                if n < t.dim() - 1:
                    factor = torch.einsum(idx1, (factor, t.cores[n]))
                else:
                    factor = torch.einsum(idx2, (factor, t.cores[n]))[..., None]
            else:  # TT core
                factor = torch.einsum(idx3, (factor, t.cores[n]))

            if self.batch:
                factor = factor.reshape([batch_size, -1, factor.shape[-1]])
            else:
                factor = factor.reshape([-1, factor.shape[-1]])

        if factor.shape[-1] > 1:
            factor = factor.sum(dim=-1)
        else:
            factor = factor[..., 0]
        factor = factor.reshape(shape)
        return factor

    def to(
        self,
        device: Any):

        """
        Moves tensor to device.

        :return: a tntorch tensor
        """

        for i in range(self.dim()):
            self.cores[i] = self.cores[i].to(device)

        for i in range(len(self.Us)):
            if self.Us[i] is not None:
                self.Us[i] = self.Us[i].to(device)

        for i in range(len(self.idxs)):
            self.idxs[i] = self.idxs[i].to(device)

        return self

    def numpy(self):
        """
        Decompresses this tensor into a NumPy ndarray.

        :return: a NumPy tensor
        """

        return self.torch().detach().cpu().numpy()

    def _cp_to_tt(
        self,
        factor: Optional[Union[None, Any]] = None):
        """
        Turn a CP factor into a TT core (each slice is a diagonal matrix)
        :param factor: CP factor. If None, all cores in this tensor will be converted

        """
        if self.batch:
            m = 3
        else:
            m = 2

        if factor is None:
            if self.cores[0].dim() == m:
                if self.batch:
                    self.cores[0] = self.cores[0][:, None, ...]
                else:
                    self.cores[0] = self.cores[0][None, ...]
            for mu in range(1, self.dim() - 1):
                self.cores[mu] = self._cp_to_tt(self.cores[mu])

            if self.cores[-1].dim() == m:
                self.cores[-1] = self.cores[-1].transpose(-1, -2)[..., None]

            return
        if factor.dim() == m + 1:  # Already a TT core
            return factor

        if self.batch:
            shape1 = (factor.shape[0], factor.shape[2], factor.shape[2] + 1, factor.shape[1])
            shape2 = (factor.shape[0], factor.shape[2] + 1, factor.shape[2], factor.shape[1])
            order = (0, 1, 3, 2)
        else:
            shape1 = (factor.shape[1], factor.shape[1] + 1, factor.shape[0])
            shape2 = (factor.shape[1] + 1, factor.shape[1], factor.shape[0])
            order = (0, 2, 1)

        core = torch.zeros(shape1)
        core[..., 0, :] = factor.transpose(-1, -2)
        return core.reshape(shape2).permute(order)[..., :-1, :, :]

    """
    Rounding and orthogonalization
    """

    def factor_orthogonalize(
        self,
        mu: int):

        """
        Pushes the factor's non-orthogonal part to its corresponding core.

        This method works in place.

        :param mu: an int between 0 and N-1
        """

        if self.Us[mu] is None:
            return

        Q, R = torch.linalg.qr(self.Us[mu])
        self.Us[mu] = Q

        if self.batch:
            m = 3
            idx1 = 'bjk,baj->bak'
            idx2 = 'bijk,baj->biak'
        else:
            m = 2
            idx1 = 'jk,aj->ak'
            idx2 = 'ijk,aj->iak'

        if self.cores[mu].dim() == m:
            self.cores[mu] = torch.einsum(idx1, (self.cores[mu], R))
        else:
            self.cores[mu] = torch.einsum(idx2, (self.cores[mu], R))

    def left_orthogonalize(
        self,
        mu: int):

        """
        Makes the mu-th core left-orthogonal and pushes the R factor to its right core. This may change the ranks
        of the cores.

        This method works in place.

        Note: internally, this method will turn CP (or CP-Tucker) cores into TT (or TT-Tucker) ones.

        :param mu: an int between 0 and N-1

        :return: the R factor
        """

        assert 0 <= mu < self.dim()-1
        self.factor_orthogonalize(mu)
        Q, R = torch.linalg.qr(tn.left_unfolding(self.cores[mu], batch=self.batch))

        if self.batch:
            self.cores[mu] = Q.reshape(self.cores[mu].shape[:-1] + (Q.shape[2], ))
        else:
            self.cores[mu] = Q.reshape(self.cores[mu].shape[:-1] + (Q.shape[1], ))

        rightcoreR = tn.right_unfolding(self.cores[mu + 1], batch=self.batch)

        if self.batch:
            self.cores[mu + 1] = (R @ rightcoreR).reshape((R.shape[0], R.shape[1]) + self.cores[mu + 1].shape[2:])
        else:
            self.cores[mu + 1] = (R @ rightcoreR).reshape((R.shape[0], ) + self.cores[mu + 1].shape[1:])
        return R

    def right_orthogonalize(
        self,
        mu: int):

        """
        Makes the mu-th core right-orthogonal and pushes the L factor to its left core. Note: this may change the ranks
         of the tensor.

        This method works in place.

        Note: internally, this method will turn CP (or CP-Tucker) cores into TT (or TT-Tucker) ones.

        :param mu: an int between 0 and N-1

        :return: the L factor
        """

        assert 1 <= mu < self.dim()
        self.factor_orthogonalize(mu)
        # Torch has no rq() decomposition
        if self.batch:
            Q, L = torch.linalg.qr(tn.right_unfolding(self.cores[mu], batch=self.batch).permute(0, 2, 1))
            L = L.permute(0, 2, 1)
            Q = Q.permute(0, 2, 1)
        else:
            Q, L = torch.linalg.qr(tn.right_unfolding(self.cores[mu], batch=self.batch).permute(1, 0))
            L = L.permute(1, 0)
            Q = Q.permute(1, 0)

        if self.batch:
            self.cores[mu] = Q.reshape((Q.shape[:2]) + self.cores[mu].shape[2:])
        else:
            self.cores[mu] = Q.reshape((Q.shape[0], ) + self.cores[mu].shape[1:])

        leftcoreL = tn.left_unfolding(self.cores[mu-1], batch=self.batch)
        if self.batch:
            self.cores[mu - 1] = (leftcoreL @ L).reshape(self.cores[mu-1].shape[:-1] + (L.shape[2], ))
        else:
            self.cores[mu - 1] = (leftcoreL @ L).reshape(self.cores[mu-1].shape[:-1] + (L.shape[1], ))
        return L

    def orthogonalize(
        self,
        mu: int):

        """
        Apply all left and right orthogonalizations needed to make the tensor mu-orthogonal.

        This method works in place.

        Note: internally, this method will turn CP (or CP-Tucker) cores into TT (or TT-Tucker) ones.

        :param mu: an int between 0 and N-1

        :return: L, R: left and right factors
        """

        if mu < 0:
            mu += self.dim()

        self._cp_to_tt()
        if self.batch:
            batch_size = self.cores[0].shape[0]
            L = torch.ones(batch_size, 1, 1)
            R = torch.ones(batch_size, 1, 1)
        else:
            L = torch.ones(1, 1)
            R = torch.ones(1, 1)
        for i in range(mu):
            R = self.left_orthogonalize(i)
        for i in range(self.dim() - 1, mu, -1):
            L = self.right_orthogonalize(i)
        return R, L

    def round_tucker(
        self,
        eps: float = 1e-14,
        rmax: Optional[Union[int, Sequence[int]]] = None,
        dim: Optional[Union[Sequence[int], str]] = 'all',
        algorithm: Optional[str] = 'svd'):

        """
        Tries to recompress this tensor in place by reducing its Tucker ranks.

        Note: this method will turn CP (or CP-Tucker) cores into TT (or TT-Tucker) ones.

        :param eps: this relative error will not be exceeded
        :param rmax: all ranks should be rmax at most (default: no limit)
        :param dim: list of factors to set; default is 'all'
        :param algorithm: 'svd' (default) or 'eig'. The latter can be faster, but less accurate
        """

        N = self.dim()

        if not hasattr(rmax, '__len__'):
            rmax = [rmax] * N
        assert len(rmax) == N
        if dim == 'all':
            dim = range(N)
        if not hasattr(dim, '__len__'):
            dim = [dim] * N

        if self.batch:
            batch_size = self.cores[0].shape[0]

        for m in dim:
            self.cores[m] = self._cp_to_tt(self.cores[m])
        self.orthogonalize(-1)
        for mu in range(N - 1, -1, -1):
            if self.Us[mu] is None:
                device = self.cores[mu].device

                if self.batch:
                    self.Us[mu] = torch.eye(self.shape[mu + 1]).repeat(batch_size, 1, 1).to(device)
                else:
                    self.Us[mu] = torch.eye(self.shape[mu]).to(device)

            # Send non-orthogonality to factor
            if self.batch:
                Q, R = torch.linalg.qr(torch.reshape(self.cores[mu].permute(0, 1, 3, 2), [self.cores[mu].shape[0], -1, self.cores[mu].shape[2]]))
                self.cores[mu] = Q.reshape([self.cores[mu].shape[0], self.cores[mu].shape[1], self.cores[mu].shape[3], -1]).permute(0, 1, 3, 2)
            else:
                Q, R = torch.linalg.qr(torch.reshape(self.cores[mu].permute(0, 2, 1), [-1, self.cores[mu].shape[1]]))
                self.cores[mu] = Q.reshape([self.cores[mu].shape[0], self.cores[mu].shape[2], -1]).permute(0, 2, 1)

            self.Us[mu] = self.Us[mu] @ R.transpose(-1, -2)

            # Split factor according to error budget
            left, right = tn.truncated_svd(
                self.Us[mu],
                eps=eps / np.sqrt(len(dim)),
                rmax=rmax[mu],
                left_ortho=True,
                algorithm=algorithm,
                batch=self.batch)
            self.Us[mu] = left

            # Push the (non-orthogonal) remainder to the core
            if self.batch:
                self.cores[mu] = torch.einsum('bijk,baj->biak', (self.cores[mu], right))
            else:
                self.cores[mu] = torch.einsum('ijk,aj->iak', (self.cores[mu], right))

            # Prepare next iteration
            if mu > 0:
                self.right_orthogonalize(mu)

    def round_tt(
        self,
        eps: float = 1e-14,
        rmax: Optional[Union[int, Sequence[int]]] = None,
        algorithm: Optional[str] = 'svd',
        verbose: Optional[bool] = False):

        """
        Tries to recompress this tensor in place by reducing its TT ranks.

        Note: this method will turn CP (or CP-Tucker) cores into TT (or TT-Tucker) ones.

        :param eps: this relative error will not be exceeded
        :param rmax: all ranks should be rmax at most (default: no limit)
        :param algorithm: 'svd' (default) or 'eig'. The latter can be faster, but less accurate
        :param verbose:
        """

        N = self.dim()
        if not hasattr(rmax, '__len__'):
            rmax = [rmax] * (N - 1)
        assert len(rmax) == N - 1

        self._cp_to_tt()
        start = time.time()
        self.orthogonalize(N - 1)  # Make everything left-orthogonal
        if verbose:
            print('Orthogonalization time:', time.time() - start)
        if self.batch:
            delta = None
        else:
            delta = eps / max(1, torch.sqrt(torch.tensor([N - 1], dtype=torch.float64, device=self.cores[-1].device))) * torch.norm(self.cores[-1])
            delta = delta.item()

        for mu in range(N - 1, 0, -1):
            M = tn.right_unfolding(self.cores[mu], batch=self.batch)
            left, right = tn.truncated_svd(M, delta=delta, rmax=rmax[mu - 1], left_ortho=False, algorithm=algorithm, verbose=verbose, batch=self.batch)

            if self.batch:
                self.cores[mu] = right.reshape([self.cores[mu].shape[0], -1, self.cores[mu].shape[2], self.cores[mu].shape[3]])
                self.cores[mu - 1] = torch.einsum('bijk,bkl->bijl', (self.cores[mu - 1], left))  # Pass factor to the left
            else:
                self.cores[mu] = right.reshape([-1, self.cores[mu].shape[1], self.cores[mu].shape[2]])
                self.cores[mu - 1] = torch.einsum('ijk,kl', (self.cores[mu - 1], left))  # Pass factor to the left

    def round(
        self,
        eps: float = 1e-14,
        **kwargs):

        """
        General recompression. Attempts to reduce TT ranks first; then does Tucker rounding with the remaining error
        budget.

        :param eps: this relative error will not be exceeded
        :param kwargs: passed to `round_tt()` and `round_tucker()`
        """

        copy = self.clone()
        self.round_tt(eps, **kwargs)
        reached = tn.relative_error(copy, self)
        if reached < eps:
            self.round_tucker((1 + eps) / (1 + reached) - 1, **kwargs)

    """
    Convenience "methods"
    """

    def dot(self, other, **kwargs):
        """
        See :func:`metrics.dot()`.
        """

        return tn.dot(self, other, **kwargs)

    def mean(self, **kwargs):
        """
        See :func:`metrics.mean()`.
        """

        return tn.mean(self, **kwargs)

    def sum(self, **kwargs):
        """
        See :func:`metrics.sum()`.
        """

        return tn.sum(self, **kwargs)

    def var(self, **kwargs):
        """
        See :func:`metrics.var()`.
        """

        return tn.var(self, **kwargs)

    def std(self, **kwargs):
        """
        See :func:`metrics.std()`.
        """

        return tn.std(self, **kwargs)

    def norm(self, **kwargs):
        """
        See :func:`metrics.norm()`.
        """

        return tn.norm(self, **kwargs)

    def normsq(self, **kwargs):
        """
        See :func:`metrics.normsq()`.
        """

        return tn.normsq(self, **kwargs)

    """
    Miscellaneous
    """

    def set_factors(
        self,
        name: str,
        dim: Optional[Union[Sequence[int], str]] = 'all',
        requires_grad: Optional[bool] = False):
        """
        Sets factors Us of this tensor to be of a certain family.

        :param name: See :func:`tools.generate_basis()`
        :param dim: list of factors to set; default is 'all'
        :param requires_grad: whether the new factors should be optimizable. Default is False
        """

        if dim == 'all':
            dim = range(self.dim())

        for m in dim:
            if self.Us[m] is None:
                if self.batch:
                    self.Us[m] = tn.generate_basis(name, (self.shape[m], self.shape[m])).repeat(self.shape[0], 1, 1)
                else:
                    self.Us[m] = tn.generate_basis(name, (self.shape[m], self.shape[m]))
            else:
                if self.batch:
                    tn.generate_basis(name, self.Us[m].shape).repeat(self.shape[0], 1, 1)
                else:
                    self.Us[m] = tn.generate_basis(name, self.Us[m].shape)
            self.Us[m].requires_grad = requires_grad

    def as_leaf(self):
        """
        Makes this tensor a leaf (optimizable) tensor, thus forgetting the operations from which it arose.

        :Example:

        >>> t = tn.rand([10]*3, requires_grad=True)  # Is a leaf
        >>> t *= 2  # Is not a leaf
        >>> t.as_leaf()  # Is a leaf again
        """

        for n in range(self.dim()):
            if self.Us[n] is not None:
                if self.Us[n].requires_grad:
                    self.Us[n] = self.Us[n].detach().clone().requires_grad_()
                else:
                    self.Us[n] = self.Us[n].detach().clone()
            if self.cores[n].requires_grad:
                self.cores[n] = self.cores[n].detach().clone().requires_grad_()
            else:
                self.cores[n] = self.cores[n].detach().clone()

    def clone(self):
        """
        Creates a copy of this tensor (calls PyTorch's `clone()` on all internal tensor network nodes)

        :return: another compressed tensor
        """

        cores = [self.cores[n].clone()for n in range(self.dim())]
        Us = []
        for n in range(self.dim()):
            if self.Us[n] is None:
                Us.append(None)
            else:
                Us.append(self.Us[n].clone())
        if hasattr(self, 'idxs'):
            return tn.Tensor(cores, Us=Us, idxs=self.idxs, batch=self.batch)
        return tn.Tensor(cores, Us=Us, batch=self.batch)

    def numel(self):
        """
        Counts the total number of uncompressed elements of this tensor.

        :return: a float64 (often, a tensor's size will not fit in integer type)
        """

        return torch.round(torch.prod(torch.tensor(self.shape).double()))

    def numcoef(self):
        """
        Counts the total number of compressed coefficients of this tensor.

        :return: an integer
        """

        result = 0
        for n in range(self.dim()):
            result += self.cores[n].numel()
            if self.Us[n] is not None:
                result += self.Us[n].numel()
        return result

    def repeat(
        self,
        *rep: Sequence[int]):

        """
        Returns another tensor repeated along one or more axes; works like PyTorch's `repeat()`.

        :param rep: a list, possibly longer than the tensor's number of dimensions

        :return: another tensor
        """

        assert len(rep) >= self.dim()
        assert all([r >= 1 for r in rep])

        t = self.clone()
        if len(rep) > self.dim():  # If requested, we add trailing new dimensions. We use CP as is cheaper
            for n in range(self.dim(), len(rep)):
                t.cores.append(torch.ones(rep[n], self.cores[-1].shape[-1]))
                t.Us.append(None)
        for n in range(self.dim()):
            if t.Us[n] is not None:
                t.Us[n] = t.Us[n].repeat(rep[n], 1)
            else:
                if self.batch:
                    if t.cores[n].dim() == 4:
                        t.cores[n] = t.cores[n].repeat(1, 1, rep[n], 1)
                    else:
                        t.cores[n] = t.cores[n].repeat(1, rep[n], 1)
                else:
                    if t.cores[n].dim() == 3:
                        t.cores[n] = t.cores[n].repeat(1, rep[n], 1)
                    else:
                        t.cores[n] = t.cores[n].repeat(rep[n], 1)
        return t


def _broadcast(
    a: Any,
    b: Any):

    if a.shape == b.shape:
        return a, b
    elif a.dim() != b.dim():
        raise ValueError('Cannot broadcast: lhs has {} dimensions, rhs has {}'.format(a.dim(), b.dim()))
    result1 = a.repeat(*[int(round(max(sh2 / sh1, 1))) for sh1, sh2 in zip(a.shape, b.shape)])
    result2 = b.repeat(*[int(round(max(sh1 / sh2, 1))) for sh1, sh2 in zip(a.shape, b.shape)])
    return result1, result2


def _core_kron(
    a: Any,
    b: Any,
    batch: Optional[bool] = False):

    if batch:
        assert a.shape[0] == b.shape[0]
        c = a[:, :, None, :, :, None] * b[:, None, :, :, None, :]
        c = c.reshape([a.shape[0], a.shape[1] * b.shape[1], -1, a.shape[-1] * b.shape[-1]])
    else:
        c = a[:, None, :, :, None] * b[None, :, :, None, :]
        c = c.reshape([a.shape[0] * b.shape[0], -1, a.shape[-1] * b.shape[-1]])
    return c

"""
    Some of the code in this file was shamelessly taken or adapted from https://github.com/Bihaqo/t3f
"""

from typing import Any, List, Optional, Sequence, Union

import torch

import tntorch as tn


class TTMatrix:
    """
    TTMatrix efficiently stores matrix in TT-like format.

    Input matrix M is of shape IxO is reshaped into d-way tensor t of shape
    i_0 x o_0, ... i_{d - 1} x o_{d - 1}.
    TT representation of t is computed and each of d TT cores is reshaped
    from r_j, t.shape[j], r_{j + 1} to r_j, i_j, o_j, r_{j + 1}.
    """

    def __init__(
        self,
        t: Union[torch.Tensor, List[torch.Tensor]],
        ranks: List[int],
        input_dims: List[int],
        output_dims: List[int],
    ):
        """
        t: torch.Tensor or List[torch.Tensor] - input can be matrix M or the pre-processed cores
        ranks: List[int] - maximal ranks for each mode
        input_dims: List[int] - i_0, ..., i_{d - 1}
        output_dims: List[int] - j_0, ..., j_{d - 1}
        """

        assert len(input_dims) == len(output_dims)
        assert len(input_dims) > 0
        assert isinstance(ranks, list) and len(ranks) == len(input_dims) - 1

        self.input_dims = torch.tensor(input_dims)
        self.output_dims = torch.tensor(output_dims)
        self.d = len(input_dims)

        if isinstance(t, list):
            core_dims = len(t[0].shape)
            assert core_dims in [4, 5]

            self.batch = (
                core_dims == 5
            )  # NOTE: b x r_{i - 1} x input_i x output_i x r_i
            self.cores = t
            self.ranks = torch.tensor([c.shape[-1] for c in t[:-1]])
            return

        M = t
        assert len(M.shape) in [2, 3]

        if len(M.shape) == 2:
            self.batch = False
        else:
            self.batch = True

        assert torch.prod(self.input_dims) == M.shape[-2]
        assert torch.prod(self.output_dims) == M.shape[-1]

        if self.batch:
            tensor = M.reshape([-1] + list(input_dims) + list(output_dims))
            dims = list(range(1, 2 * self.d + 1))
            # Note: tensor is now a reshape of matrix with dimesions stored as
            # b x i_0 x j_0, ..., i_{d - 1} x j_{d - 1}

            new_dims: List[int] = (
                torch.tensor([0] + list(zip(dims[: self.d], dims[self.d :])))
                .flatten()
                .tolist()
            )
        else:
            tensor = M.reshape(list(input_dims) + list(output_dims))
            dims = list(range(2 * self.d))
            # Note: tensor is now a reshape of matrix with dimesions stored as
            # i_0 x j_0, ..., i_{d - 1} x j_{d - 1}

            new_dims: List[int] = (
                torch.tensor(list(zip(dims[: self.d], dims[self.d :])))
                .flatten()
                .tolist()
            )
        tensor = tensor.permute(new_dims)
        if self.batch:
            tensor = tensor.reshape(
                [-1] + [input_dims[i] * output_dims[i] for i in range(self.d)]
            )
        else:
            tensor = tensor.reshape(
                [input_dims[i] * output_dims[i] for i in range(self.d)]
            )
        tt = tn.Tensor(tensor, ranks_tt=ranks, batch=self.batch)
        self.ranks = tt.ranks_tt[1:-1]

        self.cores = [
            (
                core.reshape(
                    -1, core.shape[1], input_dims[i], output_dims[i], core.shape[-1]
                )
                if self.batch
                else core.reshape(
                    core.shape[0], input_dims[i], output_dims[i], core.shape[-1]
                )
            )
            for i, core in enumerate(tt.cores)
        ]

    def torch(self):
        """
        Decompress into a PyTorch 2D tensor

        :return: a 2D torch.tensor
        """

        cores = [
            (
                c.reshape(
                    -1,
                    c.shape[1],
                    self.input_dims[i] * self.output_dims[i],
                    c.shape[-1],
                )
                if self.batch
                else c.reshape(c.shape[0], -1, c.shape[-1])
            )
            for i, c in enumerate(self.cores)
        ]
        tensor = tn.Tensor(cores, batch=self.batch).torch()
        rows = torch.prod(self.input_dims)
        cols = torch.prod(self.output_dims)

        shape: List[int] = (
            torch.tensor(list(zip(self.input_dims, self.output_dims)))
            .flatten()
            .tolist()
        )
        if self.batch:
            tensor = tensor.reshape([-1] + shape)
            dims = list(range(1, 2 * self.d + 1))
            tensor = tensor.permute([0] + dims[1::2] + dims[2::2])
            return tensor.reshape(-1, rows, cols)
        else:
            tensor = tensor.reshape(shape)
            dims = list(range(2 * self.d))
            tensor = tensor.permute(dims[0::2] + dims[1::2])
            return tensor.reshape(rows, cols)

    def to(self, device):
        self.cores = [core.to(device) for core in self.cores]
        return self

    def numpy(self):
        return self.torch().detach().cpu().numpy()

    def trace(self):
        """
        Compute the trace of a TTMatrix.

        :return: a scalar
        """
        if self.batch:
            b = self.cores[0].shape[0]
            factor = torch.ones((b, 1))
            eq = "bi,biaaj->bj"
        else:
            factor = torch.ones(1)
            eq = "i,iaaj->j"

        for c in self.cores:
            factor = torch.einsum(eq, factor, c)
        return factor[..., 0]

    def flatten(self):
        """
        Flattens this TTMatrix into a compressed vector.
        For each core, its input and output dimension will be grouped together into a single spatial dimension.

        :return: a `tn.Tensor`
        """

        return tn.Tensor(
            [
                (
                    c.reshape(
                        -1,
                        c.shape[1],
                        self.input_dims[i] * self.output_dims[i],
                        c.shape[-1],
                    )
                    if self.batch
                    else c.reshape(c.shape[0], -1, c.shape[-1])
                )
                for i, c in enumerate(self.cores)
            ],
            batch=self.batch,
        )

    def _is_kron(self):
        """
        Returns True if self is a Kronecker product matrix.

        :return: bool
        """
        return max(self.ranks) == 1

    def _check_kron_properties(self):
        """
        :raise: ValueError if the tt-cores of the provided matrix are not square,
            or the tt-ranks are not 1.
        """
        if not self._is_kron():
            raise ValueError(
                "The argument should be a Kronecker product (tt-ranks " "should be 1)"
            )

        if torch.equal(self.input_dims, self.output_dims):
            raise ValueError(
                "The argument should be a Kronecker product of square "
                "matrices (tt-cores must be square)"
            )

    def determinant(self):
        """
        Computes the determinant of a given Kronecker-factorized matrix.

        Note, that this method can suffer from overflow.

        self: `TTMatrix` object containing a matrix or a
        batch of matrices of size N x N, factorized into a Kronecker product of
        square matrices (all tt-ranks are 1 and all tt-cores are square).

        :return:
            A number or a Tensor with numbers for each element in the batch.
            The determinant of the given matrix.
        """
        self._check_kron_properties()

        rows = torch.prod(self.input_dims)

        det = 1.0
        for core_idx in range(self.d):
            if self.batch:
                core_det = torch.linalg.det(self.cores[core_idx][:, 0, :, :, 0])
            else:
                core_det = torch.linalg.det(self.cores[core_idx][0, :, :, 0])
            core_pow = rows / self.input_dims[core_idx]

            det *= torch.pow(core_det, core_pow)
        return det

    def slog_determinant(self):
        """
        Computes the sign and log-det of a given Kronecker-factorized matrix.

        self: `TTMatrix` object containing a matrix or a
        batch of matrices of size N x N, factorized into a Kronecker product of
        square matrices (all tt-ranks are 1 and all tt-cores are square).

        :return:  Two number or two Tensor with numbers for each element in the batch.
            Sign of the determinant and the log-determinant of the given
            matrix. If the determinant is zero, then sign will be 0 and logdet will be
            -Inf. In all cases, the determinant is equal to sign * np.exp(logdet).
        """
        self._check_kron_properties()

        rows = torch.prod(self.input_dims)
        logdet = 0.0
        det_sign = 1.0

        for core_idx in range(self.d):
            if self.batch:
                core_det = torch.linalg.det(self.cores[core_idx][:, 0, :, :, 0])
            else:
                core_det = torch.linalg.det(self.cores[core_idx][0, :, :, 0])

            core_abs_det = torch.abs(core_det)
            core_det_sign = torch.sign(core_det)
            core_pow = rows / self.input_dims[core_idx]
            logdet += torch.log(core_abs_det) * core_pow
            det_sign *= core_det_sign ** (core_pow)
        return det_sign, logdet

    def inv(self):
        """
        Computes the inverse of a given Kronecker-factorized matrix.

        self: `TTMatrix` object containing a matrix or a
        batch of matrices of size N x N, factorized into a Kronecker product of
        square matrices (all tt-ranks are 1 and all tt-cores are square).

        :return: `TTMatrix` of size N x N
        """
        self._check_kron_properties()

        inv_cores = []
        for core_idx in range(self.d):
            if self.batch:
                core_inv = torch.linalg.inv(self.cores[core_idx][:, 0, :, :, 0])
                core_inv = torch.unsqueeze(core_inv, 1)
            else:
                core_inv = torch.linalg.inv(self.cores[core_idx][0, :, :, 0])
                core_inv = torch.unsqueeze(core_inv, 0)
            inv_cores.append(torch.unsqueeze(core_inv, -1))

        # NOTE: ranks will be computed based on cores shape
        return TTMatrix(inv_cores, None, self.input_dims, self.output_dims)

    def cholesky(self):
        """
        Computes the Cholesky decomposition of a given Kronecker-factorized matrix.

        self: `TTMatrix` containing a matrix or a
            batch of matrices of size N x N, factorized into a Kronecker product of
            square matrices (all tt-ranks are 1 and all tt-cores are square). All the
            cores must be symmetric positive-definite.

        :return: `TTMatrix` of size N x N
        """
        self._check_kron_properties()

        cho_cores = []
        for core_idx in range(self.d):
            if self.batch:
                core_cho = torch.linalg.cholesky(self.cores[core_idx][:, 0, :, :, 0])
                core_cho = torch.unsqueeze(core_cho, 1)
            else:
                core_cho = torch.linalg.cholesky(self.cores[core_idx][0, :, :, 0])
                core_cho = torch.unsqueeze(core_cho, 0)
            core_cho.append(torch.unsqueeze(core_cho, -1))

        # NOTE: ranks will be computed based on cores shape
        return TTMatrix(cho_cores, None, self.input_dims, self.output_dims)


class CPMatrix:
    """
    CPMatrix efficiently stores matrix in CP-like format.

    Input matrix M is of shape IxO is reshaped into d-way tensor t of shape
    i_0 x o_0, ... i_{d - 1} x o_{d - 1}.
    CP representation of t is computed and each of d CP cores is reshaped
    from t.shape[j], r to i_j, o_j, r.
    """

    def __init__(
        self,
        M: torch.Tensor,
        rank: Sequence[int],
        input_dims: Sequence[int],
        output_dims: Sequence[int],
        batch_size: int = 1,
        verbose: bool = False,
    ):
        assert len(input_dims) == len(output_dims)
        assert len(input_dims) > 0
        assert isinstance(rank, int)
        assert len(M.shape) == 2

        self.rank = rank
        self.input_dims = torch.tensor(input_dims)
        self.output_dims = torch.tensor(output_dims)
        self.batch_size = batch_size

        assert torch.prod(self.input_dims) == M.shape[0]
        assert torch.prod(self.output_dims) == M.shape[1]

        self.d = len(input_dims)
        tensor = M.reshape(list(input_dims) + list(output_dims))
        dims = list(range(2 * self.d))
        # Note: tensor is now a reshape of matrix with dimesions stored as
        # i_0 x j_0, ..., i_{d - 1} x j_{d - 1}
        new_dims: List[int] = (
            torch.tensor(list(zip(dims[: self.d], dims[self.d :]))).flatten().tolist()
        )
        tensor = tensor.permute(new_dims)
        tensor = tensor.reshape([input_dims[i] * output_dims[i] for i in range(self.d)])
        cp = tn.Tensor(tensor, ranks_cp=rank)

        self.cores = [
            core.reshape(input_dims[i], output_dims[i], core.shape[-1])
            for i, core in enumerate(cp.cores)
        ]

    def torch(self):
        """
        Decompress into a PyTorch 2D tensor

        :return: a 2D torch.tensor
        """

        cores = [core.reshape(-1, core.shape[-1]) for core in self.cores]
        tensor = tn.Tensor(cores).torch()

        input_size = torch.prod(self.input_dims)
        output_size = torch.prod(self.output_dims)

        shape: List[int] = (
            torch.tensor(list(zip(self.input_dims, self.output_dims)))
            .flatten()
            .tolist()
        )
        tensor = tensor.reshape(shape)
        dims = list(range(2 * self.d))
        tensor = tensor.permute(dims[0::2] + dims[1::2])
        return tensor.reshape(input_size, output_size)

    def to(self, device):
        self.cores = [core.to(device) for core in self.cores]
        return self

    def numpy(self):
        return self.torch().detach().cpu().numpy()


def tt_multiply(tt_matrix: TTMatrix, tensor: torch.Tensor):
    """
    Multiply TTMatrix by any tensor of more than 1-way.

    For vectors, reshape them to matrix of shape 1 x I

    returns: torch.Tensor of shape b x num_cols(tt_matrix)
    """

    assert len(tensor.shape) > 1

    rows = torch.prod(tt_matrix.input_dims)
    b = tensor.reshape(-1, rows).shape[0]
    tensor = tensor.reshape(b, -1).T
    result = tensor.reshape(tt_matrix.input_dims[0], -1)
    result = torch.einsum("id,lior->ldor", result, tt_matrix.cores[0])

    for d in range(1, tt_matrix.d):
        result = result.reshape(
            tt_matrix.input_dims[d], -1, tt_matrix.cores[d].shape[0]
        )
        result = torch.einsum("idr,riob->dob", result, tt_matrix.cores[d])

    return result.reshape(b, -1)


def cp_multiply(cp_matrix: CPMatrix, tensor: torch.Tensor):
    """
    Multiply CPMatrix by any tensor of more than 1-way.
    For vectors, reshape them to matrix of shape 1 x I
    """

    assert len(tensor.shape) > 1

    rows = torch.prod(cp_matrix.input_dims)
    b = tensor.reshape(-1, rows).shape[0]
    tensor = tensor.reshape(b, -1).T

    result = tensor.reshape(cp_matrix.input_dims[0], -1)
    result = torch.einsum("ij,ior->jor", result, cp_matrix.cores[0])

    for d in range(1, cp_matrix.d):
        result = result.reshape(
            cp_matrix.input_dims[d], -1, cp_matrix.cores[d].shape[-1]
        )
        result = torch.einsum("ior,idr->dor", cp_matrix.cores[d], result)

    result = result.sum(-1)
    return result.reshape(b, -1)

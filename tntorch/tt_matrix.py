import torch
import tntorch as tn
import itertools
from typing import Any, Sequence


class TTMatrix:
    """
    TTMatrix efficiently stores matrix in TT-like format.

    Input matrix M is of shape IxO is reshaped into d-way tensor t of shape
    i_0 x o_o, ... i_{d - 1} x o_{d - 1}.
    TT representation of t is computed and each of d TT cores is reshaped
    from r_j, t.shape[j], r_{j + 1} to r_j, i_j, o_j, r_{j + 1}.
    """

    def __init__(
            self,
            M: torch.Tensor,
            ranks: Sequence[int],
            input_dims: Sequence[int],
            output_dims: Sequence[int]):
        assert len(input_dims) == len(output_dims)
        assert len(input_dims) > 0
        assert len(ranks) == len(input_dims) - 1
        assert len(M.shape) == 2

        self.ranks = ranks
        self.input_dims = torch.tensor(input_dims)
        self.output_dims = torch.tensor(output_dims)

        assert torch.prod(self.input_dims) == M.shape[0]
        assert torch.prod(self.output_dims) == M.shape[1]

        self.d = len(input_dims)
        tensor = M.reshape(list(input_dims) + list(output_dims))
        dims = list(range(2 * self.d))
        input_dims_idxs = dims[:self.d]
        output_dims_idxs = dims[self.d:]
        # Note: tensor is now a reshape of matrix with dimesions stored as
        # i_0 x j_0, ..., i_{d - 1} x j_{d - 1}
        tensor = tensor.permute(list(
            itertools.chain(
                *zip(input_dims_idxs, output_dims_idxs))))
        tensor = tensor.reshape([input_dims[i] * output_dims[i] for i in range(self.d)])
        tt = tn.Tensor(tensor, ranks_tt=ranks)

        self.cores = [
            core.reshape(core.shape[0], input_dims[i], output_dims[i], core.shape[-1])
            for i, core in enumerate(tt.cores)]

    def torch(self):
        """
        Decompress into a PyTorch 2D tensor

        :return: a 2D torch.tensor
        """

        cores = [core.reshape(core.shape[0], -1, core.shape[-1]) for core in self.cores]
        tensor = tn.Tensor(cores).torch()
        input_dims = self.input_dims
        output_dims = self.output_dims
        d = self.d
        input_size = torch.prod(input_dims)
        output_size = torch.prod(output_dims)

        tensor = tensor.reshape(list(
            itertools.chain(
                *zip(input_dims, output_dims))))
        dims = list(range(2 * d))
        tensor = tensor.permute(dims[0::2] + dims[1::2])
        return tensor.reshape(input_size, output_size)

    def numpy(self):
        return self.torch().numpy()

    def trace(self):
        """
        Compute the trace of a TTMatrix.

        :return: a scalar
        """

        factor = torch.ones(1)
        for c in self.cores:
            factor = torch.einsum('i,iaaj->j', factor, c)
        return factor[0]

    def flatten(self):
        """
        Flattens this TTMatrix into a compressed vector. For each core, its input and output dimension will be grouped together into a single spatial dimension.

        :return: a `tn.Tensor`
        """

        return tn.Tensor([c.reshape(c.shape[0], -1, c.shape[-1]) for c in self.cores])


def tt_multiply(tt_matrix: TTMatrix, tensor: torch.Tensor):
    """
    Multiply TTMatrix by any tensor of more than 1-way.

    For vectors, reshape them to matrix of shape 1 x I
    """

    assert len(tensor.shape) > 1

    result = tensor.reshape([-1] + [tt_matrix.input_dims[-1]] + [1])
    shape = torch.tensor(tensor.shape)
    b = torch.prod(shape[:-1])

    for d in reversed(range(tt_matrix.d)):
        core = tt_matrix.cores[d]
        result = torch.einsum('aijb,rib->jra', core, result)
        if d > 0:
            result = result.reshape(-1, tt_matrix.input_dims[d - 1], core.shape[0])

    output_dim = torch.prod(tt_matrix.output_dims)
    result = result.reshape(output_dim, b)
    return result.T.reshape(list(shape[:-1]) + [output_dim])

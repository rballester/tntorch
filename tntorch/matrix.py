import torch
import tntorch as tn
from typing import Any, Optional, List, Sequence


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
            M: torch.Tensor,
            ranks: List[int],
            input_dims: List[int],
            output_dims: List[int]):
        assert len(input_dims) == len(output_dims)
        assert len(input_dims) > 0
        assert isinstance(ranks, list) and len(ranks) == len(input_dims) - 1
        assert len(M.shape) == 2

        self.ranks = ranks
        self.input_dims = torch.tensor(input_dims)
        self.output_dims = torch.tensor(output_dims)

        assert torch.prod(self.input_dims) == M.shape[0]
        assert torch.prod(self.output_dims) == M.shape[1]

        self.d = len(input_dims)
        tensor = M.reshape(list(input_dims) + list(output_dims))
        dims = list(range(2 * self.d))
        # Note: tensor is now a reshape of matrix with dimesions stored as
        # i_0 x j_0, ..., i_{d - 1} x j_{d - 1}

        new_dims: List[int] = torch.tensor(list(zip(dims[:self.d], dims[self.d:]))).flatten().tolist()
        tensor = tensor.permute(new_dims)
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
        input_size = torch.prod(self.input_dims)
        output_size = torch.prod(self.output_dims)

        shape: List[int] = torch.tensor(list(zip(self.input_dims, self.output_dims))).flatten().tolist()
        tensor = tensor.reshape(shape)
        dims = list(range(2 * self.d))
        tensor = tensor.permute(dims[0::2] + dims[1::2])
        return tensor.reshape(input_size, output_size)

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
            verbose: bool = False):
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
        new_dims: List[int] = torch.tensor(list(zip(dims[:self.d], dims[self.d:]))).flatten().tolist()
        tensor = tensor.permute(new_dims)
        tensor = tensor.reshape([input_dims[i] * output_dims[i] for i in range(self.d)])
        cp = tn.Tensor(tensor, ranks_cp=rank)

        self.cores = [
            core.reshape(input_dims[i], output_dims[i], core.shape[-1])
            for i, core in enumerate(cp.cores)]

    def torch(self):
        """
        Decompress into a PyTorch 2D tensor

        :return: a 2D torch.tensor
        """

        cores = [core.reshape(-1, core.shape[-1]) for core in self.cores]
        tensor = tn.Tensor(cores).torch()

        input_size = torch.prod(self.input_dims)
        output_size = torch.prod(self.output_dims)

        shape: List[int] = torch.tensor(list(zip(self.input_dims, self.output_dims))).flatten().tolist()
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
    result = torch.einsum('id,lior->ldor', result, tt_matrix.cores[0])

    for d in range(1, tt_matrix.d):
        result = result.reshape(tt_matrix.input_dims[d], -1, tt_matrix.cores[d].shape[0])
        result = torch.einsum('idr,riob->dob', result, tt_matrix.cores[d])

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
    result = torch.einsum('ij,ior->jor', result, cp_matrix.cores[0])

    for d in range(1, cp_matrix.d):
        result = result.reshape(cp_matrix.input_dims[d], -1, cp_matrix.cores[d].shape[-1])
        result = torch.einsum('ior,idr->dor', cp_matrix.cores[d], result)

    result = result.sum(-1)
    return result.reshape(b, -1)

import itertools
import torch
import opt_einsum as oe
import tntorch as tn
from typing import Any, Sequence, Optional


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
            ranks: Sequence[int],
            input_dims: Sequence[int],
            output_dims: Sequence[int],
            batch_size: int = 1,
            verbose: bool = False):
        assert len(input_dims) == len(output_dims)
        assert len(input_dims) > 0
        assert len(ranks) == len(input_dims) - 1
        assert len(M.shape) == 2

        self.ranks = ranks
        self.input_dims = torch.tensor(input_dims)
        self.output_dims = torch.tensor(output_dims)
        self.batch_size = batch_size

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
        self.contract_fn, report = compile_tt_mul_contraction_fn(
            input_dims,
            [tuple(core.shape) for core in self.cores],
            batch_size=batch_size,
            report_flops=True)
        if verbose:
            print(report)

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



def gen_letter():
    next_letter_id = 0
    while True:
        yield oe.get_symbol(next_letter_id)
        next_letter_id += 1


def perf_report(equation: str, *shapes: Sequence[Sequence[int]], einsum_opt_method: str = 'dp'):
    _, pathinfo = oe.contract_path(equation, *shapes, shapes=True, optimize=einsum_opt_method)
    out = {
        'flops': int(pathinfo.opt_cost),
        'size_max_intermediate': int(pathinfo.largest_intermediate),
        'size_all_intermediate': int(sum(pathinfo.size_list)),
        'equation': equation,
        'input_shapes': shapes,
    }
    return out


def compile_tt_mul_contraction_fn(
        input_shape: Sequence[int],
        cores_shape: Sequence[int],
        batch_size: Optional[int] = None,
        einsum_opt_method: str = 'dp',
        report_flops: bool = False):
    letter_gen = gen_letter()
    input_word = []
    equation_left = []
    equation_right = []

    if batch_size:
        letter_batch = next(letter_gen)
        input_word.append(letter_batch)
        equation_right.append(letter_batch)


    for i in range(len(cores_shape)):
        assert input_shape[i] == cores_shape[i][1]

        letter_i = next(letter_gen)
        letter_o = next(letter_gen)
        input_word.append(letter_i)

        if i == 0:
            letter_rank_prev = next(letter_gen)
            equation_right.append(letter_rank_prev)
        else:
            letter_rank_prev = letter_rank_next

        letter_rank_next = next(letter_gen)

        equation_left.append(letter_rank_prev + letter_i + letter_o + letter_rank_next)
        equation_right.append(letter_o)

    equation_right.append(letter_rank_next)
    equation_left = [''.join(input_word)] + equation_left

    all_shapes = [input_shape] + cores_shape
    if batch_size:
        all_shapes[0] = [batch_size] + all_shapes[0]

    equation = ','.join(equation_left) + '->' + ''.join(equation_right)
    contraction_fn = oe.contract_expression(
        equation,
        *all_shapes,
        optimize=einsum_opt_method)

    if report_flops:
        report = perf_report(equation, *all_shapes, einsum_opt_method=einsum_opt_method)
        return contraction_fn, report

    return contraction_fn


def tt_multiply(tt_matrix: TTMatrix, tensor: torch.Tensor):
    """
    Multiply TTMatrix by any tensor of more than 1-way.

    For vectors, reshape them to matrix of shape 1 x I

    returns: torch.Tensor of shape b x num_cols(tt_matrix)
    """

    assert len(tensor.shape) > 1

    input_dims = [core.shape[1] for core in tt_matrix.cores]
    shape = torch.tensor(tensor.shape)
    tensor = tensor.reshape(-1, torch.prod(torch.tensor(input_dims)))
    b = tensor.shape[0]

    assert b == tt_matrix.batch_size

    return tt_matrix.contract_fn(tensor.reshape(b, *input_dims), *tt_matrix.cores).reshape(b, -1)

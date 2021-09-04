import tntorch as tn
import pytest
import torch
torch.set_default_dtype(torch.float64)


torch.manual_seed(1)

def test_tt_multiply():
    m = torch.rand(11 * 3, 23 * 2)
    v = torch.rand(30, 11 * 3)  # Note: batch = 30, 10 features

    input_dims = [11, 3]
    output_dims = [23, 2]
    ranks = [50]

    ttm = tn.TTMatrix(m, input_dims=input_dims, output_dims=output_dims, ranks=ranks)
    assert torch.allclose(v @ m, tn.tt_multiply(ttm, v))


def test_construction():
    m = torch.rand(11 * 3, 23 * 2)

    input_dims = [11, 3]
    output_dims = [23, 2]
    ranks = [50]

    ttm = tn.TTMatrix(m, input_dims=input_dims, output_dims=output_dims, ranks=ranks)
    assert torch.allclose(m, ttm.full())

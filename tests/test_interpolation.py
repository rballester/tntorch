import numpy as np
import tntorch as tn
import torch
torch.set_default_dtype(torch.float64)


def test_als_completion():
    I = 8
    train_x = torch.arange(I)[:, None].repeat(1, 2)
    train_y = torch.ones(I)
    t = tn.als_completion(train_x, train_y, ranks_tt=3)
    assert tn.relative_error(train_y, t[train_x]) < 1e-5

import numpy as np
import tntorch as tn
import torch
torch.set_default_dtype(torch.float64)


def test_unfolding():
    t = torch.rand((30, 10, 20, 10))
    assert torch.allclose(tn.unfolding(t, 2, batch=False), t.permute(2, 0, 1, 3).reshape(20, -1))
    assert torch.allclose(tn.unfolding(t, 2, batch=True), t.permute(0, 3, 1, 2).reshape(30, 10, -1))

def test_cat():

    for i in range(100):
        N = np.random.randint(1, 4)
        shape1 = np.random.randint(1, 10, N)
        mode = np.random.randint(N)
        shape2 = shape1.copy()
        shape2[mode] = np.random.randint(1, 10)
        t1 = tn.rand(shape1, ranks_tt=2, ranks_tucker=2)
        t2 = tn.rand(shape2, ranks_tt=2)
        gt = np.concatenate([t1.numpy(), t2.numpy()], mode)
        assert np.linalg.norm(gt - tn.cat([t1, t2], dim=mode).numpy()) <= 1e-7


def test_cumsum():

    for i in range(100):
        N = np.random.randint(1, 4)
        howmany = 1
        modes = np.random.choice(N, howmany, replace=False)
        shape = np.random.randint(1, 10, N)
        t = tn.rand(shape, ranks_tt=2, ranks_tucker=2)
        assert np.linalg.norm(tn.cumsum(t, modes).numpy() - np.cumsum(t.numpy(), *modes)) <= 1e-7

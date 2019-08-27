import numpy as np
import tntorch as tn
import torch
torch.set_default_dtype(torch.float64)


def test_orthogonalization():

    for i in range(100):
        gt = tn.rand(np.random.randint(1, 8, np.random.randint(2, 6)))
        t = gt.clone()
        assert tn.relative_error(gt, t) <= 1e-7
        t.left_orthogonalize(0)
        assert tn.relative_error(gt, t) <= 1e-7
        t.right_orthogonalize(t.dim()-1)
        assert tn.relative_error(gt, t) <= 1e-7
        t.orthogonalize(np.random.randint(t.dim()))
        assert tn.relative_error(gt, t) <= 1e-7


def test_truncated_svd():
    gt = torch.rand((2, 32, 32))
    u, v = tn.truncated_svd(gt, batch=True)

    for i in range(len(gt)):
        u1, v1 = tn.truncated_svd(gt[i], batch=False)
        assert torch.allclose(u1, u[i])
        assert torch.allclose(v1, v[i])


def test_truncated_svd_eig():
    gt = torch.rand((2, 32, 32))
    u, v = tn.truncated_svd(gt, batch=True, algorithm='eig')

    for i in range(len(gt)):
        u1, v1 = tn.truncated_svd(gt[i], batch=False, algorithm='eig')
        assert torch.allclose(u1, u[i])
        assert torch.allclose(v1, v[i])


def test_round_tt_svd():

    for i in range(100):
        gt = tn.rand(np.random.randint(1, 8, np.random.randint(8, 10)), ranks_tt=np.random.randint(1, 10))
        gt.round_tt(1e-8, algorithm='svd')
        t = gt+gt
        t.round_tt(1e-8, algorithm='svd')
        assert tn.relative_error(gt, t/2) <= 1e-4
        assert max(gt.ranks_tt) == max(t.ranks_tt)


def test_round_tt_eig():

    for i in range(100):
        gt = tn.rand(np.random.randint(1, 8, np.random.randint(8, 10)), ranks_tt=np.random.randint(1, 10))
        gt.round_tt(1e-8, algorithm='eig')
        t = gt+gt
        t.round_tt(1e-8, algorithm='eig')
        assert tn.relative_error(gt, t/2) <= 1e-7


def test_round_tucker():
        for i in range(100):
            eps = np.random.rand()**2
            gt = tn.rand([32]*4, ranks_tt=8, ranks_tucker=8)
            t = gt.clone()
            t.round_tucker(eps=eps)
            assert tn.relative_error(gt, t) <= eps

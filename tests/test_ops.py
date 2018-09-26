import numpy as np
import tntorch as tn


def test_ops():

    for i in range(100):
        t1 = tn.rand(np.random.randint(1, 8, np.random.randint(1, 6)), ranks_tt=3, ranks_tucker=2)
        t2 = tn.rand(t1.shape)
        x1 = t1.full().numpy()
        x2 = t2.full().numpy()
        assert tn.relative_error(t1+t2, tn.Tensor(x1+x2)).item() <= 1e-7
        assert tn.relative_error(t1-t2, tn.Tensor(x1-x2)).item() <= 1e-7
        assert tn.relative_error(t1*t2, tn.Tensor(x1*x2)).item() <= 1e-7
        assert tn.relative_error(-t1+t2, tn.Tensor(-x1+x2)).item() <= 1e-7


def test_dot():

    def check():
        x1 = t1.full().numpy()
        x2 = t2.full().numpy()
        assert abs(tn.dot(t1, t2).item() - np.sum(x1*x2)) <= 1e-7

    t1 = tn.rand(np.random.randint(1, 8, np.random.randint(1, 6)), ranks_tt=2, ranks_tucker=None)
    t2 = tn.rand(t1.shape, ranks_tt=3, ranks_tucker=None)
    check()

    t1 = tn.rand(np.random.randint(1, 8, np.random.randint(1, 6)), ranks_tt=2, ranks_tucker=4)
    t2 = tn.rand(t1.shape, ranks_tt=3, ranks_tucker=None)
    check()

    t1 = tn.rand(np.random.randint(1, 8, np.random.randint(1, 6)), ranks_tt=2, ranks_tucker=None)
    t2 = tn.rand(t1.shape, ranks_tt=3, ranks_tucker=4)
    check()

    t1 = tn.rand(np.random.randint(1, 8, np.random.randint(1, 6)), ranks_tt=2, ranks_tucker=3)
    t2 = tn.rand(t1.shape, ranks_tt=3, ranks_tucker=4)
    check()

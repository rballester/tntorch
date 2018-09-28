import numpy as np
import tntorch as tn
import torch


def test_ops():

    def check():
        x1 = t1.full()
        x2 = t2.full()
        assert tn.relative_error(t1+t2, x1+x2) <= 1e-7
        assert tn.relative_error(t1-t2, x1-x2) <= 1e-7
        assert tn.relative_error(t1*t2, x1*x2) <= 1e-7
        assert tn.relative_error(-t1+t2, -x1+x2) <= 1e-7

    for i in range(100):
        t1 = tn.rand(np.random.randint(1, 8, np.random.randint(1, 6)), ranks_tt=3, ranks_tucker=2)
        t2 = tn.rand(t1.shape)
        check()

    t1 = tn.rand([32] * 4, ranks_tt=[3, None, None], ranks_cp=[None, None, 2, 2], ranks_tucker=5)
    t2 = tn.rand([32]*4, ranks_tt=[None, 2, None], ranks_cp=[4, None, None, 3])
    check()

    t2 = t1*2
    check()


def test_dot():

    def check():
        x1 = t1.full()
        x2 = t2.full()
        gt = torch.dot(x1.flatten(), x2.flatten())
        assert tn.relative_error(tn.dot(t1, t2), gt) <= 1e-7

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

    t1 = tn.rand([32] * 4, ranks_tt=[3, None, None], ranks_cp=[None, None, 10, 10], ranks_tucker=5)
    t2 = tn.rand([32]*4, ranks_tt=[None, 2, None], ranks_cp=[4, None, None, 5])
    check()

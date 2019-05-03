import numpy as np
import tntorch as tn
import torch
torch.set_default_dtype(torch.float64)
from util import random_format


def check(t1, t2):
    x1 = t1.torch()
    x2 = t2.torch()
    assert tn.relative_error(t1+t2, x1+x2) <= 1e-7
    assert tn.relative_error(t1-t2, x1-x2) <= 1e-7
    assert tn.relative_error(t1*t2, x1*x2) <= 1e-7
    assert tn.relative_error(-t1+t2, -x1+x2) <= 1e-7


def test_ops():

    for i in range(100):
        t1 = tn.rand(np.random.randint(1, 8, np.random.randint(1, 6)), ranks_tt=3, ranks_tucker=2)
        t2 = tn.rand(t1.shape)
        check(t1, t2)

    shape = [8]*4

    t1 = tn.rand(shape, ranks_tt=[3, None, None], ranks_cp=[None, None, 2, 2], ranks_tucker=5)
    t2 = tn.rand(shape, ranks_tt=[None, 2, None], ranks_cp=[4, None, None, 3])
    check(t1, t2)

    t2 = t1*2
    check(t1, t2)

    for i in range(100):
        t1 = random_format(shape)
        t2 = random_format(shape)
        check(t1, t2)


def test_broadcast():

    for i in range(10):
        shape1 = np.random.randint(1, 10, 4)
        shape2 = shape1.copy()
        shape2[np.random.choice(len(shape1), np.random.randint(0, len(shape1)+1))] = 1
        t1 = random_format(shape1)
        t2 = random_format(shape2)
        check(t1, t2)


def test_dot():

    def check():
        x1 = t1.torch()
        x2 = t2.torch()
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

    shape = [8]*4
    for i in range(100):
        t1 = random_format(shape)
        t2 = random_format(shape)
        check()


def test_stats():

    def check():
        x = t.torch()
        assert tn.relative_error(tn.mean(t), torch.mean(x)) <= 1e-3
        assert tn.relative_error(tn.var(t), torch.var(x)) <= 1e-3
        assert tn.relative_error(tn.norm(t), torch.norm(x)) <= 1e-3

    shape = [8]*4
    for i in range(100):
        t = random_format(shape)
        check()


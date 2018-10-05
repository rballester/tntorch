import numpy as np
import tntorch as tn
from util import random_format


def check(x, t, idx):

    xidx = x[idx]
    tidx = t[idx].numpy()
    assert np.array_equal(xidx.shape, tidx.shape)
    assert np.linalg.norm(xidx - tidx) / np.linalg.norm(xidx) <= 1e-7


def test_squeeze():

    for i in range(100):
        x = np.random.randint(1, 3, np.random.randint(2, 10))
        t = tn.Tensor(x)
        x = np.squeeze(x)
        t = tn.squeeze(t)
        assert np.array_equal(x.shape, t.shape)


def test_slicing():

    t = tn.rand(shape=[1, 3, 1, 2, 1], ranks_tt=3, ranks_tucker=2)
    x = t.numpy()
    idx = slice(None)
    check(x, t, idx)
    idx = (slice(None), slice(1, None))
    check(x, t, idx)
    idx = (slice(None), slice(0, 2, None), slice(0, 1))
    check(x, t, idx)


def test_mixed():

    def check_one_tensor(t):

        x = t.numpy()

        idxs = []
        idxs.append(([0, 0, 0], None, None, 3))
        idxs.append(([0, 0, 0, 0, 0], slice(None), None, 0))
        idxs.append((0, [0]))
        idxs.append(([0], [0]))
        idxs.append(([0], None, None, None, 0, 1))
        idxs.append((slice(None), [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
        idxs.append(([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
        idxs.append((slice(None), slice(None), slice(None), 0))
        idxs.append((slice(None), slice(None), [0, 1], 0))
        idxs.append((0, np.array([0]), None, 0))

        for idx in idxs:
            check(x, t, idx)

    check_one_tensor(tn.rand(shape=[6, 7, 8, 9], ranks_tt=3, ranks_tucker=2))
    check_one_tensor(tn.rand(shape=[6, 7, 8, 9], ranks_tt=None, ranks_tucker=2, ranks_cp=3))
    check_one_tensor(tn.rand(shape=[6, 7, 8, 9], ranks_tt=[4, None, None], ranks_tucker=2, ranks_cp=[None, None, 3, 3]))
    check_one_tensor(tn.rand(shape=[6, 7, 8, 9], ranks_tt=[4, None, None], ranks_tucker=[2, None, 2, None], ranks_cp=[None, None, 3, 3]))
    check_one_tensor(tn.rand(shape=[6, 7, 8, 9], ranks_tt=[None, 4, 4], ranks_tucker=2, ranks_cp=[3, None, None, None]))

    for i in range(100):
        check_one_tensor(random_format(shape=[6, 7, 8, 9]))

    t = tn.rand([6, 7, 8, 9], ranks_cp=[3, 3, 3, 3])
    t.cores[-1] = t.cores[-1].permute(1, 0)[:, :, None]
    check_one_tensor(t)

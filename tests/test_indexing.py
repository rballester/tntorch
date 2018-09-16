from unittest import TestCase
import numpy as np
import tntorch as tn




class TestIndexing(TestCase):

    def check(self, x, t, idx):

        xidx = x[idx]
        tidx = t[idx].full().detach().numpy()
        self.assertTrue(np.array_equal(xidx.shape, tidx.shape))
        self.assertAlmostEqual(np.linalg.norm(xidx - tidx) / np.linalg.norm(xidx), 0)

    def test_squeeze(self):

        for i in range(100):
            x = np.random.randint(1, 3, np.random.randint(2, 10))
            t = tn.Tensor(x)
            x = np.squeeze(x)
            t = tn.squeeze(t)
            self.assertTrue(np.array_equal(x.shape, t.shape))

    def test_slicing(self):

        t = tn.rand(shape=[1, 3, 1, 2, 1], ranks_tt=3, ranks_tucker=2)
        x = t.full().detach().numpy()
        idx = slice(None)
        self.check(x, t, idx)
        idx = (slice(None), slice(1, None))
        self.check(x, t, idx)
        idx = (slice(None), slice(0, 2, None), slice(0, 1))
        self.check(x, t, idx)

    def test_mixed(self):

        t = tn.rand(shape=[6, 7, 8, 9], ranks_tt=3, ranks_tucker=2)
        x = t.full().detach().numpy()

        idxs = []
        idxs.append(([0, 0, 0], None, None, 3))
        idxs.append(([0, 0, 0, 0, 0], slice(None), None, 0))
        idxs.append((0, [0]))
        idxs.append(([0], [0]))
        idxs.append(([0], None, None, None, 0, 1))
        idxs.append((slice(None), [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))
        idxs.append(([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]))

        for idx in idxs:
            self.check(x, t, idx)

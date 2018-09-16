from unittest import TestCase
import numpy as np
import tntorch as tn


class TestOps(TestCase):

    def test_ops(self):

        for i in range(100):
            t1 = tn.rand(np.random.randint(1, 8, np.random.randint(1, 6)), ranks_tt=3, ranks_tucker=2)
            t2 = tn.rand(t1.shape)
            x1 = t1.full().numpy()
            x2 = t2.full().numpy()
            self.assertAlmostEqual(tn.relative_error(t1+t2, tn.Tensor(x1+x2)).item(), 0)
            self.assertAlmostEqual(tn.relative_error(t1-t2, tn.Tensor(x1-x2)).item(), 0)
            self.assertAlmostEqual(tn.relative_error(t1*t2, tn.Tensor(x1*x2)).item(), 0)
            self.assertAlmostEqual(tn.relative_error(-t1+t2, tn.Tensor(-x1+x2)).item(), 0)

    def test_dot(self):

        def check():
            x1 = t1.full().numpy()
            x2 = t2.full().numpy()
            self.assertAlmostEqual(tn.dot(t1, t2).item(), np.sum(x1*x2))

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

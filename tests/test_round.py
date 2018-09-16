from unittest import TestCase
import numpy as np
import tntorch as tn


class TestRound(TestCase):

    def test_orthogonalization(self):

        for i in range(100):
            gt = tn.rand(np.random.randint(1, 8, np.random.randint(2, 6)))
            t = gt.clone()
            self.assertAlmostEqual(tn.relative_error(gt, t).item(), 0)
            t.left_orthogonalize(0)
            self.assertAlmostEqual(tn.relative_error(gt, t).item(), 0)
            t.right_orthogonalize(t.ndim-1)
            self.assertAlmostEqual(tn.relative_error(gt, t).item(), 0)
            t.orthogonalize(np.random.randint(t.ndim))
            self.assertAlmostEqual(tn.relative_error(gt, t).item(), 0)

    def test_round_svd(self):

        for i in range(100):
            gt = tn.rand(np.random.randint(1, 8, np.random.randint(8, 10)), ranks_tt=np.random.randint(1, 10))
            gt.round(1e-8, algorithm='svd')
            t = gt+gt
            t.round(1e-8, algorithm='svd')
            self.assertAlmostEqual(tn.relative_error(gt, t/2).item(), 0, places=4)
            self.assertEqual(max(gt.ranks_tt), max(t.ranks_tt))

    def test_round_eig(self):

        for i in range(100):
            gt = tn.rand(np.random.randint(1, 8, np.random.randint(8, 10)), ranks_tt=np.random.randint(1, 10))
            gt.round(1e-8, algorithm='eig')
            t = gt+gt
            t.round(1e-8, algorithm='eig')
            self.assertAlmostEqual(tn.relative_error(gt, t/2).item(), 0, places=7)

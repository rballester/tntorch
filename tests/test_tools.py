from unittest import TestCase
import numpy as np
import tntorch as tn


class TestTools(TestCase):

    def test_cat(self):

        for i in range(100):
            N = np.random.randint(1, 4)
            sh1 = np.random.randint(1, 10, N)
            mode = np.random.randint(N)
            sh2 = sh1.copy()
            sh2[mode] = np.random.randint(1, 10)
            t1 = tn.rand(sh1, ranks_tt=2, ranks_tucker=2)
            t2 = tn.rand(sh2, ranks_tt=2)
            gt = np.concatenate([t1.numpy(), t2.numpy()], mode)
            self.assertAlmostEqual(np.linalg.norm(gt - tn.cat([t1, t2], mode).numpy()), 0)

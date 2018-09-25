from unittest import TestCase
import numpy as np
import tntorch as tn


class TestInit(TestCase):

    def test_from_ndarray(self):

        for i in range(100):
            gt = np.random.rand(*np.random.randint(1, 8, np.random.randint(1, 6)))
            t = tn.Tensor(gt)
            reco = t.numpy()
            self.assertAlmostEqual(np.linalg.norm(gt - reco) / np.linalg.norm(gt), 0)

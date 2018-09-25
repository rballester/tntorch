from unittest import TestCase
import numpy as np
import tntorch as tn


class TestDerivatives(TestCase):

    def test_divergence(self):

        t = tn.rand([10] * 3 + [3], ranks_tt=3)
        d = tn.divergence([t[..., 0], t[..., 1], t[..., 2]])
        x = t.numpy()

        def partial(x, mode):
            return np.concatenate([np.diff(x, axis=mode),
                                   np.zeros([sh for sh in x.shape[:mode]] + [1] + [sh for sh in x.shape[mode+1:]])],
                                  axis=mode)

        gt = partial(x[..., 0], 0)
        gt += partial(x[..., 1], 1)
        gt += partial(x[..., 2], 2)
        self.assertAlmostEqual(np.linalg.norm(d.numpy() - gt) / np.linalg.norm(gt), 0)
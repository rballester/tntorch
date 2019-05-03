import numpy as np
import tntorch as tn
import torch
torch.set_default_dtype(torch.float64)


def test_from_ndarray():

    for i in range(100):
        gt = np.random.rand(*np.random.randint(1, 8, np.random.randint(1, 6)))
        t = tn.Tensor(gt)
        reco = t.numpy()
        assert np.linalg.norm(gt - reco) / np.linalg.norm(gt) <+ 1e-7

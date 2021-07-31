import tntorch as tn
import numpy as np
import torch
torch.set_default_dtype(torch.float64)


def test_weight_mask():

    for N in range(1, 5):
        for k in range(1, N):
            gt = tn.automata.weight_mask(N, k)
            idx = torch.Tensor(np.array(np.unravel_index(np.arange(gt.numel(), dtype=int), list(gt.shape))).T)
            assert torch.norm((torch.sum(idx, dim=1).round() == k).float() - gt[idx].torch().round().float()) <= 1e-7

def test_accepted_inputs():

    for i in range(10):
        gt = tn.Tensor(torch.randint(0, 2, (1, 2, 3, 4)))
        idx = tn.automata.accepted_inputs(gt)
        assert len(idx) == round(tn.sum(gt).item())
        assert torch.norm(gt[idx].torch() - 1).item() <= 1e-7

from unittest import TestCase
import tntorch as tn
import torch
import numpy as np


class TestAutomata(TestCase):

    def test_weight_mask(self):

        for N in range(1, 5):
            for k in range(1, N):
                gt = tn.automata.weight_mask(N, k)
                idx = torch.Tensor(np.array(np.unravel_index(np.arange(gt.size, dtype=np.int), list(gt.shape))).T)
                self.assertAlmostEqual(torch.norm((torch.sum(idx, dim=1).round() == k).float() - gt[idx].full().round().float()), 0)

    def test_accepted_inputs(self):

        for i in range(10):
            gt = tn.Tensor(torch.randint(0, 2, (1, 2, 3, 4)))
            idx = tn.automata.accepted_inputs(gt)
            self.assertEqual(len(idx), round(tn.sum(gt).item()))
            self.assertAlmostEqual(torch.norm(gt[idx].full()-1).item(), 0)

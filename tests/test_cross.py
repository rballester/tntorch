import numpy as np
import tntorch as tn
import torch
from util import random_format


def test_domain():

    def function(Xs):
        return 1. / torch.sum(Xs, dim=1)

    domain = [torch.linspace(1, 10, 10) for n in range(3)]
    t = tn.cross(function=function, domain=domain, ranks_tt=3)
    gt = torch.meshgrid(domain)
    gt = 1. / sum(gt)

    assert tn.relative_error(gt, t) < 5e-2


def test_tensors():

    for i in range(100):
        t = random_format([10] * 6)
        t2 = tn.cross(function=lambda x: x, tensors=t, ranks_tt=15, verbose=False)
        assert tn.relative_error(t, t2) < 1e-6

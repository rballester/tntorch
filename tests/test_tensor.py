import tntorch as tn
import torch
from tntorch.tensor import lstsq
torch.set_default_dtype(torch.float64)

def test_lstsq():
    A = torch.rand((5, 6))
    b = torch.rand((5, 6))

    assert torch.allclose(lstsq(b, A, 'lstsq'), lstsq(b, A, 'qr'))
    assert torch.norm(lstsq(b, A, 'qr') - lstsq(b, A, 'cvxpylayers', lam=0, eps=1e-8)) < 1e-3

def test_tensor():
    a = torch.rand(10, 5, 5, 5, 5)
    b = tn.Tensor(a, batch=True)

    for i in range(len(a)):
        c = tn.Tensor(a[i], batch=False)

        for j, core in enumerate(c.cores):
            assert torch.allclose(core, b.cores[j][i, ...])

        assert torch.allclose(c.torch(), b.torch()[i])


def test_tt_tensor():
    a = torch.rand(10, 5, 5, 5, 5)
    b = tn.Tensor(a, ranks_tt=3, batch=True)

    for i in range(len(a)):
        c = tn.Tensor(a[i], ranks_tt=3, batch=False)

        for j, core in enumerate(c.cores):
            assert torch.allclose(core, b.cores[j][i, ...])

        assert torch.allclose(c.torch(), b.torch()[i])

    a = torch.rand(10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    b = tn.Tensor(a, ranks_tt=3, batch=True)

    for i in range(len(a)):
        c = tn.Tensor(a[i], ranks_tt=3, batch=False)

        for j, core in enumerate(c.cores):
            assert torch.allclose(core, b.cores[j][i, ...])

        assert torch.allclose(c.torch(), b.torch()[i])


def test_cp_tensor():
    a = torch.rand(10, 5, 5, 5, 5)
    b = tn.Tensor(a, ranks_cp=3, batch=True)

    for i in range(len(a)):
        c = tn.Tensor(a[i], ranks_cp=3, batch=False)

        for j, core in enumerate(c.cores):
            assert torch.norm(core - b.cores[j][i, ...]) < 1e1 # Due to random initialization

        assert torch.norm(c.torch() - b.torch()[i]) < 1e1 # Due to random initialization


def test_tucker_tensor():
    a = torch.rand(10, 5, 5, 5, 5)
    b = tn.Tensor(a, ranks_tucker=3, batch=True)

    for i in range(len(a)):
        c = tn.Tensor(a[i], ranks_tucker=3, batch=False)

        for j, core in enumerate(c.cores):
            assert torch.allclose(core, b.cores[j][i, ...])

        assert torch.allclose(c.torch(), b.torch()[i])


def test_tt_tensor_eig():
    a = torch.rand(10, 5, 5, 5, 5)
    b = tn.Tensor(a, ranks_tucker=3, batch=True, algorithm='eig')

    for i in range(len(a)):
        c = tn.Tensor(a[i], ranks_tucker=3, batch=False, algorithm='eig')

        for j, core in enumerate(c.cores):
            assert torch.allclose(core, b.cores[j][i, ...])

        assert torch.allclose(c.torch(), b.torch()[i])

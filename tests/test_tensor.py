import tntorch as tn
import pytest
import torch
from tntorch.tensor import lstsq
torch.set_default_dtype(torch.float64)

torch.manual_seed(1)

def test_lstsq():
    a = torch.rand((5, 6))
    b = torch.rand((5, 6))

    assert torch.allclose(lstsq(b, a, 'lstsq'), lstsq(b, a, 'qr'))
    assert torch.norm(lstsq(b, a, 'qr') - lstsq(b, a, 'cvxpylayers', lam=0, eps=1e-8)) < 1e-3

    a = torch.rand((10, 5, 6))
    b = torch.rand((10, 5, 6))

    assert torch.allclose(lstsq(b, a, 'lstsq'), lstsq(b, a, 'qr'))

    a = torch.rand((10, 5, 6))
    b = torch.rand((10, 5, 7))

    assert torch.allclose(lstsq(b, a, 'lstsq'), lstsq(b, a, 'qr'))

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
            assert torch.norm(core - b.cores[j][i, ...]) < 1e1

        print(torch.norm(c.torch() - b.torch()[i]))
        assert torch.norm(c.torch() - b.torch()[i]) < 1e1


def test_tucker_tensor():
    a = torch.rand(10, 5, 5, 5, 5)
    b = tn.Tensor(a, ranks_tucker=3, batch=True)

    for i in range(len(a)):
        c = tn.Tensor(a[i], ranks_tucker=3, batch=False)

        for j, core in enumerate(c.cores):
            assert torch.allclose(core, b.cores[j][i, ...])

        assert torch.allclose(c.torch(), b.torch()[i])


def test_tucker_cp_tensor():
    a = torch.rand(10, 5, 5, 5, 5)
    b = tn.Tensor(a, ranks_tucker=3, ranks_cp=4, batch=True)

    for i in range(len(a)):
        c = tn.Tensor(a[i], ranks_tucker=3, ranks_cp=4, batch=False)

        assert torch.norm(c.torch() - b.torch()[i]) < 1e1


def test_tt_tensor_eig():
    a = torch.rand(10, 5, 5, 5, 5)
    b = tn.Tensor(a, ranks_tucker=3, batch=True, algorithm='eig')

    for i in range(len(a)):
        c = tn.Tensor(a[i], ranks_tucker=3, batch=False, algorithm='eig')

        assert torch.allclose(c.torch(), b.torch()[i])

    a = torch.rand(10, 5, 5, 5, 5)
    b = tn.Tensor(a, ranks_tt=3, batch=True, algorithm='eig')

    for i in range(len(a)):
        c = tn.Tensor(a[i], ranks_tt=3, batch=False, algorithm='eig')

        assert torch.allclose(c.torch(), b.torch()[i])

    a = torch.rand(10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    b = tn.Tensor(a, ranks_tt=3, batch=True)

    for i in range(len(a)):
        c = tn.Tensor(a[i], ranks_tt=3, batch=False, algorithm='eig')

        assert torch.allclose(c.torch(), b.torch()[i])


def test_sum():
    a = tn.rand((10, 5, 6), ranks_tt=3)
    b = tn.rand((10, 5, 6), ranks_tt=3)

    assert torch.allclose((a + b).torch(), a.torch() + b.torch())

    a = tn.rand((10, 5, 6), ranks_cp=3)
    b = tn.rand((10, 5, 6), ranks_cp=3)

    assert torch.allclose((a + b).torch(), a.torch() + b.torch())

    a = tn.rand((10, 5, 6), ranks_tucker=3)
    b = tn.rand((10, 5, 6), ranks_tucker=3)

    assert torch.allclose((a + b).torch(), a.torch() + b.torch())

    a = tn.rand((10, 5, 6), ranks_tucker=3, ranks_cp=3)
    b = tn.rand((10, 5, 6), ranks_tucker=3, ranks_cp=3)

    assert torch.allclose((a + b).torch(), a.torch() + b.torch())

    a = tn.rand((10, 5, 6), ranks_tucker=3, ranks_tt=3)
    b = tn.rand((10, 5, 6), ranks_tucker=3, ranks_tt=3)

    assert torch.allclose((a + b).torch(), a.torch() + b.torch())

    a = tn.rand((10, 5, 6), ranks_tt=3)
    b = tn.rand((10, 5, 6), ranks_tucker=3)

    assert torch.allclose((a + b).torch(), a.torch() + b.torch())

    a = tn.rand((10, 5, 6), ranks_tt=3)
    b = tn.rand((10, 5, 6), ranks_cp=3)

    assert torch.allclose((a + b).torch(), a.torch() + b.torch())

    a = tn.rand((10, 5, 6), ranks_tucker=3)
    b = tn.rand((10, 5, 6), ranks_cp=3)

    assert torch.allclose((a + b).torch(), a.torch() + b.torch())

    with pytest.raises(ValueError) as exc_info:
        a = tn.rand((10, 5, 6), ranks_cp=3, ranks_tt=3)
    assert exc_info.value.args[0] == 'The ranks_tt and ranks_cp provided are incompatible'

    a = tn.rand((10, 5, 6), ranks_tt=3, batch=True)
    b = tn.rand((10, 5, 6), ranks_tt=3, batch=True)

    assert torch.allclose((a + b).torch(), a.torch() + b.torch())

    a = tn.rand((10, 5, 6), ranks_cp=3, batch=True)
    b = tn.rand((10, 5, 6), ranks_cp=3, batch=True)

    assert torch.allclose((a + b).torch(), a.torch() + b.torch())
    
    a = tn.rand((10, 5, 6), ranks_tucker=3, batch=True)
    b = tn.rand((10, 5, 6), ranks_tucker=3, batch=True)

    assert torch.allclose((a + b).torch(), a.torch() + b.torch())

    a = tn.rand((10, 5, 6), ranks_tt=3)
    b = 5
    assert torch.allclose((a + b).torch(), a.torch() + b)


def test_mul():
    a = tn.rand((10, 5, 6), ranks_tt=3)
    b = tn.rand((10, 5, 6), ranks_tt=3)

    assert torch.allclose((a * b).torch(), a.torch() * b.torch())

    a = tn.rand((10, 5, 6), ranks_cp=3)
    b = tn.rand((10, 5, 6), ranks_cp=3)

    assert torch.allclose((a * b).torch(), a.torch() * b.torch())

    a = tn.rand((10, 5, 6), ranks_tucker=3)
    b = tn.rand((10, 5, 6), ranks_tucker=3)

    assert torch.allclose((a * b).torch(), a.torch() * b.torch())

    a = tn.rand((10, 5, 6), ranks_tucker=3, ranks_cp=3)
    b = tn.rand((10, 5, 6), ranks_tucker=3, ranks_cp=3)

    assert torch.allclose((a * b).torch(), a.torch() * b.torch())

    a = tn.rand((10, 5, 6), ranks_tucker=3, ranks_tt=3)
    b = tn.rand((10, 5, 6), ranks_tucker=3, ranks_tt=3)

    assert torch.allclose((a * b).torch(), a.torch() * b.torch())

    a = tn.rand((10, 5, 6), ranks_tt=3)
    b = tn.rand((10, 5, 6), ranks_tucker=3)

    assert torch.allclose((a * b).torch(), a.torch() * b.torch())

    a = tn.rand((10, 5, 6), ranks_tt=3)
    b = tn.rand((10, 5, 6), ranks_cp=3)

    assert torch.allclose((a * b).torch(), a.torch() * b.torch())

    a = tn.rand((10, 5, 6), ranks_tucker=3)
    b = tn.rand((10, 5, 6), ranks_cp=3)

    assert torch.allclose((a * b).torch(), a.torch() * b.torch())

    with pytest.raises(ValueError) as exc_info:
        a = tn.rand((10, 5, 6), ranks_cp=3, ranks_tt=3)
    assert exc_info.value.args[0] == 'The ranks_tt and ranks_cp provided are incompatible'

    a = tn.rand((10, 5, 6), ranks_tt=3, batch=True)
    b = tn.rand((10, 5, 6), ranks_tt=3, batch=True)

    assert torch.allclose((a * b).torch(), a.torch() * b.torch())

    a = tn.rand((10, 5, 6), ranks_cp=3, batch=True)
    b = tn.rand((10, 5, 6), ranks_cp=3, batch=True)

    assert torch.allclose((a * b).torch(), a.torch() * b.torch())
    
    a = tn.rand((10, 5, 6), ranks_tucker=3, batch=True)
    b = tn.rand((10, 5, 6), ranks_tucker=3, batch=True)

    assert torch.allclose((a * b).torch(), a.torch() * b.torch())

    a = tn.rand((10, 5, 6), ranks_tt=3, batch=True)
    b = 5
    assert torch.allclose((a + b).torch(), a.torch() + b)


def test_indexing():
    a = tn.rand((10, 5, 6), ranks_tt=3)
    b = a.torch()

    assert torch.allclose(a[None].torch(), b[None])
    assert torch.allclose(a[None, ..., None].torch(), b[None, ..., None])
    assert torch.allclose(a[0, ..., 1].torch(), b[0, ..., 1])
    assert torch.allclose(a[None, ..., 1].torch(), b[None, ..., 1])
    assert torch.allclose(a[None, ..., -1].torch(), b[None, ..., -1])
    assert torch.allclose(a[None, ..., -1].torch(), b[None, ..., -1])

    a = tn.rand((10, 5, 6), ranks_cp=3)
    b = a.torch()

    assert torch.allclose(a[None].torch(), b[None])
    assert torch.allclose(a[None, ..., None].torch(), b[None, ..., None])
    assert torch.allclose(a[0, ..., 1].torch(), b[0, ..., 1])
    assert torch.allclose(a[None, ..., 1].torch(), b[None, ..., 1])
    assert torch.allclose(a[None, ..., -1].torch(), b[None, ..., -1])
    assert torch.allclose(a[None, ..., -1].torch(), b[None, ..., -1])

    a = tn.rand((10, 5, 6), ranks_tucker=3)
    b = a.torch()

    assert torch.allclose(a[None].torch(), b[None])
    assert torch.allclose(a[None, ..., None].torch(), b[None, ..., None])
    assert torch.allclose(a[0, ..., 1].torch(), b[0, ..., 1])
    assert torch.allclose(a[None, ..., 1].torch(), b[None, ..., 1])
    assert torch.allclose(a[None, ..., -1].torch(), b[None, ..., -1])
    assert torch.allclose(a[None, ..., -1].torch(), b[None, ..., -1])

    a = tn.rand((10, 5, 6), ranks_tt=3, ranks_tucker=3)
    b = a.torch()

    assert torch.allclose(a[None].torch(), b[None])
    assert torch.allclose(a[None, ..., None].torch(), b[None, ..., None])
    assert torch.allclose(a[0, ..., 1].torch(), b[0, ..., 1])
    assert torch.allclose(a[None, ..., 1].torch(), b[None, ..., 1])
    assert torch.allclose(a[None, ..., -1].torch(), b[None, ..., -1])
    assert torch.allclose(a[None, ..., -1].torch(), b[None, ..., -1])

    a = tn.rand((10, 5, 6), ranks_cp=3, ranks_tucker=3)
    b = a.torch()

    assert torch.allclose(a[None].torch(), b[None])
    assert torch.allclose(a[None, ..., None].torch(), b[None, ..., None])
    assert torch.allclose(a[0, ..., 1].torch(), b[0, ..., 1])
    assert torch.allclose(a[None, ..., 1].torch(), b[None, ..., 1])
    assert torch.allclose(a[None, ..., -1].torch(), b[None, ..., -1])
    assert torch.allclose(a[None, ..., -1].torch(), b[None, ..., -1])

    a = tn.rand((10, 5, 6), ranks_tt=3, batch=True)
    b = a.torch()

    with pytest.raises(ValueError) as exc_info:
        a[None].torch(), b[None]
    assert exc_info.value.args[0] == 'Cannot change batch dimension'

    assert torch.allclose(a[..., None].torch(), b[..., None])
    assert torch.allclose(a[0, ..., 1].torch(), b[0, ..., 1])
    assert torch.allclose(a[..., 1].torch(), b[..., 1])
    assert torch.allclose(a[..., -1].torch(), b[..., -1])
    assert torch.allclose(a[..., -1].torch(), b[..., -1])

    a = tn.rand((10, 5, 6), ranks_tucker=3, batch=True)
    b = a.torch()

    with pytest.raises(ValueError) as exc_info:
        a[None].torch(), b[None]
    assert exc_info.value.args[0] == 'Cannot change batch dimension'

    assert torch.allclose(a[..., None].torch(), b[..., None])
    assert torch.allclose(a[0, ..., 1].torch(), b[0, ..., 1])
    assert torch.allclose(a[..., 1].torch(), b[..., 1])
    assert torch.allclose(a[..., -1].torch(), b[..., -1])
    assert torch.allclose(a[..., -1].torch(), b[..., -1])

    a = tn.rand((10, 5, 6), ranks_cp=3, batch=True)
    b = a.torch()

    with pytest.raises(ValueError) as exc_info:
        a[None].torch(), b[None]
    assert exc_info.value.args[0] == 'Cannot change batch dimension'

    assert torch.allclose(a[..., None].torch(), b[..., None])
    assert torch.allclose(a[0, ..., 1].torch(), b[0, ..., 1])
    assert torch.allclose(a[..., 1].torch(), b[..., 1])
    assert torch.allclose(a[..., -1].torch(), b[..., -1])
    assert torch.allclose(a[..., -1].torch(), b[..., -1])

    a = tn.rand((10, 5, 6), ranks_cp=3, ranks_tucker=3, batch=True)
    b = a.torch()

    with pytest.raises(ValueError) as exc_info:
        a[None].torch(), b[None]
    assert exc_info.value.args[0] == 'Cannot change batch dimension'

    assert torch.allclose(a[..., None].torch(), b[..., None])
    assert torch.allclose(a[0, ..., 1].torch(), b[0, ..., 1])
    assert torch.allclose(a[..., 1].torch(), b[..., 1])
    assert torch.allclose(a[..., -1].torch(), b[..., -1])
    assert torch.allclose(a[..., -1].torch(), b[..., -1])

    a = tn.rand((10, 5, 6), ranks_tt=3, ranks_tucker=3, batch=True)
    b = a.torch()

    with pytest.raises(ValueError) as exc_info:
        a[None].torch(), b[None]
    assert exc_info.value.args[0] == 'Cannot change batch dimension'

    assert torch.allclose(a[..., None].torch(), b[..., None])
    assert torch.allclose(a[0, ..., 1].torch(), b[0, ..., 1])
    assert torch.allclose(a[..., 1].torch(), b[..., 1])
    assert torch.allclose(a[..., -1].torch(), b[..., -1])
    assert torch.allclose(a[..., -1].torch(), b[..., -1])
    
def test_round_tucker():
    a = tn.rand((10, 5, 6), ranks_tucker=3)
    b = a.clone()
    a.round_tucker(eps=1e-8)
    assert torch.norm(b.torch() - a.torch()) < 1e-8

    a = tn.rand((10, 5, 6), ranks_tucker=3, batch=True)
    b = a.clone()
    a.round_tucker(eps=1e-8)
    assert torch.norm(b.torch() - a.torch()) < 1e-8

def test_round_tt():
    a = tn.rand((10, 5, 6), ranks_tt=3)
    b = a.clone()
    a.round_tt(eps=1e-8)
    assert torch.norm(b.torch() - a.torch()) < 1e-8

    a = tn.rand((10, 5, 6), ranks_tt=3, batch=True)
    b = a.clone()
    a.round_tt(eps=1e-8)
    assert torch.norm(b.torch() - a.torch()) < 1e-8

    a = tn.rand((10, 5, 6), ranks_cp=3)
    b = a.clone()
    a.round_tt(eps=1e-8)
    assert torch.norm(b.torch() - a.torch()) < 1e-8

    a = tn.rand((10, 5, 6), ranks_cp=3, batch=True)
    b = a.clone()
    a.round_tt(eps=1e-8)
    assert torch.norm(b.torch() - a.torch()) < 1e-8

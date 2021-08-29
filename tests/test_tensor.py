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


def test_set_item():
    a = tn.rand((10, 5, 6), ranks_tt=3)
    b = a.torch()
    a[5, 2, 3] = 6
    b[5, 2, 3] = 6

    assert a[5, 2, 3] == b[5, 2, 3] and b[5, 2, 3] == 6

    a[5, 2, :] = 7
    b[5, 2, :] = 7

    assert (a[5, 2, :].torch() - b[5, 2, :]).sum() == 0 and b[5, 2, 0] == 7

    a[..., :] = 8
    b[..., :] = 8

    assert (a[..., :].torch() - b[..., :]).sum() == 0 and b[5, 2, 0] == 8

    a = tn.rand((10, 5, 6), ranks_tt=3)
    c = torch.zeros_like(b[:, 2, 0])
    i = torch.rand(c.shape)
    a[:, 2, 0] = i
    b[:, 2, 0] = i

    assert torch.allclose(a[:, 2, 0].torch(), b[:, 2, 0])

    c = torch.zeros_like(b[:, :, 0])
    add = torch.rand(c.shape)
    a[:, :, 0] = add
    b[:, :, 0] = add

    assert torch.allclose(a[:, :, 0].torch(), b[:, :, 0])

    c = torch.zeros_like(b[..., 3:5])
    add = torch.rand(c.shape)
    a[..., 3:5] = add
    b[..., 3:5] = add

    assert torch.allclose(a[..., 3:5].torch(), b[..., 3:5])

    a = tn.rand((10, 5, 6), ranks_tt=3)
    c = torch.zeros_like(b[2, :, 3:5])
    i = torch.rand(c.shape)
    a[2, :, 3:5] = i
    b[2, :, 3:5] = i

    assert torch.allclose(a[2, :, 3:5].torch(), b[2, :, 3:5])

    # batch
    a = tn.rand((10, 5, 6), ranks_tt=3, batch=True)
    b = a.torch()
    a[5] = 6
    b[5] = 6

    assert torch.allclose(a[5].torch(), b[5]) and a[5, 0, 0] == 6

    a = tn.rand((10, 5, 6), ranks_tt=3, batch=True)
    b = a.torch()
    a[5, 2, 3] = 6
    b[5, 2, 3] = 6

    assert a[5, 2, 3] == b[5, 2, 3] and b[5, 2, 3] == 6

    a[5, 2, :] = 7
    b[5, 2, :] = 7

    assert (a[5, 2, :].torch() - b[5, 2, :]).sum() == 0 and b[5, 2, 0] == 7

    a[..., :] = 8
    b[..., :] = 8

    assert (a[..., :].torch() - b[..., :]).sum() == 0 and b[5, 2, 0] == 8

    a = tn.rand((10, 5, 6), ranks_tt=3, batch=True)
    c = torch.zeros_like(b[:, 2, 0])
    i = torch.rand(c.shape)
    a[:, 2, 0] = i
    b[:, 2, 0] = i

    assert torch.allclose(a[:, 2, 0], b[:, 2, 0])

    c = torch.zeros_like(b[:, :, 0])
    add = torch.rand(c.shape)
    a[:, :, 0] = add
    b[:, :, 0] = add

    assert torch.allclose(a[:, :, 0].torch(), b[:, :, 0])

    c = torch.zeros_like(b[..., 3:5])
    add = torch.rand(c.shape)
    a[..., 3:5] = add
    b[..., 3:5] = add

    assert torch.allclose(a[..., 3:5].torch(), b[..., 3:5])

    a = tn.rand((10, 5, 6), ranks_tt=3, batch=True)
    c = torch.zeros_like(b[2, :, 3:5])
    i = torch.rand(c.shape)
    a[2, :, 3:5] = i
    b[2, :, 3:5] = i

    assert torch.allclose(a[2, :, 3:5].torch(), b[2, :, 3:5])

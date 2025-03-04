import parir
import pytest
import torch

@parir.jit
def copy_2d(dst, src):
    parir.label('i')
    parir.label('j')
    dst[:,:] = src[:,:]

@parir.jit
def copy_3d(dst, src):
    parir.label('i')
    parir.label('j')
    parir.label('k')
    dst[:,:,:] = src[:,:,:]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_copy_two_dims():
    N, M = 20, 30
    x = torch.randn((N, M), dtype=torch.float32)
    y = torch.empty_like(x)
    copy_2d(y, x, seq=True)

    x_cu = x.cuda()
    y_cu = torch.empty_like(x_cu)
    p = {
        'i': [parir.threads(N)],
        'j': [parir.threads(M)]
    }
    copy_2d(y_cu, x_cu, parallelize=p, cache=False)

    assert torch.allclose(y, y_cu.cpu())

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
def test_copy_three_dims():
    N, M, K = 20, 30, 40
    x = torch.randn((N, M, K), dtype=torch.float32)
    y = torch.empty_like(x)
    copy_3d(y, x, seq=True)

    x_cu = x.cuda()
    y_cu = torch.empty_like(x_cu)
    p = {
        'i': [parir.threads(N)],
        'j': [parir.threads(M)],
        'k': [parir.threads(K)]
    }
    copy_3d(y_cu, x_cu, parallelize=p, cache=False)

    assert torch.allclose(y, y_cu.cpu())

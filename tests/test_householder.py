import pytest
import torch

from qr_practice import blocked_compact_householder_qr, compact_householder_qr


def _assert_valid_qr(a: torch.Tensor, h: torch.Tensor, tau: torch.Tensor) -> None:
    if a.ndim == 2:
        a = a.unsqueeze(0)
        h = h.unsqueeze(0)
        tau = tau.unsqueeze(0)

    batch, n, _ = a.shape
    assert h.shape == (batch, n, n)
    assert tau.shape == (batch, n)
    assert h.dtype == a.dtype
    assert tau.dtype == a.dtype

    q = torch.linalg.householder_product(h, tau)
    r = torch.triu(h)
    a_check = a.double()
    q_check = q.double()
    r_check = r.double()

    factor = torch.linalg.matrix_norm(r_check - q_check.transpose(-1, -2) @ a_check, ord=1)
    recon = torch.linalg.matrix_norm(q_check @ r_check - a_check, ord=1)
    orth = torch.linalg.matrix_norm(
        q_check.transpose(-1, -2) @ q_check
        - torch.eye(n, dtype=torch.float64, device=a.device).expand(batch, n, n),
        ord=1,
    )
    scale = torch.linalg.matrix_norm(a_check, ord=1).clamp_min(1.0)
    eps = torch.finfo(a.dtype).eps
    assert bool((factor <= 100 * n * eps * scale).all())
    assert bool((recon <= 100 * n * eps * scale).all())
    assert bool((orth <= 100 * n * eps).all())


@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_unblocked_compact_qr_random(n: int) -> None:
    gen = torch.Generator().manual_seed(1000 + n)
    a = torch.randn((3, n, n), dtype=torch.float32, generator=gen)
    h, tau = compact_householder_qr(a)
    _assert_valid_qr(a, h, tau)


@pytest.mark.parametrize("block_size", [1, 2, 4])
@pytest.mark.parametrize("n", [2, 5, 9])
def test_blocked_compact_qr_random(n: int, block_size: int) -> None:
    gen = torch.Generator().manual_seed(2000 + 10 * n + block_size)
    a = torch.randn((3, n, n), dtype=torch.float32, generator=gen)
    h, tau = blocked_compact_householder_qr(a, block_size=block_size)
    _assert_valid_qr(a, h, tau)


@pytest.mark.parametrize("rank", [0, 2])
def test_blocked_compact_qr_rank_deficient(rank: int) -> None:
    gen = torch.Generator().manual_seed(3000 + rank)
    a = torch.randn((2, 6, 6), dtype=torch.float32, generator=gen)
    a[:, :, rank:] = 0
    h, tau = blocked_compact_householder_qr(a, block_size=3)
    _assert_valid_qr(a, h, tau)

from __future__ import annotations

import torch

from qr_practice import blocked_compact_householder_qr, compact_householder_qr


def unblocked_householder_kernel(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return compact_householder_qr(data)


def blocked_householder_kernel(
    data: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return blocked_compact_householder_qr(data, block_size=block_size)

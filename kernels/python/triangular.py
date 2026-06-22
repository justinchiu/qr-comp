from __future__ import annotations

import torch


def triangular_kernel(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = torch.triu(data).contiguous()
    tau = torch.zeros(data.shape[0], data.shape[1], dtype=data.dtype, device=data.device)
    return h, tau

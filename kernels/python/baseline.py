from __future__ import annotations

import torch


def geqrf_kernel(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.geqrf(data)

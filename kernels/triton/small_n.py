from __future__ import annotations

import torch


def triton_small_n_kernel(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Placeholder for a Triton small-n QR implementation.

    Keep this import-safe on non-CUDA machines. Add the real Triton dependency
    and JIT kernel only when developing on a CUDA box.
    """
    raise NotImplementedError("Triton small-n QR kernel is not implemented yet")

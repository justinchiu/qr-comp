from __future__ import annotations

import torch

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    return torch.geqrf(data)

import pytest
import torch

import local_eval
from qr_practice import blocked_compact_householder_qr


@pytest.mark.parametrize(
    "case",
    ["dense", "upper", "diagonal", "rankdef", "nearrank", "clustered", "band", "rowscale"],
)
def test_local_eval_accepts_torch_geqrf(case: str) -> None:
    data = local_eval.generate_input(batch=2, n=8, cond=2, seed=1234, case=case)
    good, message = local_eval.check_implementation(data, torch.geqrf(data))
    assert good, message


def test_local_eval_generates_mixed_batch() -> None:
    data = local_eval.generate_input(batch=6, n=8, cond=2, seed=5678, case="mixed")
    assert data.shape == (6, 8, 8)
    assert data.dtype == torch.float32
    good, message = local_eval.check_implementation(data, torch.geqrf(data))
    assert good, message


@pytest.mark.parametrize("case", ["dense", "rankdef", "clustered"])
def test_local_eval_accepts_practice_blocked_qr(case: str) -> None:
    data = local_eval.generate_input(batch=2, n=8, cond=1, seed=9000, case=case)
    good, message = local_eval.check_implementation(
        data,
        blocked_compact_householder_qr(data, block_size=4),
    )
    assert good, message

import pytest

import local_eval
from kernels.python import (
    blocked_householder_kernel,
    geqrf_kernel,
    triangular_kernel,
    unblocked_householder_kernel,
)


@pytest.mark.parametrize(
    "kernel",
    [
        geqrf_kernel,
        unblocked_householder_kernel,
        lambda data: blocked_householder_kernel(data, block_size=4),
    ],
)
def test_python_kernels_pass_dense_contract(kernel) -> None:
    data = local_eval.generate_input(batch=2, n=8, cond=1, seed=1234, case="dense")
    good, message = local_eval.check_implementation(data, kernel(data))
    assert good, message


def test_triangular_kernel_passes_upper_contract() -> None:
    data = local_eval.generate_input(batch=2, n=8, cond=1, seed=5678, case="upper")
    good, message = local_eval.check_implementation(data, triangular_kernel(data))
    assert good, message


def test_triangular_kernel_is_not_general_qr() -> None:
    data = local_eval.generate_input(batch=2, n=8, cond=1, seed=91011, case="dense")
    good, _ = local_eval.check_implementation(data, triangular_kernel(data))
    assert not good

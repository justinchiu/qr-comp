import pytest

from autotune.hardware import (
    A100_80GB_PCIE,
    A100_80GB_SXM,
    B200,
    H100_80GB_SXM,
    auto_block_sizes,
)


def test_b200_roofline_values() -> None:
    assert B200.gpu == "NVIDIA B200 SXM"
    assert B200.memory_gb == 180
    assert B200.bandwidth_tb_s == 8.0
    assert B200.fp32_tflops == 75.0
    assert B200.tf32_dense_tflops == 1125.0
    assert B200.fp16_bf16_dense_tflops == 2250.0
    assert B200.fp8_dense_tflops == 4500.0
    assert B200.fp4_dense_tflops == 9000.0


def test_b200_ridge_points() -> None:
    rows = B200.roofline_rows()
    assert rows["ridge_fp32_flop_per_byte"] == pytest.approx(9.375)
    assert rows["ridge_tf32_dense_flop_per_byte"] == pytest.approx(140.625)
    assert rows["ridge_fp16_bf16_dense_flop_per_byte"] == pytest.approx(281.25)


def test_b200_auto_block_sizes() -> None:
    assert auto_block_sizes(B200, [32]) == [1, 2, 4, 8, 16]
    assert auto_block_sizes(B200, [512, 1024]) == [16, 32, 64]


def test_a100_roofline_values() -> None:
    assert A100_80GB_SXM.gpu == "NVIDIA A100 80GB SXM"
    assert A100_80GB_SXM.memory_gb == 80
    assert A100_80GB_SXM.bandwidth_tb_s == pytest.approx(2.039)
    assert A100_80GB_SXM.fp32_tflops == 19.5
    assert A100_80GB_SXM.tf32_dense_tflops == 156.0
    assert A100_80GB_SXM.fp16_bf16_dense_tflops == 312.0
    assert A100_80GB_PCIE.bandwidth_tb_s == pytest.approx(1.935)


def test_a100_ridge_points() -> None:
    rows = A100_80GB_SXM.roofline_rows()
    assert rows["ridge_fp32_flop_per_byte"] == pytest.approx(19.5 / 2.039)
    assert rows["ridge_tf32_dense_flop_per_byte"] == pytest.approx(156.0 / 2.039)


def test_h100_roofline_values() -> None:
    assert H100_80GB_SXM.gpu == "NVIDIA H100 80GB SXM"
    assert H100_80GB_SXM.memory_gb == 80
    assert H100_80GB_SXM.bandwidth_tb_s == pytest.approx(3.35)
    assert H100_80GB_SXM.fp32_tflops == 67.0
    assert H100_80GB_SXM.tf32_dense_tflops == 494.5
    assert H100_80GB_SXM.fp16_bf16_dense_tflops == 989.5
    assert H100_80GB_SXM.fp8_dense_tflops == 1979.0

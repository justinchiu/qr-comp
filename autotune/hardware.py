from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class HardwareProfile:
    name: str
    gpu: str
    architecture: str
    memory_gb: int
    bandwidth_tb_s: float
    system_gpus: int
    system_memory_gb: int
    system_bandwidth_tb_s: float
    nvlink_generation: str
    nvlink_gpu_to_gpu_tb_s: float
    tensor_core_generation: str
    fp4_dense_tflops: float
    fp4_sparse_tflops: float
    fp8_dense_tflops: float
    fp8_sparse_tflops: float
    fp16_bf16_dense_tflops: float
    fp16_bf16_sparse_tflops: float
    tf32_dense_tflops: float
    tf32_sparse_tflops: float
    fp32_tflops: float
    fp64_tflops: float
    notes: str

    def block_size_candidates(self, n: int) -> tuple[int, ...]:
        if n <= 32:
            return (1, 2, 4, 8, 16)
        if n <= 64:
            return (4, 8, 16, 32)
        if n <= 352:
            return (8, 16, 32)
        if n <= 1024:
            return (16, 32, 64)
        return (32, 64, 128)

    def ridge_flop_per_byte(self, peak_tflops: float) -> float:
        return peak_tflops / self.bandwidth_tb_s

    def roofline_rows(self) -> dict[str, float]:
        return {
            "ridge_fp64_flop_per_byte": self.ridge_flop_per_byte(self.fp64_tflops),
            "ridge_fp32_flop_per_byte": self.ridge_flop_per_byte(self.fp32_tflops),
            "ridge_tf32_dense_flop_per_byte": self.ridge_flop_per_byte(
                self.tf32_dense_tflops
            ),
            "ridge_fp16_bf16_dense_flop_per_byte": self.ridge_flop_per_byte(
                self.fp16_bf16_dense_tflops
            ),
            "ridge_fp8_dense_flop_per_byte": self.ridge_flop_per_byte(
                self.fp8_dense_tflops
            ),
            "ridge_fp4_dense_flop_per_byte": self.ridge_flop_per_byte(
                self.fp4_dense_tflops
            ),
        }


B200 = HardwareProfile(
    name="b200",
    gpu="NVIDIA B200 SXM",
    architecture="Blackwell",
    memory_gb=180,
    bandwidth_tb_s=8.0,
    system_gpus=8,
    system_memory_gb=1440,
    system_bandwidth_tb_s=64.0,
    nvlink_generation="5",
    nvlink_gpu_to_gpu_tb_s=1.8,
    tensor_core_generation="5",
    # Per-GPU values derived from NVIDIA HGX B200 8-GPU system specs.
    # HGX B200 lists FP4 as sparse|dense and lists other Tensor Core precisions
    # as sparse values where dense is half the shown sparse value.
    fp4_dense_tflops=9000.0,
    fp4_sparse_tflops=18000.0,
    fp8_dense_tflops=4500.0,
    fp8_sparse_tflops=9000.0,
    fp16_bf16_dense_tflops=2250.0,
    fp16_bf16_sparse_tflops=4500.0,
    tf32_dense_tflops=1125.0,
    tf32_sparse_tflops=2250.0,
    fp32_tflops=75.0,
    fp64_tflops=37.0,
    notes=(
        "QR v2 leaderboard target. Treat CPU and non-B200 runs as smoke tests only. "
        "Roofline peaks are per-GPU dense values unless explicitly marked sparse."
    ),
)

A100_80GB_SXM = HardwareProfile(
    name="a100_80gb_sxm",
    gpu="NVIDIA A100 80GB SXM",
    architecture="Ampere",
    memory_gb=80,
    bandwidth_tb_s=2.039,
    system_gpus=1,
    system_memory_gb=80,
    system_bandwidth_tb_s=2.039,
    nvlink_generation="3",
    nvlink_gpu_to_gpu_tb_s=0.6,
    tensor_core_generation="3",
    fp4_dense_tflops=0.0,
    fp4_sparse_tflops=0.0,
    fp8_dense_tflops=0.0,
    fp8_sparse_tflops=0.0,
    fp16_bf16_dense_tflops=312.0,
    fp16_bf16_sparse_tflops=624.0,
    tf32_dense_tflops=156.0,
    tf32_sparse_tflops=312.0,
    fp32_tflops=19.5,
    fp64_tflops=19.5,
    notes=(
        "A100 80GB SXM is a development/profiling target for CUDA iteration, not the "
        "QR v2 leaderboard target."
    ),
)

A100_80GB_PCIE = HardwareProfile(
    name="a100_80gb_pcie",
    gpu="NVIDIA A100 80GB PCIe",
    architecture="Ampere",
    memory_gb=80,
    bandwidth_tb_s=1.935,
    system_gpus=1,
    system_memory_gb=80,
    system_bandwidth_tb_s=1.935,
    nvlink_generation="3",
    nvlink_gpu_to_gpu_tb_s=0.6,
    tensor_core_generation="3",
    fp4_dense_tflops=0.0,
    fp4_sparse_tflops=0.0,
    fp8_dense_tflops=0.0,
    fp8_sparse_tflops=0.0,
    fp16_bf16_dense_tflops=312.0,
    fp16_bf16_sparse_tflops=624.0,
    tf32_dense_tflops=156.0,
    tf32_sparse_tflops=312.0,
    fp32_tflops=19.5,
    fp64_tflops=9.7,
    notes=(
        "A100 80GB PCIe is a development/profiling target for CUDA iteration, not the "
        "QR v2 leaderboard target."
    ),
)

H100_80GB_SXM = HardwareProfile(
    name="h100_80gb_sxm",
    gpu="NVIDIA H100 80GB SXM",
    architecture="Hopper",
    memory_gb=80,
    bandwidth_tb_s=3.35,
    system_gpus=1,
    system_memory_gb=80,
    system_bandwidth_tb_s=3.35,
    nvlink_generation="4",
    nvlink_gpu_to_gpu_tb_s=0.9,
    tensor_core_generation="4",
    fp4_dense_tflops=0.0,
    fp4_sparse_tflops=0.0,
    fp8_dense_tflops=1979.0,
    fp8_sparse_tflops=3958.0,
    fp16_bf16_dense_tflops=989.5,
    fp16_bf16_sparse_tflops=1979.0,
    tf32_dense_tflops=494.5,
    tf32_sparse_tflops=989.0,
    fp32_tflops=67.0,
    fp64_tflops=34.0,
    notes=(
        "H100 80GB SXM is a strong CUDA/profiling target for iteration, but B200 "
        "remains the QR v2 leaderboard target."
    ),
)

PROFILES = {
    A100_80GB_PCIE.name: A100_80GB_PCIE,
    A100_80GB_SXM.name: A100_80GB_SXM,
    B200.name: B200,
    H100_80GB_SXM.name: H100_80GB_SXM,
}


def get_profile(name: str) -> HardwareProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown hardware profile {name!r}; known profiles: {known}") from exc


def auto_block_sizes(profile: HardwareProfile, ns: list[int]) -> list[int]:
    values = set()
    for n in ns:
        values.update(profile.block_size_candidates(n))
    return sorted(values)

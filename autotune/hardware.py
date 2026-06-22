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

PROFILES = {
    B200.name: B200,
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

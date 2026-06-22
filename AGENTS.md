# QR Comp Agent Guide

## Competition Summary

This repo targets GPU MODE `qr_v2`.

The performance target is NVIDIA B200. Every kernel, classifier, block-size
choice, and profiling workflow should be B200-aware. CPU timings on this Mac are
only smoke tests.

Use `B200.md` for roofline assumptions. Current per-GPU working peaks are:

```text
HBM bandwidth               8.0 TB/s
FP32                        75 TFLOP/s
TF32 Tensor Core dense      1,125 TFLOP/s
FP16/BF16 dense             2,250 TFLOP/s
FP8/FP6 dense               4,500 TFLOP/s
FP4 dense                   9,000 TFLOP/s
```

The FP32 ridge point is about `9.4` FLOP/byte. Treat lower-precision Tensor Core
paths as experimental until the final compact Householder output passes the
official FP32-style checker.

Implement:

```python
def custom_kernel(data):
    return H, tau
```

Input:

- `data`: CUDA FP32 tensor, shape `batch x n x n`

Output:

- `H`: CUDA FP32 tensor, shape `batch x n x n`
- `tau`: CUDA FP32 tensor, shape `batch x n`
- `triu(H)` is treated as `R`
- `Q = torch.linalg.householder_product(H, tau)`

The checker does not compare bit-for-bit against `torch.geqrf`. It validates QR
invariants against the original FP32 input:

- `R ~= Q.T @ A`
- `Q.T @ Q ~= I`
- `Q @ R ~= A`
- no NaN/Inf in materialized intermediates

The official input generator/checker is mirrored in `local_eval.py`.

## Important Constraints

- Final Popcorn upload is a single `submission.py` file.
- Development code can live in `kernels/`, `classifiers/`, and `autotune/`, but
  final code must be packed or inlined into `submission.py`.
- Do not depend on local imports in the final submitted file unless the submit
  path explicitly packs them.
- Returned factors must be FP32 compact Householder factors.
- Low precision can only be internal and only if the final `(H, tau)` passes.
- Cholesky QR is not banned, but it is a probe/risky fast path until it can
  cheaply produce valid compact Householder output and avoid hard-case failures.
- Do not use CUDA streams in kernels/submission code. Treat stream usage as
  disallowed for this competition workflow because it can create timing and
  synchronization behavior that is too easy to abuse or mismeasure. Use the
  default stream and explicit correctness-preserving synchronization only in
  benchmark/profiling harnesses.

## Input Cases

Official generated cases include:

```text
dense
upper
diagonal
rankdef
nearrank
clustered
band
rowscale
nearcollinear
mixed
```

`mixed` batches contain independently assigned per-matrix conditioning profiles.
Never classify an entire batch from a small sample.

## Kernel Development Layout

Use:

```text
kernels/python/   runnable PyTorch/Python variants
kernels/cuda/     CUDA C++ templates and extension sources
kernels/triton/   Triton experiments
qr_practice/      readable math prototypes
autotune/         sweep and dispatch-table experiments
```

Runnable current variants:

```text
python_geqrf
python_unblocked
python_blocked
python_triangular
cholesky_probe
```

Stable baseline modules live under `baselines/`. Use
`baselines.geqrf_baseline` when profiling the `torch.geqrf` reference so later
changes to `submission.py` do not overwrite the baseline.

Future variants should be named by language and purpose:

```text
cuda_small_n32
cuda_blocked_wy_b32
cuda_robust_512_b32
triton_small_n
```

## Choosing Block Sizes

Block size means panel width for blocked Householder QR. The outer loop advances
by the block size, while the panel itself still factors one column at a time.

B200 is the tuning target. Start with candidate panel widths:

```text
n=32:       special small-n path, no large panel
n=176/352:  8, 16, 32
n=512:     16, 32, 64
n=1024:    16, 32, 64
n>=2048:   32, 64, 128
```

Choose block sizes empirically, not by taste. For each `(n, case, kernel,
block_size)` answer:

- Does it pass `local_eval.check_implementation`?
- Is panel factorization or trailing update dominant?
- Does larger block size improve GEMM-like trailing updates?
- Does larger block size hurt occupancy through registers/shared memory?
- Does NCU show the kernel is memory-bound, compute-bound, latency-bound, or
  launch-bound on B200?
- Does it still pass rank-deficient, near-rank-deficient, clustered, and mixed
  cases?
- Is a fast path faster enough to pay for classifier/mask/extra-launch overhead?

Use:

```bash
uv run --group practice python -m autotune.sweep \
  --hardware b200 \
  --n 32,64,128 \
  --cases dense,rankdef,clustered,mixed \
  --variants python_geqrf,python_blocked,cholesky_probe \
  --block-sizes 4,8,16,32,64 \
  --output results/block_sweep.csv
```

On a CUDA/B200-style machine:

```bash
uv run --group practice python -m autotune.sweep \
  --hardware b200 \
  --n 512,1024 \
  --batch 16 \
  --cases dense,rankdef,nearrank,clustered,rowscale,nearcollinear,mixed \
  --variants python_geqrf,python_blocked,cholesky_probe \
  --block-sizes 16,32,64,128 \
  --repeats 5 \
  --output results/b200_dispatch_sweep.csv
```

## Classifier Development

Classifiers must use cheap numeric signals only:

```text
n
batch
column norm ratio
row norm ratio
lower-triangular norm ratio
zero or tiny column norms
panel norm ratios
tiny reflector diagonal candidates
```

Useful split:

```python
easy_mask = classify_easy(A, features)
hard_mask = ~easy_mask
```

Then compare:

```text
split dispatch by mask
robust whole-batch dispatch
```

Do not split unless measurements show the fast path wins after masking and
extra-launch overhead.

## Local Validation

Install dependencies:

```bash
uv sync --group practice
```

Run checks:

```bash
uv run ruff check .
uv run --group practice pytest -q
uv run --group practice python local_benchmark.py --hardware b200 --suite smoke --mode benchmark
```

This validates mechanics only. Do not select final kernels or block sizes from
CPU results.

Run official-shape local checks on a CUDA machine:

```bash
uv run --group practice python local_benchmark.py --hardware b200 --suite official --mode test
uv run --group practice python local_benchmark.py --hardware b200 --suite official --mode benchmark
```

## Profiling

This Mac does not have an NVIDIA GPU. `ncu` must run on a CUDA machine.

List official benchmark cases:

```bash
uv run --group practice python local_benchmark.py --hardware b200 --suite official --list-cases
```

Profile one case:

```bash
QR_HARDWARE=b200 QR_CASE_INDEX=3 ./scripts/ncu_qr.sh
```

Profile the `torch.geqrf` baseline first:

```bash
./scripts/profile_geqrf_baseline.sh
```

Useful overrides:

```bash
QR_CASE_INDEX=7 NCU_SET=roofline ./scripts/ncu_qr.sh
QR_CASE_INDEX=3 NCU_KERNEL_NAME='.*qr.*' ./scripts/ncu_qr.sh
QR_CASE_INDEX=3 NCU_LAUNCH_SKIP=10 NCU_LAUNCH_COUNT=1 ./scripts/ncu_qr.sh
```

Reports go under `profiles/`, which is ignored by git.

Use NCU data to estimate arithmetic intensity and compare against the B200
ridge points in `B200.md`. Do not promote kernels based on wall time alone when
the bottleneck is unclear.

## Submission

Install/register Popcorn once:

```bash
curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
popcorn register discord
```

Submit:

```bash
popcorn submit --leaderboard qr_v2 --gpu B200 --mode test submission.py
popcorn submit --leaderboard qr_v2 --gpu B200 --mode benchmark submission.py
popcorn submit --leaderboard qr_v2 --gpu B200 --mode leaderboard submission.py
```

Before submitting, ensure `submission.py` is self-contained and does not rely on
unpacked local modules.

## Promotion Rule

Do not promote an experimental kernel into `submission.py` until it answers:

```text
What input bucket is it for?
Does it pass local_eval?
Is it faster than the current choice?
What profiler/autotune evidence explains the result?
Does it avoid CUDA stream usage?
```

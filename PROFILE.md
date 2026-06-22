# Nsight Compute Profiling

This Mac can edit and smoke-test the code, but `ncu` requires an NVIDIA GPU and
Nsight Compute on the machine that runs the workload. The target machine is B200;
non-B200 profiles are useful for debugging but not final decisions.

## Profiling Objective

Use profiling to answer one question:

```text
Where is the kernel losing B200 time: data movement, compute, synchronization,
or launch overhead?
```

The B200-specific design principle is to minimize data movement:

- avoid extra global reads/writes of the full matrix
- keep panel data in registers/shared memory
- reuse `V/T` over many trailing-update tiles
- make trailing updates GEMM-like
- avoid many tiny launches
- avoid split dispatch unless it beats mask/scatter and extra-launch costs

See [B200.md](B200.md) for peak rates and ridge points. The FP32 ridge point is
about `9.375 FLOP/byte`.

## Cases To Profile First

List official benchmark cases:

```bash
uv run --group practice python local_benchmark.py --hardware b200 --suite official --list-cases
```

Priority order:

```text
case 3:  batch=640, n=512,  dense      main leaderboard target
case 7:  batch=640, n=512,  mixed      mixed-batch dispatch stress
case 9:  batch=640, n=512,  rankdef    robust reflector stress
case 10: batch=640, n=512,  clustered  scale/pathology stress
case 4:  batch=60,  n=1024, dense      larger trailing-update target
case 8:  batch=60,  n=1024, mixed      robust larger case
case 11: batch=60,  n=1024, nearrank   hard correctness case
```

Profile `n=32`, `2048`, and `4096` after the 512/1024 paths have a credible
baseline, unless geometric-mean results show they dominate.

## Run NCU

On a CUDA/B200 machine:

```bash
uv sync --group practice
QR_HARDWARE=b200 QR_CASE_INDEX=3 ./scripts/ncu_qr.sh
```

Before profiling custom kernels, profile the stable `torch.geqrf` baseline:

```bash
uv sync --group practice
./scripts/profile_geqrf_baseline.sh
```

This runs `baselines.geqrf_baseline`, not `submission.py`, so the baseline stays
available after `submission.py` changes.

Useful overrides:

```bash
QR_CASE_INDEX=7 NCU_SET=roofline ./scripts/ncu_qr.sh
QR_CASE_INDEX=3 NCU_KERNEL_NAME='.*qr.*' ./scripts/ncu_qr.sh
QR_CASE_INDEX=3 NCU_LAUNCH_SKIP=10 NCU_LAUNCH_COUNT=1 ./scripts/ncu_qr.sh
QR_CASE_INDEX=3 QR_REPEATS=3 ./scripts/ncu_qr.sh
GEQRF_BASELINE_CASES="3 7" NCU_SET=full ./scripts/profile_geqrf_baseline.sh
```

Reports are written under `profiles/` and ignored by git.

The script profiles `local_benchmark.py` with correctness recheck disabled during
the profiled loop. Run correctness first:

```bash
uv run --group practice python local_benchmark.py \
  --hardware b200 \
  --suite official \
  --mode test
```

## Required Measurements

Collect at least:

```text
kernel duration
number of launches
DRAM throughput
DRAM bytes read/written
L2 throughput and hit rate
global load/store efficiency
SM active / SM throughput
achieved occupancy
registers per thread
shared memory per CTA
shared-memory bank conflicts
warp stall reasons
eligible warps per cycle
```

For Tensor Core experiments, also collect tensor-pipe utilization and precision
path metrics. Do not promote Tensor Core paths unless final compact Householder
output passes the FP32-style QR checker.

For the `torch.geqrf` baseline, record the cuSOLVER/kernel names that dominate
each case. This establishes whether the library baseline is spending time in
panel factorization, trailing updates, memory movement, or launch overhead.

## Bottleneck Classification

Use the profile to classify each kernel phase:

```text
memory-bound:
  DRAM throughput high, SM utilization low, arithmetic intensity below ridge

compute-bound:
  SM or tensor utilization high, DRAM not saturated

latency/synchronization-bound:
  neither DRAM nor compute saturated, warp stalls dominate

launch-bound:
  many short kernels, especially n=32 and classifier-heavy paths

occupancy-limited:
  low active warps due to register or shared-memory pressure
```

Panel factorization is expected to be latency/synchronization-bound. Trailing
updates should move toward the FP32 roofline by increasing reuse and arithmetic
intensity.

## Shape-Specific Questions

For `n=32`:

- Are we launch-bound?
- Does one CTA per matrix waste too much hardware?
- Can one CTA process multiple matrices?
- Are extra global round trips dominating?

For `n=176/352`:

- Does `b=16` or `b=32` improve trailing-update reuse?
- Does larger `b` reduce occupancy through registers/shared memory?
- Is panel work dominating too much of the runtime?

For `n=512`:

- Does `b=64` beat `b=32` on dense?
- Does `b=32` beat `b=64` on rankdef/clustered/mixed?
- Is the trailing update GEMM-like enough?
- Does split dispatch beat robust whole-batch dispatch?
- Are we reading/writing the trailing matrix more times than necessary?

For `n=1024`:

- Does larger tile/panel size improve arithmetic intensity?
- Is robust handling causing a measurable panel bottleneck?
- Are there enough CTAs per matrix to occupy B200?

For `n=2048/4096`:

- Is intra-matrix parallelism enough with small batch?
- Is fallback competitive?
- Does panel serialization dominate?
- Are tiled trailing updates saturating memory or compute?

## Profile-Autotune Loop

1. Run correctness:

```bash
uv run --group practice python local_benchmark.py --hardware b200 --suite official --mode test
```

2. Profile the stable `torch.geqrf` baseline:

```bash
./scripts/profile_geqrf_baseline.sh
```

3. Run autotune timing sweep:

```bash
uv run --group practice python -m autotune.sweep \
  --hardware b200 \
  --n 512,1024 \
  --batch 16 \
  --cases dense,rankdef,nearrank,clustered,rowscale,nearcollinear,mixed \
  --variants python_geqrf,python_blocked,cholesky_probe \
  --block-sizes auto \
  --repeats 5 \
  --output results/b200_dispatch_sweep.csv
```

4. Profile the fastest passing candidates and the surprising failures.

5. Update the dispatch/classifier only when the profile explains the timing.

6. Re-run Popcorn benchmark after packing into `submission.py`.

## Submission Note

Popcorn `qr_v2` currently does not provide a profile mode, so this workflow is
for a separate CUDA/B200 box where the repo can run directly.

# Kernel Development Loop

The goal is not to guess the final QR kernel. The goal is to create a repeatable
loop that turns one kernel idea into evidence: correctness, timing, profiler
data, and a dispatch decision.

## 1. Define The Kernel Variant

Each kernel idea should have a narrow purpose:

```text
small_n_householder
blocked_wy_b16
blocked_wy_b32
robust_householder_b32
triangular_fast_path
large_n_fallback
cholesky_probe
```

Record the intended scope before coding:

- target `n`
- target cases
- panel width
- expected fast path or robust path
- expected failure modes
- whether it returns valid compact Householder `(H, tau)`

If it cannot return compact Householder output, it is a probe, not a submission
kernel.

## 2. Prototype In Python First

Start in `qr_practice/` when the math is unclear.

Use `local_eval.py` to check the exact QR v2 contract:

```bash
uv run --group practice pytest
uv run --group practice python -m autotune.sweep \
  --n 8,16,32 \
  --cases dense,rankdef,clustered,mixed \
  --variants geqrf,blocked,cholesky \
  --block-sizes 2,4,8,16 \
  --output results/python_probe.csv
```

This phase answers: does the algorithm produce valid `(H, tau)` at all?

## 3. Add A Kernel Wrapper

Put candidate implementations under `kernels/` during development. Keep wrappers
small and explicit:

```python
def custom_kernel(data):
    return blocked_wy_kernel(data, block_size=32)
```

The repo can use multiple files for development. The final Popcorn upload must
still be a single `submission.py`, so do not rely on local imports unless there
is a pack step.

## 4. Test Correctness By Case

Run official-style generated cases locally:

```bash
uv run --group practice python local_benchmark.py --suite smoke --mode test
uv run --group practice python local_benchmark.py --suite official --mode test --max-cases 3
```

On a CUDA machine, run the full official test suite:

```bash
uv run --group practice python local_benchmark.py --suite official --mode test
```

Failure triage:

- shape or dtype error: output contract bug
- NaN/Inf: reflector norm or divide-by-zero bug
- `R - Q.T @ A` failure: factorization/update bug
- orthogonality failure: unstable reflector/update path
- hard case only: classifier or robust path bug

## 5. Sweep Parameters

Use autotune to measure block sizes and variants:

```bash
uv run --group practice python -m autotune.sweep \
  --n 32,64,128 \
  --cases dense,rankdef,clustered,mixed \
  --variants geqrf,blocked \
  --block-sizes 4,8,16,32,64 \
  --output results/block_sweep.csv
```

On a CUDA/B200-style machine:

```bash
uv run --group practice python -m autotune.sweep \
  --n 512,1024 \
  --batch 16 \
  --cases dense,rankdef,nearrank,clustered,rowscale,nearcollinear,mixed \
  --variants geqrf,blocked,cholesky \
  --block-sizes 16,32,64,128 \
  --repeats 5 \
  --output results/b200_dispatch_sweep.csv
```

The sweep should decide whether a kernel is:

- passing and fast enough to keep
- passing but slower than baseline
- correct only on easy cases
- not worth dispatch complexity
- useful only as a probe

## 6. Profile With NCU

After a kernel passes correctness and looks promising in timing, profile one
case at a time on a CUDA machine:

```bash
uv run --group practice python local_benchmark.py --suite official --list-cases
QR_CASE_INDEX=3 ./scripts/ncu_qr.sh
```

Useful variants:

```bash
QR_CASE_INDEX=7 NCU_SET=roofline ./scripts/ncu_qr.sh
QR_CASE_INDEX=3 NCU_KERNEL_NAME='.*qr.*' ./scripts/ncu_qr.sh
QR_CASE_INDEX=3 NCU_LAUNCH_SKIP=10 NCU_LAUNCH_COUNT=1 ./scripts/ncu_qr.sh
```

Profiler questions:

- Is the bottleneck panel factorization or trailing update?
- Is occupancy limited by registers or shared memory?
- Is global memory bandwidth saturated?
- Are there many tiny launches?
- Is the trailing update GEMM-like enough?
- Does the hard case take a different path than dense?

## 7. Update The Classifier

Classifiers should use cheap numeric signals only:

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

Classifier changes need evidence:

- which inputs move to the fast path
- which inputs move to the robust path
- correctness of each bucket
- runtime gain after mask/scatter/extra launch overhead

For mixed batches, compare:

```text
split dispatch by mask
robust whole-batch dispatch
```

Do not split unless it wins empirically.

## 8. Promote To Submission

When a kernel and dispatch rule are ready:

1. Add or update tests that cover the cases it claims to handle.
2. Add autotune CSV results under `results/` locally, but do not commit large raw
   profiler reports.
3. Pack or inline the required code into `submission.py`.
4. Run local checks:

```bash
uv run ruff check .
uv run --group practice pytest
uv run --group practice python local_benchmark.py --suite smoke --mode benchmark
```

5. Run Popcorn:

```bash
popcorn submit --leaderboard qr_v2 --gpu B200 --mode test submission.py
popcorn submit --leaderboard qr_v2 --gpu B200 --mode benchmark submission.py
popcorn submit --leaderboard qr_v2 --gpu B200 --mode leaderboard submission.py
```

## Keep / Kill Criteria

Keep a kernel if:

- it passes the official checker for its claimed input bucket
- it beats the current dispatch choice for that bucket
- it does not make mixed batches fragile
- its classifier cost is small relative to the speedup

Kill or demote a kernel if:

- it only wins on CPU
- it fails hard cases without a reliable classifier
- it needs expensive condition estimation
- it cannot produce compact Householder output
- it adds dispatch complexity without geometric-mean improvement

## Development Rule

Every kernel change should answer four questions:

```text
What input bucket is it for?
Does it pass local_eval?
Is it faster than the current choice?
What profiler evidence explains the result?
```

If we cannot answer those, keep it as an experiment and do not promote it into
`submission.py`.

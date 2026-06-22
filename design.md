# QR v2 Design Notes

## Contract

`custom_kernel(A)` receives a CUDA FP32 tensor with shape `batch x n x n`.
It must return `(H, tau)` in the compact Householder convention used by
`torch.geqrf`.

- `H`: FP32, shape `batch x n x n`
- `tau`: FP32, shape `batch x n`
- `triu(H)` is treated as `R`
- `Q = torch.linalg.householder_product(H, tau)`

The checker does not compare bitwise against `torch.geqrf`. It checks QR
invariants against the original input: `R ~= Q.T @ A`, `Q.T @ Q ~= I`, and
`Q @ R ~= A`.

## What Is a Panel?

A panel is a small block of adjacent columns that is factored before applying a
larger update to the rest of the matrix.

For panel width `b = 32`:

```text
columns:  0..31 | 32..n-1
          panel   trailing matrix
```

Inside a panel, Householder QR still advances one column at a time:

```python
for k in range(panel_start, panel_end):
    make_householder_for_column_k()
    update_remaining_columns_inside_panel()
```

After the panel is factored, the panel reflectors are applied together to the
trailing matrix:

```python
A_trailing = A_trailing - V @ T.T @ (V.T @ A_trailing)
```

Here `V` stores the panel's Householder vectors and `T` is the small triangular
compact-WY factor. The outer loop shifts by the panel width:

```python
for panel_start in range(0, n, b):
    factor_panel(panel_start, b)
    update_trailing_matrix(panel_start, b)
```

So unblocked QR shifts by one column. Blocked QR shifts by `b` columns outside,
while still shifting by one column inside each panel.

## Shape Dispatch

All dispatch decisions target B200. CPU and non-B200 runs are useful for
correctness and smoke timing only.

The central B200 principle is to minimize data movement. QR has serial panel
dependencies, so the panel work rarely reaches peak compute. The best chance to
use B200 well is to keep panel data resident and make trailing updates large,
tiled, coalesced, and GEMM-like.

Data-movement priorities:

1. Avoid extra global reads/writes of the full matrix.
2. Keep active panel data in registers/shared memory where possible.
3. Reuse `V` and `T` across many trailing-update tiles.
4. Fuse small operations when it does not harm correctness.
5. Avoid many tiny kernels, especially for `n=32`.
6. Prefer contiguous/coalesced access for trailing matrix updates.
7. Do not split mixed batches unless the saved compute outweighs mask/scatter
   and extra-launch costs.

Use shape to choose the kernel family first:

```python
if n <= 64:
    small_n_householder
elif n in (176, 352):
    medium_blocked_householder
elif n == 512:
    fast_or_robust_512
elif n == 1024:
    robust_1024
else:
    large_blocked_or_fallback
```

Suggested first panel widths:

```text
n=32:      1, 2, 4, 8, 16 or a special small-n path
n=176/352: 8, 16, 32
n=512:     16, 32, 64
n=1024:    16, 32, 64
n>=2048:   32, 64, 128
```

These are tuning starting points, not rules. The best value depends on shared
memory, register pressure, occupancy, and trailing-update efficiency.

## B200 Shape Hypotheses

Working model:

```text
panel factorization: latency/synchronization-bound
naive updates:       memory-bound
blocked updates:     best chance to become compute-efficient
```

For FP32 on B200, the rough ridge point is `75 TFLOP/s / 8 TB/s = 9.375`
FLOP/byte. Work below that arithmetic intensity is likely HBM-bound. Panel QR
will often be below the ridge or limited by synchronization; trailing updates
should be designed to reuse data enough to move right on the roofline.

### n=32, batch=20

Likely bottleneck: launch overhead, synchronization, and global-memory traffic,
not raw FLOPs.

Hypotheses:

- Use a specialized small-n path.
- Keep the full matrix in registers/shared memory.
- One CTA per matrix, or one CTA handling multiple matrices, should be tested.
- Avoid blocked-WY machinery unless it proves faster.
- Avoid Tensor Cores; setup overhead and QR dependencies likely dominate.
- Minimize full-matrix global round trips.

### n=176, batch=40

Likely bottleneck: panel overhead plus moderate trailing-update work.

Hypotheses:

- Test panel widths `8`, `16`, `32`.
- Start with `b=16`.
- Keep panels in shared memory.
- Use warp/block reductions for reflector norms.
- Make trailing updates tiled and FP32.
- Tensor Cores are experimental only; correctness risk is high.

### n=352, batch=40

Likely bottleneck: trailing updates start to matter more.

Hypotheses:

- Test `b=16` and `b=32`.
- `b=32` may win for dense inputs by improving update efficiency.
- `b=16` may be safer if register/shared-memory pressure hurts occupancy.
- Reuse panel reflectors across multiple trailing tiles.
- Coalesced global writes matter.

### n=512, batch=640

This is the main optimization target. Large batch makes occupancy easier, so
per-matrix data movement and update efficiency dominate.

Hypotheses:

- Build a specialized 512 path.
- Test `b=32` and `b=64`.
- Dense cases may prefer `b=64` if trailing updates dominate.
- Hard cases may prefer robust `b=32`.
- Keep `V/T` panel data resident and reused over trailing tiles.
- Make trailing updates as GEMM-like as possible.
- Classifier or mask dispatch is plausible only if the dense fast path is much
  faster than robust whole-batch dispatch.

### n=1024, batch=60

Less batch parallelism than `512`, but more work per matrix.

Hypotheses:

- Start with robust blocked Householder.
- Test `b=32` and `b=64`.
- Larger trailing tiles should improve arithmetic intensity.
- Dense may benefit from `b=64`.
- `mixed` and `nearrank` likely need robust `b=32` or careful panel checks.
- Avoid fragile approximate updates until FP32 path is passing.

### n=2048, batch=8

Trailing updates dominate, but batch parallelism is small.

Hypotheses:

- Need intra-matrix parallelism, not just one CTA per matrix.
- Test `b=32`, `b=64`, `b=128`.
- Multi-CTA trailing updates are likely required.
- Library fallback may be competitive early.
- Data movement dominates: avoid repeated full trailing-matrix reads.

### n=4096, batch=2

Tiny batch, huge matrices.

Hypotheses:

- Need strong intra-matrix parallelism.
- Panel factorization can become the serial bottleneck.
- Large tiled trailing updates are mandatory for B200 utilization.
- Conservative fallback is acceptable early while optimizing `512` and `1024`.
- Custom work here should be driven by geometric-mean evidence.

## B200 Case Hypotheses

- `upper` / `diagonal`: triangular fast path can be excellent if lower-triangle
  norm is truly negligible. Return `H=triu(A), tau=0`.
- `dense cond=1/2`: best candidate for fastest blocked path.
- `dense cond=4`: possible fast path, but residuals need careful testing.
- `rankdef`: robust Householder only; Cholesky is invalid.
- `nearrank`: robust Householder only.
- `clustered`: global scale checks can miss later hard panels; panel checks
  matter.
- `rowscale`: row norm ratio matters; column-only classifiers miss it.
- `nearcollinear`: orthogonality risk; avoid approximate updates.
- `mixed`: compare per-matrix mask dispatch against robust whole-batch dispatch.

Tensor Core hypothesis:

- Use scalar FP32 for reflector generation and robust panel work.
- First make FP32 tiled trailing updates fast and correct.
- Experiment with TF32/FP16/BF16/FP8 only for restricted dense cases or internal
  updates with validation/refinement.
- Do not use low precision if final compact Householder factors fail the
  official checker.

## Property Dispatch

Shape is not enough. `n=512` and `n=1024` include dense, mixed, rank-deficient,
clustered-scale, and near-rank-deficient cases. Classification should be
per-matrix, and later per-panel if we add split paths.

Cheap signals:

- column norm min/max
- row norm min/max
- matrix norm
- lower-triangular norm
- tiny or zero column norms
- panel norm ratios
- tiny diagonal candidate during reflector generation

Sketch:

```python
def classify(A):
    col_norms = norm(A, dim=1)
    row_norms = norm(A, dim=2)
    mat_norm = norm(A)

    col_ratio = max(col_norms) / clamp_min(min(col_norms), eps)
    row_ratio = max(row_norms) / clamp_min(min(row_norms), eps)
    lower_ratio = norm(tril(A, -1)) / clamp_min(mat_norm, eps)

    hard = (
        min(col_norms) < tiny * mat_norm
        or col_ratio > 1e4
        or row_ratio > 1e4
    )
    triangular = lower_ratio < 1e-7
    return hard, triangular
```

Thresholds need tuning against `local_eval.py`.

## Dispatch Strategy

Start conservative:

1. Use compact Householder everywhere.
2. Specialize memory layout, block size, and warp count by `n`.
3. Make the `512` path the main optimization target.
4. Make `1024` robust before making it clever.
5. Handle zero and near-zero reflector norms correctly.
6. Add triangular fast path only when the lower triangle is truly negligible.
7. Add property-based split paths only after the robust path is passing.

For mixed batches, a split dispatch can return a mask:

```python
easy_mask = classify_each_matrix(A)
fast_kernel(A[easy_mask])
robust_kernel(A[~easy_mask])
```

This is worthwhile only if the fast path saves more than the mask/scatter and
extra-launch overhead cost.

Use [EXPERIMENTS.md](EXPERIMENTS.md) to establish these choices empirically. The
dispatch table should come from CSV sweeps over `n`, generated case, panel type,
and block size, not from assumptions.

## Kernel Collection And Autotuning Plan

Use the repo to develop a collection of candidate kernels, then autotune dispatch
choices empirically.

Proposed layout:

```text
kernels/
  python/
    baseline.py
    householder.py
    triangular.py
  cuda/
    compact_householder.cu
  triton/
    small_n.py

classifiers/
  features.py
  classify.py

autotune/
  sweep.py
  summarize.py
  dispatch_table.py

submission.py
```

Important constraint: Popcorn submissions are a single `submission.py` file. The
repo can use `kernels/*.py` and `classifiers/*.py` for development, but the final
submitted file must either inline the selected code or be generated by a packing
script. Do not assume local imports will work in the leaderboard runtime.

Second important constraint: do not use CUDA streams in submission kernels. Use
the default stream. Streams can create timing and synchronization behavior that
is too easy to abuse or mismeasure for this competition workflow.

Kernel variants should carry an implementation language in autotune output:

```text
python_geqrf
python_blocked
python_triangular
cuda_small_n32
cuda_blocked_wy_b32
triton_small_n
cholesky_probe
```

Runtime shape:

```python
def custom_kernel(A):
    batch, n, _ = A.shape
    features = classify_features(A)

    if n <= 64:
        return small_n_kernel(A)

    if n == 512:
        easy_mask, hard_mask = classify_512(A, features)
        return dispatch_masked(A, easy_mask, hard_mask)

    if n == 1024:
        return robust_1024_kernel(A)

    if features.triangular:
        return triangular_kernel(A)

    return generic_blocked_or_fallback(A)
```

The classifier cannot use official case labels like `rankdef` or `mixed`; it only
sees numeric input features:

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

Autotuning should answer questions like:

```text
n=512, case=dense,      kernel=fast_512,   block=64: pass? time?
n=512, case=rankdef,    kernel=fast_512,   block=64: pass? time?
n=512, case=rankdef,    kernel=robust_512, block=32: pass? time?
n=1024, case=nearrank,  kernel=robust_1024, block=32: pass? time?
n=2048, case=mixed,     kernel=large_blocked, block=64: pass? time?
```

The output should become a dispatch table:

```python
DISPATCH = {
    32: {"kernel": "small_n", "block": 32},
    176: {"kernel": "blocked_wy", "block": 16},
    352: {"kernel": "blocked_wy", "block": 32},
    512: {"easy": "fast_512_b64", "hard": "robust_512_b32"},
    1024: {"kernel": "robust_1024_b32"},
    2048: {"kernel": "large_blocked_b64"},
    4096: {"kernel": "fallback"},
}
```

For mixed batches, the classifier may produce masks:

```python
easy_mask = classify_easy(A, features)
hard_mask = ~easy_mask
```

Then compare two strategies empirically:

1. Split dispatch: run fast and robust kernels on subsets.
2. Robust whole-batch dispatch: avoid mask/scatter and extra launch overhead.

Split dispatch is only useful if the fast path saves more time than the subset
selection and extra launches cost.

Cholesky should remain a probe until proven otherwise. It is useful for measuring
the potential speed of normal-equation-style QR on easy dense inputs, but it is
not a core kernel until we can cheaply produce valid compact Householder `(H,
tau)` output and avoid hard-case failures.

## Case Notes

- `dense`: best candidate for fast blocked Householder.
- `upper`: can use `H = triu(A), tau = 0` only if lower-triangular norm is tiny.
- `rankdef`: must handle zero reflector norms and no-op reflectors.
- `nearrank`: avoid approximate or low-precision updates.
- `clustered`: global scale checks may miss later hard panels.
- `rowscale`: column-only checks can miss row dynamic range.
- `nearcollinear`: orthogonality can fail before reconstruction looks terrible.
- `mixed`: never classify the whole batch from a sample.

## Cholesky QR

The rules do not explicitly ban Cholesky QR, but the output must still be a valid
compact Householder `(H, tau)` representation. Cholesky QR is unsafe as a general
path for rank-deficient and near-rank-deficient cases, and returning Cholesky
factors directly does not satisfy the output contract.

If used at all, Cholesky-style logic should be an experimental fast path for
extremely easy matrices, with a conservative classifier and a way to produce
valid compact Householder output.

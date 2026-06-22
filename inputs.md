# QR_2 Input Edge Cases and Dispatch Notes

The dispatch problem is not just about `n`. The benchmark contains matrices with very different numerical and structural behavior inside the same shape bucket, especially in the mixed batches. A kernel that is fast for dense well-conditioned inputs can fail on rank-deficient, near-rank-deficient, clustered-scale, or near-collinear cases.

The safe default assumption is: each matrix in the batch must stand on its own. Do not infer that the whole batch is easy because a few matrices look easy.

## Hard Constraints

The implementation must return compact Householder factors compatible with `torch.geqrf`.

- `H`: FP32, shape `batch x n x n`
- `tau`: FP32, shape `batch x n`
- `triu(H)` is treated as `R`
- `Q = torch.linalg.householder_product(H, tau)`
- Correctness is checked against the original FP32 input
- Residual and orthogonality are measured tightly enough that low precision output is not viable

Internal FP16, FP8, NVFP4, TF32, or mixed-precision tricks are only useful if the final compact Householder representation still passes FP32-style QR checks.

## Benchmark Shape Buckets

Known shapes:

```text
batch=20,  n=32,   cond=1
batch=40,  n=176,  cond=1
batch=40,  n=352,  cond=1
batch=640, n=512,  cond=2
batch=60,  n=1024, cond=2
batch=8,   n=2048, cond=1
batch=2,   n=4096, cond=1

batch=640, n=512,  case=mixed,     cond=2
batch=60,  n=1024, case=mixed,     cond=2
batch=640, n=512,  case=rankdef,   cond=0
batch=640, n=512,  case=clustered, cond=0
batch=60,  n=1024, case=nearrank,  cond=0
```

Shape-based dispatch is useful, but not sufficient. `n=512` and `n=1024` have both easy dense cases and hard structured cases.

## Dispatch-Relevant Input Properties

### Size

Size controls the basic kernel family.

- `n <= 32`: overhead dominates; a simple one-block or small-batch path may win.
- `n = 176` / `352`: medium-size kernels need good occupancy and limited synchronization overhead.
- `n = 512`: most important throughput target; many matrices per batch.
- `n = 1024`: fewer matrices, more work per matrix, harder numerical cases.
- `n >= 2048`: global memory traffic and trailing updates dominate.
- `n = 4096`: batch is tiny; fallback or heavily blocked path may be acceptable if faster paths are risky.

Practical dispatch:

```text
if n <= 64:
    small_n_kernel
elif n <= 352:
    medium_panel_kernel
elif n == 512:
    512_specialized_kernel
elif n == 1024:
    1024_robust_kernel
else:
    large_n_blocked_or_fallback
```

### Batch Size

Batch size changes the tradeoff between per-matrix specialization and launch overhead.

- Large batches, like `batch=640`, reward one-kernel-per-stage designs with high occupancy.
- Small batches, like `batch=2` or `8`, cannot hide inefficient memory traffic as easily.
- For small batches and large `n`, using library-backed operations or fallback paths may be competitive.

Avoid dispatch that assumes all matrices in a large batch share one condition profile. The mixed case violates that assumption.

### Column Scaling / Dynamic Range

Dense conditioned cases multiply columns by a logspace scale. This creates columns with very different norms.

Risks:

- Tiny trailing columns can underflow relative to the leading columns.
- Reflector norm computation can lose accuracy.
- Mixed precision can produce reflectors that look reasonable but fail residual checks.
- TF32 can be too inaccurate for the QR gates.

Useful cheap signal:

```text
column_norm_max / column_norm_min
```

If this ratio is large, route to a robust FP32 path. Be careful with zero or near-zero column norms.

### Rank Deficiency

Rank-deficient inputs have dependent or zero-information directions.

Risks:

- Cholesky-style QR is invalid or unstable.
- Normal-equation methods amplify conditioning problems.
- Reflector generation must handle near-zero vector norms.
- `tau` may need to become zero for a no-op reflector.

Useful cheap signals:

```text
very small column norm
very small panel norm
very small diagonal candidate after previous reflectors
```

Do not use these signals to skip correctness. Use them to select a safer Householder path.

### Near Rank Deficiency

Near-rank-deficient inputs are worse than clean rank-deficient inputs because failures may be subtle.

Risks:

- Fast approximate paths may pass most matrices but fail a few.
- Low-bit or TF32 internal work may produce lower-triangular leakage in `Q.T @ A`.
- Iterative correction may not recover enough accuracy within the time budget.

Dispatch rule: if a panel has a large norm ratio or a suspiciously small new diagonal, prefer the robust path.

### Clustered Scale

Clustered-scale inputs have groups of columns or rows with similar scale, separated from other groups by large gaps.

Risks:

- Per-panel heuristics can misclassify the matrix as easy if the current panel is locally well-scaled.
- Later panels can become much harder after applying previous reflectors.
- A single global scale estimate may not represent every panel.

Better signal: compute cheap scale information per panel, not just once per matrix.

### Near-Collinear Columns

Near-collinear columns can make local reflector updates numerically fragile.

Risks:

- Orthogonality can fail before the reconstruction residual looks terrible.
- Approximate trailing updates accumulate error.
- Low precision dot products are especially risky.

Signal:

```text
abs(dot(col_i, col_j)) / (norm(col_i) * norm(col_j))
```

This is expensive to check globally, but local panel Gram estimates can detect obvious cases.

### Upper-Triangular Inputs

Upper-triangular inputs are already close to `R`.

Potential optimization:

- Detect tiny lower triangle and use a faster path.

Risk:

- The checker still expects a valid compact Householder representation.
- Returning the input with `tau = 0` is only correct if the lower triangle is actually negligible and `Q = I` is valid.

Safe rule: only use a triangular fast path when lower-triangular norm is extremely small relative to matrix norm.

### Banded Inputs

Banded inputs have localized nonzeros.

Potential optimization:

- Avoid updating known-zero regions.

Risk:

- Householder transformations fill in trailing regions.
- The compact `H` output still needs the correct reflectors.

Unless the banded structure is guaranteed and preserved by the algorithm, treat this as a dense QR case.

### Row-Scaled Inputs

Row scaling changes which reflectors are numerically sensitive.

Risks:

- Norm computation can be dominated by a few rows.
- Panel-level shared-memory reductions need stable accumulation.
- Per-column scale checks may miss row-scale pathologies.

Useful signal:

```text
row_norm_max / row_norm_min
```

If large, route to FP32 accumulation and robust normalization.

## Safe Dispatch Signals

Cheap signals worth computing:

- `n`
- `batch`
- matrix norm
- per-column norm min/max
- per-row norm min/max
- lower-triangular norm
- panel norm min/max
- zero or near-zero diagonal candidates during factorization
- whether `n` is one of the benchmark-specialized sizes

These signals are robust enough to choose between kernel families.

## Risky Dispatch Signals

Avoid relying on:

- A few sampled matrices to classify the whole batch
- A single global condition estimate for the whole batch
- Exact condition number estimation, which is too expensive
- Assuming `cond=1` or `cond=2` means easy
- Assuming mixed batches have a majority behavior that can be applied to every matrix
- Choosing Cholesky QR based on apparent positive definiteness
- Choosing low precision based only on column norms

The benchmark explicitly includes mixed batches, so routing must be per matrix or per panel when numerical difficulty matters.

## Candidate Kernel Families

### Small-N Kernel

For `n=32`.

Goal:

- Minimize overhead.
- Keep the whole matrix or most of it resident if possible.
- Avoid complex blocked machinery.

Likely dispatch:

```text
n <= 64
```

### Medium Blocked Householder Kernel

For `n=176` and `n=352`.

Goal:

- Use compact Householder.
- Keep panels in shared memory.
- Use efficient trailing updates without spilling.

Likely dispatch:

```text
64 < n <= 352
```

### Specialized 512 Kernel

For `batch=640, n=512`.

Goal:

- Optimize the main leaderboard shape.
- Use blocked Householder with GEMM-like trailing updates.
- Tune block size and warp count aggressively.

Dispatch complication:

- There are dense, mixed, rank-deficient, and clustered `n=512` cases.
- The fast path must either be robust enough for all of them or have a cheap per-matrix safety check.

Likely dispatch:

```text
if n == 512 and matrix_or_panel_is_easy:
    fast_512_kernel
else:
    robust_512_kernel
```

### Robust 1024 Kernel

For `batch=60, n=1024`.

Goal:

- Prioritize correctness under mixed and near-rank-deficient cases.
- Avoid low precision unless it is only used in places that cannot affect final QR accuracy too much.

Likely dispatch:

```text
n == 1024
```

with internal panel-level routing between fast and robust updates.

### Large-N Fallback or Blocked Path

For `n=2048` and `n=4096`.

Goal:

- Avoid spending too much contest time on rare large cases.
- Use a conservative path if custom kernels are not clearly faster.

Likely dispatch:

```text
n >= 2048
```

Fallback may be acceptable for `n=4096` because the batch is only `2`, but it still affects geometric mean runtime.

## Practical Dispatch Sketch

```python
def choose_kernel(A):
    batch, n, _ = A.shape

    if n <= 64:
        return "small_n"

    if n in (176, 352):
        return "medium_blocked"

    if n == 512:
        # Use per-matrix or per-panel checks.
        # Do not classify the whole batch from one sample.
        if looks_numerically_easy(A):
            return "fast_512"
        return "robust_512"

    if n == 1024:
        return "robust_1024"

    if n >= 2048:
        return "large_blocked_or_fallback"

    return "generic_blocked"
```

For mixed batches, the dispatch may need to return a mask instead of a single kernel:

```python
easy_mask = classify_each_matrix(A)
run_fast_kernel(A[easy_mask])
run_robust_kernel(A[~easy_mask])
```

This has overhead, so it is only worth doing if the fast kernel is much faster and the classification is cheap. Otherwise, make the default kernel robust enough for the entire benchmark bucket.

## Simplest Safe Strategy

The safest contest strategy is:

1. Dispatch primarily by `n`.
2. Use blocked Householder everywhere.
3. Specialize memory layout, block size, and warp count per shape.
4. Avoid Cholesky QR, TF32, FP8, and NVFP4 unless fully validated.
5. Add robust handling for zero and near-zero reflector norms.
6. Use per-panel scale checks if adding fast paths.

This is less flashy than structure-specific routing, but it matches the checker: correctness is a hard gate, and bad numerical routing loses the submission entirely.


# Empirical QR Sweeps

Use `autotune.sweep` to answer dispatch questions with measurements instead
of guesses. It sweeps input dimensions, generated input cases, panel/block sizes,
and algorithm variants, then writes CSV.

Default smoke sweep. The target hardware profile is B200 even when the actual
device is CPU:

```bash
uv run --group practice python -m autotune.sweep --hardware b200
```

Focused block-size sweep:

```bash
uv run --group practice python -m autotune.sweep \
  --hardware b200 \
  --n 32,64,128 \
  --cases dense,rankdef,clustered,mixed \
  --variants python_geqrf,python_blocked \
  --block-sizes 4,8,16,32,64 \
  --output results/block_sweep.csv
```

CUDA/B200-style run for larger shapes:

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

CSV columns include:

- shape: `batch`, `n`, `case`, `cond`, `seed`
- hardware target: `hardware`, `target_gpu`, `target_arch`,
  `target_memory_gb`, `target_bandwidth_tb_s`, target peak TFLOP/s fields, and
  ridge-point fields
- algorithm: `variant`, `language`, `panel_type`, `block_size`
- correctness: `passed`, `message`
- timing: `mean_ms`, `best_ms`, `std_ms`, `worst_ms`
- input properties: `col_ratio_max`, `row_ratio_max`, `lower_ratio_max`,
  `min_col_norm`, `matrix_norm_max`

Legacy aliases like `geqrf`, `blocked`, `unblocked`, `triangular`, and
`cholesky` still work, but new sweeps should use explicit names such as
`python_geqrf`, `python_blocked`, `python_triangular`, and `cholesky_probe`.

`cholesky_probe` is reported as a separate QR probe. It is useful for speed and
stability measurements, but it is not a valid QR v2 submission output because it
does not return compact Householder `(H, tau)`.

The empirical questions to answer from the CSV:

- For each `n`, which passing variant is fastest?
- For each `n`, which block size wins for dense inputs?
- Does that block size still pass hard cases?
- Is the measured path plausibly memory-bound or compute-bound under the B200
  roofline?
- Which input-property signals correlate with failures or slow paths?
- Does Cholesky fail exactly where expected?
- Is any fast path enough faster to justify split dispatch overhead?

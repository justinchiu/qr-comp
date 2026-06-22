# Nsight Compute Profiling

This Mac can edit and smoke-test the code, but `ncu` requires an NVIDIA GPU and
Nsight Compute on the machine that runs the workload.

## List Cases

```bash
uv run --group practice python local_benchmark.py --suite official --list-cases
```

The official benchmark case indices match `qr_v2/task.yml`; for example, case
`3` is the main `batch=640, n=512` dense benchmark.

## Run NCU

On a CUDA machine:

```bash
uv sync --group practice
QR_CASE_INDEX=3 ./scripts/ncu_qr.sh
```

Useful overrides:

```bash
QR_CASE_INDEX=7 NCU_SET=roofline ./scripts/ncu_qr.sh
QR_CASE_INDEX=3 NCU_KERNEL_NAME='.*my_qr.*' ./scripts/ncu_qr.sh
QR_CASE_INDEX=3 NCU_LAUNCH_SKIP=10 NCU_LAUNCH_COUNT=1 ./scripts/ncu_qr.sh
```

Reports are written under `profiles/` and ignored by git.

The script profiles `local_benchmark.py` with one warmup and one measured repeat,
with correctness recheck disabled during the profiled loop. Run correctness first:

```bash
uv run --group practice python local_benchmark.py --suite official --mode test
```

Popcorn `qr_v2` currently does not provide a profile mode, so this is for a
separate CUDA box where you can run the repo directly.

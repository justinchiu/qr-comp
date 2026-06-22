# QR v2 Submission

This repo is set up as a lightweight `uv` project for editing and linting QR submissions.
The actual competition runtime is provided by Popcorn.

## Local Setup

```bash
uv sync
uv run ruff check .
```

Torch is available as an optional practice dependency for local CPU experiments:

```bash
uv sync --group practice
uv run --group practice pytest
uv run --group practice python local_benchmark.py --hardware b200 --suite smoke
```

Popcorn still injects the official `task.py` and CUDA runtime dependencies when it
evaluates `submission.py`.

The local benchmark harness has a small smoke suite by default. On a CUDA machine,
you can run the official QR v2 shapes locally:

```bash
uv run --group practice python local_benchmark.py --hardware b200 --suite official --mode test
uv run --group practice python local_benchmark.py --hardware b200 --suite official --mode benchmark
```

The real leaderboard timing is still Popcorn on B200.

## Profile With Nsight Compute

`ncu` requires an NVIDIA GPU machine. See [PROFILE.md](PROFILE.md):

```bash
uv run --group practice python local_benchmark.py --hardware b200 --suite official --list-cases
./scripts/profile_geqrf_baseline.sh
QR_HARDWARE=b200 QR_CASE_INDEX=3 ./scripts/ncu_qr.sh
```

## Submit

Install and register Popcorn once:

```bash
curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
popcorn register discord
```

Run the QR v2 checks:

```bash
popcorn submit --leaderboard qr_v2 --gpu B200 --mode test submission.py
popcorn submit --leaderboard qr_v2 --gpu B200 --mode benchmark submission.py
popcorn submit --leaderboard qr_v2 --gpu B200 --mode leaderboard submission.py
```

The current `submission.py` is the official correctness baseline:

```python
def custom_kernel(data):
    return torch.geqrf(data)
```

Replace that implementation with optimized kernels while preserving the return contract:
`(H, tau)`, where `H` has shape `batch x n x n` and `tau` has shape `batch x n`.

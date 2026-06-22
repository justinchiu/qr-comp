# QR_2 Kernel Notes

Notes for reasoning about GPU MODE QR_2 kernels and input-aware dispatch.

## Contents

- [inputs.md](inputs.md): edge cases, benchmark shape buckets, and dispatch notes for compact Householder QR.
- [design.md](design.md): panel/blocking explanation and dispatch strategy.
- [kernel_development_loop.md](kernel_development_loop.md): workflow for developing, tuning, profiling, and promoting kernels.
- [SUBMIT.md](SUBMIT.md): `uv` setup and Popcorn submission commands for `qr_v2`.
- [PROFILE.md](PROFILE.md): Nsight Compute workflow for CUDA machines.
- [EXPERIMENTS.md](EXPERIMENTS.md): empirical sweep workflow for dispatch and block-size choices.
- [local_eval.py](local_eval.py): local copy of the official `qr_v2` input generator and checker.
- [qr_practice/](qr_practice): local PyTorch practice implementations for compact Householder QR.

## Context

The challenge is to implement batched square compact-Householder QR factorization compatible with `torch.geqrf`.

The main implementation problem is not only choosing a fast QR kernel by shape, but deciding when input properties require a more robust path. Mixed batches, rank-deficient cases, near-rank-deficient cases, clustered scales, and row or column scaling can make naive fast paths fail correctness.

## Local Practice

```bash
uv sync --group practice
uv run --group practice pytest
uv run --group practice python local_benchmark.py --suite smoke
```

The practice code includes an unblocked compact Householder QR and a blocked compact-WY trailing update implementation for CPU-side experimentation.

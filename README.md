# QR_2 Kernel Notes

Notes for reasoning about GPU MODE QR_2 kernels and input-aware dispatch.

## Contents

- [inputs.md](inputs.md): edge cases, benchmark shape buckets, and dispatch notes for compact Householder QR.
- [design.md](design.md): panel/blocking explanation and dispatch strategy.
- [B200.md](B200.md): B200-specific hardware assumptions and tuning guidance.
- [REMOTE_RUNBOOK.md](REMOTE_RUNBOOK.md): Prime Intellect and AWS Spot GPU rental plus A100/H100/B200 sweep runbooks.
- [SWARM.md](SWARM.md): simple file-based multi-agent coordination for remote GPU jobs.
- [kernel_development_loop.md](kernel_development_loop.md): workflow for developing, tuning, profiling, and promoting kernels.
- [AGENTS.md](AGENTS.md): contributor/agent guide for the competition workflow and constraints.
- [SUBMIT.md](SUBMIT.md): `uv` setup and Popcorn submission commands for `qr_v2`.
- [PROFILE.md](PROFILE.md): Nsight Compute workflow for CUDA machines.
- [baselines/](baselines): stable baseline modules, including `torch.geqrf`, for profiling.
- [EXPERIMENTS.md](EXPERIMENTS.md): empirical sweep workflow for dispatch and block-size choices.
- [local_eval.py](local_eval.py): local copy of the official `qr_v2` input generator and checker.
- [kernels/](kernels): Python kernel variants plus CUDA/Triton templates.
- [qr_practice/](qr_practice): local PyTorch practice implementations for compact Householder QR.

## Context

The challenge is to implement batched square compact-Householder QR factorization compatible with `torch.geqrf`.

The main implementation problem is not only choosing a fast QR kernel by shape, but deciding when input properties require a more robust path. Mixed batches, rank-deficient cases, near-rank-deficient cases, clustered scales, and row or column scaling can make naive fast paths fail correctness.

## Local Practice

```bash
uv sync --group practice
uv run --group practice pytest
uv run --group practice python local_benchmark.py --hardware b200 --suite smoke
```

The practice code includes an unblocked compact Householder QR and a blocked compact-WY trailing update implementation for CPU-side experimentation.

## Learn

An interactive notebook walks through Householder QR from scratch — the geometry
of reflections, the compact `(H, tau)` `geqrf` contract, reconstructing `Q`, and
the compact-WY blocking that the GPU kernels rely on. It runs against the
reference code in `qr_practice/householder.py`.

```bash
uv sync --group learn
uv run --group learn jupyter lab learn_householder_qr.ipynb
```

# QR_2 Kernel Notes

Notes for reasoning about GPU MODE QR_2 kernels and input-aware dispatch.

## Contents

- [inputs.md](inputs.md): edge cases, benchmark shape buckets, and dispatch notes for compact Householder QR.

## Context

The challenge is to implement batched square compact-Householder QR factorization compatible with `torch.geqrf`.

The main implementation problem is not only choosing a fast QR kernel by shape, but deciding when input properties require a more robust path. Mixed batches, rank-deficient cases, near-rank-deficient cases, clustered scales, and row or column scaling can make naive fast paths fail correctness.


// CUDA development template for QR v2 compact Householder kernels.
//
// This file is intentionally not compiled by the default local test suite. It
// sketches the expected shape of a CUDA implementation that eventually returns
// H and tau in the torch.geqrf-compatible compact Householder format.

#include <cuda_runtime.h>

extern "C" __global__ void qr_small_n32_kernel(
    const float* __restrict__ a,
    float* __restrict__ h,
    float* __restrict__ tau,
    int batch_stride
) {
    // TODO:
    // - one CTA per matrix for n=32
    // - compute Householder reflectors in FP32
    // - store R in triu(H), reflector tails below diagonal, tau per column
    // - handle zero-norm reflectors with tau=0
    const int batch = blockIdx.x;
    const int tid = threadIdx.x;
    (void)a;
    (void)h;
    (void)tau;
    (void)batch_stride;
    (void)batch;
    (void)tid;
}

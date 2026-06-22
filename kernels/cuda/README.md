# CUDA Kernel Templates

This directory is for CUDA C++ kernels that can later be loaded from Python with
`torch.utils.cpp_extension` during local development, then inlined or packed into
`submission.py` if they become part of the final entry.

Files here are templates until they are wired into a build path and validated on
a CUDA machine. This Mac cannot compile or run them.

Development target:

```text
Python wrapper -> CUDA extension -> compact Householder (H, tau)
```

Do not promote a CUDA kernel into `submission.py` until it passes
`local_eval.check_implementation` on official QR v2 cases.

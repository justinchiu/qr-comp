from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import time
from collections.abc import Callable
from pathlib import Path

import torch

import local_eval
from qr_practice import blocked_compact_householder_qr, compact_householder_qr

CompactKernel = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclasses.dataclass(frozen=True)
class SweepSpec:
    batch: int
    n: int
    cond: int
    seed: int
    case: str


@dataclasses.dataclass(frozen=True)
class Variant:
    name: str
    panel_type: str
    block_size: int | None


@dataclasses.dataclass
class Timing:
    runs: int
    mean_ms: float
    std_ms: float
    best_ms: float
    worst_ms: float


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_call(fn: Callable[[], object]) -> float:
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        _synchronize()
        start.record()
        fn()
        end.record()
        _synchronize()
        return float(start.elapsed_time(end))

    start_ns = time.perf_counter_ns()
    fn()
    end_ns = time.perf_counter_ns()
    return (end_ns - start_ns) / 1e6


def _stats(samples_ms: list[float]) -> Timing:
    mean = sum(samples_ms) / len(samples_ms)
    variance = sum((sample - mean) ** 2 for sample in samples_ms)
    std = math.sqrt(variance / (len(samples_ms) - 1)) if len(samples_ms) > 1 else 0.0
    return Timing(
        runs=len(samples_ms),
        mean_ms=mean,
        std_ms=std,
        best_ms=min(samples_ms),
        worst_ms=max(samples_ms),
    )


def _compact_kernel_for(variant: Variant) -> CompactKernel:
    if variant.name == "geqrf":
        return torch.geqrf
    if variant.name == "unblocked":
        return compact_householder_qr
    if variant.name == "blocked":
        if variant.block_size is None:
            raise ValueError("blocked variant needs a block size")

        def kernel(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return blocked_compact_householder_qr(data, block_size=variant.block_size)

        return kernel
    raise ValueError(f"unknown compact variant: {variant.name}")


def _input_properties(data: torch.Tensor) -> dict[str, float]:
    eps = torch.finfo(data.dtype).eps
    col_norms = torch.linalg.vector_norm(data, dim=1)
    row_norms = torch.linalg.vector_norm(data, dim=2)
    mat_norm = torch.linalg.matrix_norm(data, ord=1, dim=(-2, -1))
    lower_norm = torch.linalg.matrix_norm(torch.tril(data, diagonal=-1), ord=1, dim=(-2, -1))

    col_min = col_norms.amin(dim=1)
    col_max = col_norms.amax(dim=1)
    row_min = row_norms.amin(dim=1)
    row_max = row_norms.amax(dim=1)
    col_ratio = col_max / col_min.clamp_min(eps)
    row_ratio = row_max / row_min.clamp_min(eps)
    lower_ratio = lower_norm / mat_norm.clamp_min(eps)

    return {
        "col_ratio_max": float(col_ratio.amax().item()),
        "row_ratio_max": float(row_ratio.amax().item()),
        "lower_ratio_max": float(lower_ratio.amax().item()),
        "min_col_norm": float(col_min.amin().item()),
        "matrix_norm_max": float(mat_norm.amax().item()),
    }


def _check_compact(
    data: torch.Tensor,
    output: tuple[torch.Tensor, torch.Tensor],
) -> tuple[bool, str]:
    good, message = local_eval.check_implementation(data, output)
    return good, message


def _run_compact_variant(
    data: torch.Tensor,
    variant: Variant,
    warmups: int,
    repeats: int,
) -> tuple[bool, str, Timing]:
    kernel = _compact_kernel_for(variant)
    reference = data.clone()

    output = kernel(data.clone())
    good, message = _check_compact(reference, output)
    if not good:
        return False, message, Timing(0, math.nan, math.nan, math.nan, math.nan)

    for _ in range(warmups):
        kernel(data.clone())

    samples = [_time_call(lambda: kernel(data.clone())) for _ in range(repeats)]
    return True, message, _stats(samples)


def _cholesky_qr(
    data: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    gram = data.transpose(-1, -2) @ data
    chol, info = torch.linalg.cholesky_ex(gram)
    if bool((info != 0).any().item()):
        return None, None, info

    r = chol.transpose(-1, -2)
    qt = torch.linalg.solve_triangular(
        r.transpose(-1, -2),
        data.transpose(-1, -2),
        upper=False,
    )
    q = qt.transpose(-1, -2)
    return q, r, info


def _check_cholesky_qr(data: torch.Tensor) -> tuple[bool, str]:
    q, r, info = _cholesky_qr(data)
    if q is None or r is None:
        worst = int(info.argmax().item())
        return False, f"cholesky_ex failed: matrix={worst}, info={int(info[worst].item())}"

    batch, n, _ = data.shape
    eps = torch.finfo(torch.float32).eps
    a64 = data.double()
    q64 = q.double()
    r64 = r.double()
    scale = torch.linalg.matrix_norm(a64, ord=1, dim=(-2, -1)).clamp_min(1e-30)

    factor = torch.linalg.matrix_norm(r64 - q64.transpose(-1, -2) @ a64, ord=1, dim=(-2, -1))
    recon = torch.linalg.matrix_norm(q64 @ r64 - a64, ord=1, dim=(-2, -1))
    eye = torch.eye(n, device=data.device, dtype=torch.float64).expand(batch, n, n)
    orth = torch.linalg.matrix_norm(q64.transpose(-1, -2) @ q64 - eye, ord=1, dim=(-2, -1))

    factor_ok = factor <= 20 * n * eps * scale
    recon_ok = recon <= 20 * n * eps * scale
    orth_ok = orth <= 100 * n * eps
    if bool((factor_ok & recon_ok & orth_ok).all().item()):
        return True, (
            f"factor={factor.amax().item():.3g}; recon={recon.amax().item():.3g}; "
            f"orth={orth.amax().item():.3g}"
        )

    return False, (
        f"factor={factor.amax().item():.3g}; recon={recon.amax().item():.3g}; "
        f"orth={orth.amax().item():.3g}"
    )


def _run_cholesky_variant(
    data: torch.Tensor,
    warmups: int,
    repeats: int,
) -> tuple[bool, str, Timing]:
    good, message = _check_cholesky_qr(data)
    if not good:
        return False, message, Timing(0, math.nan, math.nan, math.nan, math.nan)

    for _ in range(warmups):
        _cholesky_qr(data)

    samples = [_time_call(lambda: _cholesky_qr(data)) for _ in range(repeats)]
    return True, message, _stats(samples)


def _make_specs(
    ns: list[int],
    cases: list[str],
    batch: int,
    cond: int,
    seed: int,
) -> list[SweepSpec]:
    specs = []
    for n in ns:
        for offset, case in enumerate(cases):
            specs.append(
                SweepSpec(
                    batch=batch,
                    n=n,
                    cond=cond,
                    seed=seed + 1000 * n + offset,
                    case=case,
                )
            )
    return specs


def _make_variants(names: list[str], block_sizes: list[int]) -> list[Variant]:
    variants = []
    for name in names:
        if name == "blocked":
            variants.extend(
                Variant(name="blocked", panel_type="compact_wy", block_size=block_size)
                for block_size in block_sizes
            )
        elif name == "unblocked":
            variants.append(Variant(name="unblocked", panel_type="scalar", block_size=None))
        elif name == "geqrf":
            variants.append(Variant(name="geqrf", panel_type="library", block_size=None))
        elif name == "cholesky":
            variants.append(
                Variant(name="cholesky", panel_type="normal_equations", block_size=None)
            )
        else:
            raise ValueError(f"unknown variant: {name}")
    return variants


def _row(
    spec: SweepSpec,
    variant: Variant,
    props: dict[str, float],
    passed: bool,
    message: str,
    timing: Timing,
) -> dict[str, object]:
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "batch": spec.batch,
        "n": spec.n,
        "case": spec.case,
        "cond": spec.cond,
        "seed": spec.seed,
        "variant": variant.name,
        "panel_type": variant.panel_type,
        "block_size": "" if variant.block_size is None else variant.block_size,
        "passed": passed,
        "message": message,
        "runs": timing.runs,
        "mean_ms": timing.mean_ms,
        "std_ms": timing.std_ms,
        "best_ms": timing.best_ms,
        "worst_ms": timing.worst_ms,
        **props,
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Empirical QR dispatch sweep.")
    parser.add_argument("--n", default="16,32,64")
    parser.add_argument("--cases", default="dense,upper,rankdef,clustered,mixed")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--cond", type=int, default=2)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--variants", default="geqrf,unblocked,blocked,cholesky")
    parser.add_argument("--block-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="results/qr_sweep.csv")
    args = parser.parse_args()

    ns = _parse_csv_ints(args.n)
    cases = _parse_csv_strings(args.cases)
    names = _parse_csv_strings(args.variants)
    block_sizes = _parse_csv_ints(args.block_sizes)
    specs = _make_specs(ns, cases, batch=args.batch, cond=args.cond, seed=args.seed)
    variants = _make_variants(names, block_sizes)

    rows: list[dict[str, object]] = []
    print(
        f"device={'cuda' if torch.cuda.is_available() else 'cpu'} torch={torch.__version__} "
        f"specs={len(specs)} variants={len(variants)}"
    )

    for spec in specs:
        data = local_eval.generate_input(
            batch=spec.batch,
            n=spec.n,
            cond=spec.cond,
            seed=spec.seed,
            case=spec.case,
        )
        props = _input_properties(data)
        for variant in variants:
            if variant.name == "cholesky":
                passed, message, timing = _run_cholesky_variant(data, args.warmups, args.repeats)
            else:
                passed, message, timing = _run_compact_variant(
                    data, variant, args.warmups, args.repeats
                )

            rows.append(_row(spec, variant, props, passed, message, timing))
            status = "PASS" if passed else "FAIL"
            block = "" if variant.block_size is None else f" b={variant.block_size}"
            print(
                f"{status} n={spec.n} case={spec.case} variant={variant.name}{block} "
                f"mean={timing.mean_ms:.3f}ms"
            )

    output = Path(args.output)
    _write_rows(output, rows)
    print(f"wrote {output}")
    return 0 if all(bool(row["passed"]) for row in rows if row["variant"] != "cholesky") else 1


if __name__ == "__main__":
    raise SystemExit(main())

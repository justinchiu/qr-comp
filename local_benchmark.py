from __future__ import annotations

import argparse
import dataclasses
import importlib
import math
import time
from collections.abc import Callable
from typing import Any

import torch

import local_eval

Spec = dict[str, int | str]
Kernel = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

SMOKE_BENCHMARKS: list[Spec] = [
    {"batch": 4, "n": 32, "cond": 1, "seed": 43214},
    {"batch": 2, "n": 64, "cond": 2, "seed": 770001, "case": "mixed"},
    {"batch": 2, "n": 64, "cond": 0, "seed": 770003, "case": "rankdef"},
    {"batch": 2, "n": 64, "cond": 0, "seed": 770004, "case": "clustered"},
]

OFFICIAL_TESTS: list[Spec] = [
    {"batch": 20, "n": 32, "cond": 1, "seed": 53124},
    {"batch": 40, "n": 176, "cond": 1, "seed": 3321},
    {"batch": 40, "n": 352, "cond": 1, "seed": 1200},
    {"batch": 16, "n": 512, "cond": 2, "seed": 32523},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 4327},
    {"batch": 1, "n": 4096, "cond": 1, "seed": 75342},
    {"batch": 16, "n": 512, "cond": 4, "seed": 32524, "case": "dense"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32525, "case": "rankdef"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32526, "case": "clustered"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32527, "case": "band"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32528, "case": "rowscale"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32529, "case": "nearcollinear"},
    {"batch": 4, "n": 1024, "cond": 4, "seed": 4328, "case": "dense"},
    {"batch": 4, "n": 1024, "cond": 0, "seed": 4329, "case": "rankdef"},
    {"batch": 4, "n": 1024, "cond": 0, "seed": 4330, "case": "nearrank"},
    {"batch": 4, "n": 1024, "cond": 0, "seed": 4331, "case": "clustered"},
    {"batch": 2, "n": 2048, "cond": 2, "seed": 224466, "case": "dense"},
    {"batch": 2, "n": 2048, "cond": 0, "seed": 224467, "case": "rankdef"},
    {"batch": 1, "n": 4096, "cond": 0, "seed": 75343, "case": "upper"},
    {"batch": 16, "n": 512, "cond": 2, "seed": 32530, "case": "mixed"},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 4332, "case": "mixed"},
    {"batch": 2, "n": 2048, "cond": 2, "seed": 224468, "case": "mixed"},
]

OFFICIAL_BENCHMARKS: list[Spec] = [
    {"batch": 20, "n": 32, "cond": 1, "seed": 43214},
    {"batch": 40, "n": 176, "cond": 1, "seed": 423011},
    {"batch": 40, "n": 352, "cond": 1, "seed": 123456},
    {"batch": 640, "n": 512, "cond": 2, "seed": 1029},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 75342},
    {"batch": 8, "n": 2048, "cond": 1, "seed": 224466},
    {"batch": 2, "n": 4096, "cond": 1, "seed": 32412},
    {"batch": 640, "n": 512, "cond": 2, "seed": 770001, "case": "mixed"},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 770002, "case": "mixed"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 770003, "case": "rankdef"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 770004, "case": "clustered"},
    {"batch": 60, "n": 1024, "cond": 0, "seed": 770005, "case": "nearrank"},
]


@dataclasses.dataclass
class Stats:
    runs: int
    mean_ms: float
    std_ms: float
    best_ms: float
    worst_ms: float


def _clone_data(data: Any) -> Any:
    if isinstance(data, torch.Tensor):
        return data.clone()
    if isinstance(data, tuple):
        return tuple(_clone_data(item) for item in data)
    if isinstance(data, list):
        return [_clone_data(item) for item in data]
    if isinstance(data, dict):
        return {key: _clone_data(value) for key, value in data.items()}
    return data


def _spec_name(spec: Spec) -> str:
    case = spec.get("case", "dense")
    return (
        f"batch={spec['batch']} n={spec['n']} cond={spec['cond']} "
        f"case={case} seed={spec['seed']}"
    )


def _load_kernel(module_name: str) -> Kernel:
    module = importlib.import_module(module_name)
    kernel = getattr(module, "custom_kernel", None)
    if kernel is None:
        raise ValueError(f"{module_name} does not define custom_kernel")
    return kernel


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_one(kernel: Kernel, data: torch.Tensor) -> float:
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        _synchronize()
        start.record()
        kernel(data)
        end.record()
        _synchronize()
        return float(start.elapsed_time(end))

    start_ns = time.perf_counter_ns()
    kernel(data)
    end_ns = time.perf_counter_ns()
    return (end_ns - start_ns) / 1e6


def _stats(samples_ms: list[float]) -> Stats:
    mean = sum(samples_ms) / len(samples_ms)
    variance = sum((sample - mean) ** 2 for sample in samples_ms)
    std = math.sqrt(variance / (len(samples_ms) - 1)) if len(samples_ms) > 1 else 0.0
    return Stats(
        runs=len(samples_ms),
        mean_ms=mean,
        std_ms=std,
        best_ms=min(samples_ms),
        worst_ms=max(samples_ms),
    )


def _check(kernel: Kernel, spec: Spec) -> str | None:
    data = local_eval.generate_input(**spec)
    output = kernel(_clone_data(data))
    good, message = local_eval.check_implementation(data, output)
    return None if good else message


def _benchmark(kernel: Kernel, spec: Spec, warmups: int, repeats: int, recheck: bool) -> Stats:
    data = local_eval.generate_input(**spec)
    reference = data.clone()

    for _ in range(warmups):
        output = kernel(_clone_data(data))
        good, message = local_eval.check_implementation(reference, output)
        if not good:
            raise RuntimeError(message)

    samples_ms = []
    for _ in range(repeats):
        duration_ms = _time_one(kernel, data)
        samples_ms.append(duration_ms)
        if recheck:
            output = kernel(_clone_data(data))
            good, message = local_eval.check_implementation(reference, output)
            if not good:
                raise RuntimeError(message)

    return _stats(samples_ms)


def _select_specs(mode: str, suite: str) -> list[Spec]:
    if suite == "smoke":
        return SMOKE_BENCHMARKS
    if mode == "test":
        return OFFICIAL_TESTS
    return OFFICIAL_BENCHMARKS


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Local qr_v2 checker and benchmark runner.")
    parser.add_argument("--module", default="submission", help="Module containing custom_kernel.")
    parser.add_argument("--mode", choices=["test", "benchmark"], default="benchmark")
    parser.add_argument("--suite", choices=["smoke", "official"], default="smoke")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--no-recheck", action="store_true")
    args = parser.parse_args()

    kernel = _load_kernel(args.module)
    specs = _select_specs(args.mode, args.suite)
    if args.max_cases is not None:
        specs = specs[: args.max_cases]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} module={args.module}")
    print(f"mode={args.mode} suite={args.suite} cases={len(specs)}")

    if args.mode == "test":
        failed = False
        for idx, spec in enumerate(specs):
            error = _check(kernel, spec)
            status = "FAIL" if error else "PASS"
            print(f"{idx:02d} {status} {_spec_name(spec)}")
            if error:
                print(f"    {error}")
                failed = True
        return 1 if failed else 0

    means = []
    for idx, spec in enumerate(specs):
        try:
            stats = _benchmark(
                kernel,
                spec,
                warmups=args.warmups,
                repeats=args.repeats,
                recheck=not args.no_recheck,
            )
        except RuntimeError as exc:
            print(f"{idx:02d} FAIL {_spec_name(spec)}")
            print(f"    {exc}")
            return 1

        means.append(stats.mean_ms)
        print(
            f"{idx:02d} mean={stats.mean_ms:.3f}ms best={stats.best_ms:.3f}ms "
            f"std={stats.std_ms:.3f}ms runs={stats.runs} {_spec_name(spec)}"
        )

    print(f"geomean={_geomean(means):.3f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

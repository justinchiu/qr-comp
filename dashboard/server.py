#!/usr/bin/env python3
"""Zero-dependency dashboard server for the QR swarm.

Serves the static TS/HTML frontend and a small JSON API that reads the real
project data:

  * results/*.csv        -> benchmark results, speedup vs the geqrf baseline
  * swarm/queue/*         -> file-based job queue (pending/running/done/failed)
  * swarm/agents/*/       -> per-agent status.txt / task.txt / inbox.txt

Run with the project's interpreter so paths resolve from the repo root:

    uv run python dashboard/server.py --port 8787

Then open http://localhost:8787/. No external packages required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_DIR = REPO_ROOT / "results"
SWARM_DIR = REPO_ROOT / "swarm"

BASELINE_VARIANT = "python_geqrf"
QUEUE_STATES = ("pending", "running", "done", "failed")

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


# --------------------------------------------------------------------------- #
# Loose "key: value" parser shared by job and status files (see SWARM.md).
# --------------------------------------------------------------------------- #
def parse_kv(text: str) -> dict:
    """Parse the YAML-ish key:value + simple list format used by swarm files.

    Handles top-level `key: value`, `key:` followed by `  - item` lines, and
    strips inline `# comments`. Good enough for the small swarm files; not a
    full YAML parser.
    """
    data: dict = {}
    current_list_key: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and current_list_key is not None:
            existing = data.get(current_list_key)
            if not isinstance(existing, list):
                existing = []
                data[current_list_key] = existing
            existing.append(_clean(stripped[2:]))
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = _clean(value)
        if value == "":
            # Could be the start of a list or just an empty field.
            current_list_key = key
            data[key] = ""
        else:
            current_list_key = None
            data[key] = value
    # Drop placeholder empty strings that were actually list headers with items.
    return data


def _clean(value: str) -> str:
    value = value.strip()
    # Strip trailing inline comments that are clearly comments (preceded by ws).
    value = re.sub(r"\s+#.*$", "", value).strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1]
    return value


# --------------------------------------------------------------------------- #
# Results: parse CSVs and compute speedup vs the geqrf baseline.
# --------------------------------------------------------------------------- #
def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_results() -> dict:
    """Read every results/*.csv into best-per-config rows plus baseline speedup.

    A "config" is (hardware, batch, n, case). Within a config we keep the best
    (lowest mean_ms) passing run per variant, then attach speedup vs the
    baseline variant's best run for the same config.
    """
    if not RESULTS_DIR.is_dir():
        return {"rows": [], "baseline": BASELINE_VARIANT, "sources": []}

    # best[config][variant] = row dict
    best: dict[tuple, dict[str, dict]] = {}
    sources: list[str] = []

    for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
        sources.append(csv_path.name)
        with csv_path.open(newline="") as fh:
            for raw in csv.DictReader(fh):
                mean = _to_float(raw.get("mean_ms"))
                if mean is None:
                    continue
                hardware = raw.get("hardware", "")
                config = (
                    hardware,
                    raw.get("batch", ""),
                    raw.get("n", ""),
                    raw.get("case", ""),
                )
                variant = raw.get("variant", "")
                row = {
                    "source": csv_path.name,
                    "hardware": hardware,
                    "target_gpu": raw.get("target_gpu", ""),
                    "batch": raw.get("batch", ""),
                    "n": raw.get("n", ""),
                    "case": raw.get("case", ""),
                    "cond": raw.get("cond", ""),
                    "variant": variant,
                    "panel_type": raw.get("panel_type", ""),
                    "block_size": raw.get("block_size", ""),
                    "passed": (raw.get("passed", "") or "").strip().lower() == "true",
                    "mean_ms": mean,
                    "std_ms": _to_float(raw.get("std_ms")),
                    "best_ms": _to_float(raw.get("best_ms")),
                }
                bucket = best.setdefault(config, {})
                prev = bucket.get(variant)
                # Prefer passing rows; among same pass-state keep the fastest.
                if prev is None or _better(row, prev):
                    bucket[variant] = row

    rows: list[dict] = []
    for config, variants in best.items():
        baseline = variants.get(BASELINE_VARIANT)
        baseline_ms = baseline["mean_ms"] if baseline else None
        for variant, row in variants.items():
            row = dict(row)
            if baseline_ms and row["mean_ms"]:
                row["speedup"] = round(baseline_ms / row["mean_ms"], 3)
            else:
                row["speedup"] = None
            row["is_baseline"] = variant == BASELINE_VARIANT
            row["baseline_ms"] = baseline_ms
            rows.append(row)

    rows.sort(key=lambda r: (r["hardware"], int(r["n"] or 0), r["case"], r["variant"]))
    return {"rows": rows, "baseline": BASELINE_VARIANT, "sources": sources}


def _better(row: dict, prev: dict) -> bool:
    if row["passed"] != prev["passed"]:
        return row["passed"]  # passing always beats failing
    return row["mean_ms"] < prev["mean_ms"]


def build_leaderboard(results: dict) -> dict:
    """Rank variants per hardware: wins (fastest in a config) and geomean speedup."""
    by_hw_config: dict[str, dict[tuple, list[dict]]] = {}
    for row in results["rows"]:
        if not row["passed"]:
            continue
        hw = row["hardware"]
        config = (row["batch"], row["n"], row["case"])
        by_hw_config.setdefault(hw, {}).setdefault(config, []).append(row)

    boards: list[dict] = []
    for hw, configs in sorted(by_hw_config.items()):
        stats: dict[str, dict] = {}
        for _config, rows in configs.items():
            winner = min(rows, key=lambda r: r["mean_ms"])
            for r in rows:
                s = stats.setdefault(
                    r["variant"], {"variant": r["variant"], "wins": 0, "configs": 0, "speedups": []}
                )
                s["configs"] += 1
                if r["speedup"]:
                    s["speedups"].append(r["speedup"])
            stats[winner["variant"]]["wins"] += 1

        entries = []
        for s in stats.values():
            speedups = s.pop("speedups")
            s["geomean_speedup"] = round(_geomean(speedups), 3) if speedups else None
            entries.append(s)
        entries.sort(key=lambda e: (e["wins"], e["geomean_speedup"] or 0), reverse=True)
        boards.append({"hardware": hw, "entries": entries})
    return {"boards": boards, "baseline": results["baseline"]}


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


# --------------------------------------------------------------------------- #
# Swarm: queue jobs and agent status.
# --------------------------------------------------------------------------- #
def load_queue() -> dict:
    out: dict[str, list] = {state: [] for state in QUEUE_STATES}
    if not SWARM_DIR.is_dir():
        return out
    for state in QUEUE_STATES:
        state_dir = SWARM_DIR / "queue" / state
        if not state_dir.is_dir():
            continue
        for job_path in sorted(state_dir.glob("*.job")):
            job = parse_kv(job_path.read_text())
            job["state"] = state
            job["file"] = job_path.name
            job.setdefault("job_id", job_path.stem)
            out[state].append(job)
    return out


def load_agents() -> list[dict]:
    agents_dir = SWARM_DIR / "agents"
    if not agents_dir.is_dir():
        return []
    agents: list[dict] = []
    for agent_dir in sorted(p for p in agents_dir.iterdir() if p.is_dir()):
        status_file = agent_dir / "status.txt"
        task_file = agent_dir / "task.txt"
        inbox_file = agent_dir / "inbox.txt"
        agent = {"name": agent_dir.name, "state": "unknown"}
        if status_file.is_file():
            agent.update(parse_kv(status_file.read_text()))
        if task_file.is_file():
            task = parse_kv(task_file.read_text())
            agent["task_summary"] = task.get("task") or task.get("agent") or ""
        if inbox_file.is_file():
            text = inbox_file.read_text()
            agent["inbox_messages"] = text.count("\n---") + (
                1 if text.strip().startswith("---") else 0
            )
        agents.append(agent)
    return agents


# --------------------------------------------------------------------------- #
# HTTP handler.
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401 - quiet by default
        pass

    def do_GET(self):  # noqa: N802 - http.server API
        path = urlparse(self.path).path
        if path == "/api/results":
            return self._json(load_results())
        if path == "/api/leaderboard":
            return self._json(build_leaderboard(load_results()))
        if path == "/api/queue":
            return self._json(load_queue())
        if path == "/api/agents":
            return self._json({"agents": load_agents()})
        if path == "/api/all":
            results = load_results()
            return self._json(
                {
                    "results": results,
                    "leaderboard": build_leaderboard(results),
                    "queue": load_queue(),
                    "agents": load_agents(),
                }
            )
        return self._static(path)

    def _json(self, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path: str):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (STATIC_DIR / rel).resolve()
        if STATIC_DIR not in target.parents or not target.is_file():
            self.send_error(404, "Not found")
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"QR swarm dashboard: http://{args.host}:{args.port}/")
    print(f"  repo root : {REPO_ROOT}")
    print(f"  results   : {RESULTS_DIR}")
    print(f"  swarm     : {SWARM_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()


if __name__ == "__main__":
    main()

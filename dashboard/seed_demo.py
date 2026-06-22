#!/usr/bin/env python3
"""Write sample agents and queue jobs into swarm/ so the dashboard has data.

swarm/ is gitignored runtime state; this just makes the Agents and Queue panels
visible before a real worker is running. Safe to re-run; safe to delete swarm/.

    uv run python dashboard/seed_demo.py
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SWARM = REPO_ROOT / "swarm"


def now(offset_min: int = 0) -> str:
    return (datetime.now(UTC) + timedelta(minutes=offset_min)).strftime("%Y-%m-%dT%H:%M:%SZ")


AGENTS = {
    "coordinator": f"""state: running
task: schedule b200 candidate profiles
last_update: {now(-1)}
pid: 4101
""",
    "remote_operator": f"""state: idle
task: b200 spot pod up, repo synced
last_update: {now(-3)}
pid: 4102
""",
    "cuda_worker": f"""state: running
task: profile geqrf baseline case 7
last_update: {now()}
pid: 4108
cuda_visible_devices: 0
current_job: 002_geqrf_b200_case7
latest_log: logs/002_geqrf_b200_case7.log
""",
    "kernel_engineer_large_n512_1024_dense": f"""state: blocked
task: waiting on b200 profile of candidate blocked kernel
last_update: {now(-14)}
pid: 4120
current_job: 004_candidate_b200_case3
""",
}

JOBS = {
    "pending": {
        "003_sweep_h100_512_1024.job": """job_id: 003_sweep_h100_512_1024
priority: 20
agent: any_cuda_worker
hardware: h100_80gb_sxm
command: python -m autotune.sweep --hardware h100_80gb_sxm --n 512,1024 --batch 16
log: logs/003_sweep_h100_512_1024.log
""",
        "004_candidate_b200_case3.job": """job_id: 004_candidate_b200_case3
priority: 5
agent: any_cuda_worker
hardware: b200
depends_on:
  - 001_geqrf_b200_case3
command: QR_HARDWARE=b200 QR_MODULE=submission QR_CASE_INDEX=3 NCU_SET=roofline ./scripts/ncu_qr.sh
log: logs/004_candidate_b200_case3.log
""",
    },
    "running": {
        "002_geqrf_b200_case7.job": f"""job_id: 002_geqrf_b200_case7
priority: 10
agent: cuda_worker
hardware: b200
command: GEQRF_BASELINE_CASES="7" ./scripts/profile_geqrf_baseline.sh
log: logs/002_geqrf_b200_case7.log
claimed_at: {now(-2)}
""",
    },
    "done": {
        "001_geqrf_b200_case3.job": """job_id: 001_geqrf_b200_case3
state: done
priority: 10
hardware: b200
case: 3
command: QR_HARDWARE=b200 QR_CASE_INDEX=3 NCU_SET=roofline ./scripts/ncu_qr.sh
passed: true
runtime_s: 184
next_action: profile candidate kernel for same case
""",
    },
    "failed": {
        "000_sweep_a100_oom.job": """job_id: 000_sweep_a100_oom
state: failed
priority: 30
hardware: a100_80gb_sxm
case: 11
command: python -m autotune.sweep --hardware a100_80gb_sxm --n 2048 --batch 64
passed: false
runtime_s: 42
next_action: reduce batch and requeue
""",
    },
}


def main() -> None:
    for name, status in AGENTS.items():
        agent_dir = SWARM / "agents" / name
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "status.txt").write_text(status)
    for state, jobs in JOBS.items():
        state_dir = SWARM / "queue" / state
        state_dir.mkdir(parents=True, exist_ok=True)
        for filename, body in jobs.items():
            (state_dir / filename).write_text(body)
    print(f"seeded demo swarm data under {SWARM}")
    print("remove with: rm -rf swarm")


if __name__ == "__main__":
    main()

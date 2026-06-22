# Swarm Coordination

Keep remote GPU coordination simple. The swarm is a set of agents sharing files
in the repo or on the remote machine. No database, broker, service, or scheduler
- just plain-text files and directory moves.

Start lean. Add roles only when the coordination pain is real.

## Model

Each agent has a name, one current task (`task.txt`), one inbox (`inbox.txt`), a
status line (`status.txt`), and optional outputs under ignored directories like
`results/`, `profiles/`, and `logs/`. Agents communicate by appending plain-text
messages to another agent's inbox.

B200 is the final decision target. A100/H100 runs can keep iteration moving, but
they should not decide final dispatch, block sizes, or promotion by themselves.

## Agents

The minimum runtime set:

```text
coordinator
remote_operator
cuda_worker
kernel_engineer_<bucket>
```

- `coordinator` owns priorities and queue shape: which experiments run next,
  reading summaries, writing follow-up tasks, and deciding promotions. It wears
  the sweep-design and profile-analysis hats until those become real bottlenecks.
- `remote_operator` owns cloud machines: rents Prime/AWS instances, verifies CUDA
  and NCU, starts `tmux`, syncs the repo, copies artifacts back, shuts down.
- `cuda_worker` owns remote execution: claims jobs, runs benchmark/profile
  commands, writes logs, updates status, packages outputs. This is the one
  long-lived process worth keeping running (see Remote GPU Rule).
- `kernel_engineer_<bucket>` owns one input shape/type bucket (see below).

Add `correctness_guard` (validation) and `submission_packer` (self-contained
`submission.py`, no local imports, no CUDA streams) when you are close to
promoting a kernel. Split sweep design or profile analysis into their own agents
only once the coordinator can no longer keep up. Keep implementation,
validation, profiling, and promotion as separate decisions - agents should not
self-promote their own work.

## Kernel Engineer Agents

Create a separate kernel engineer for each serious candidate path. The name
encodes the bucket it owns, e.g.:

```text
kernel_engineer_small_n32_dense
kernel_engineer_mid_n176_352_dense
kernel_engineer_large_n512_1024_dense
kernel_engineer_mixed_batch_dispatch
```

Each owns exactly one claim:

```text
For this input shape/type bucket, my kernel is correct and faster than the
current dispatch choice on the target hardware.
```

Its `task.txt` pins the bucket and success criteria:

```text
agent: kernel_engineer_large_n512_1024_dense
bucket: {n: 512,1024, cases: dense,band,upper, batch: 16}
candidate_kernel: kernels/cuda/<name>.cu
baseline: baselines.geqrf_baseline
target_hardware: b200
success_criteria:
  - passes local_eval for owned bucket
  - beats current dispatch on B200
  - NCU bottleneck is understood
  - no CUDA streams
  - promotion path to self-contained submission.py is clear
```

Don't tune outside your bucket. If a kernel accidentally helps another bucket,
report it to the coordinator rather than expanding ownership silently.

## Layout

Use an ignored runtime directory. `swarm/` stays out of git (local state).

```text
swarm/
  agents/<name>/{task.txt,inbox.txt,status.txt}
  queue/{pending,running,done,failed}/
```

## Task File

Each `task.txt` is short and concrete:

```text
agent: cuda_worker
task: profile geqrf baseline on official cases 3,7,9
hardware: b200
command: GEQRF_BASELINE_CASES="3 7 9" ./scripts/profile_geqrf_baseline.sh
done_when:
  - command exits successfully
  - local_benchmark correctness passed for same cases
  - .ncu-rep files and logs are present
```

## Inbox Protocol

Append-only text. Don't edit or delete old messages; append replies. Read your
inbox before starting a task and after finishing one.

```text
---
from: coordinator
to: cuda_worker
time: 2026-06-22T23:50:00Z
subject: run baseline profiles

Please profile official cases 3,7,9 with the geqrf baseline.
Write logs to logs/geqrf_baseline_*.log and report the archive name.
```

## Status File

Each agent updates `status.txt` whenever state changes, and at least every five
minutes while running so stuck jobs are visible without a service process.

```text
state: running          # idle queued running blocked done failed
task: profile geqrf baseline on official cases 3,7,9
last_update: 2026-06-22T23:58:20Z
pid: 12345
cuda_visible_devices: 0
current_job: 001_geqrf_b200_case3
latest_log: logs/geqrf_baseline_cases_3_7_9.log
```

If `last_update` is older than ~10 minutes, the coordinator should inspect the
remote process before requeueing; don't assume failure while `ncu` is still
writing output.

## Queue Protocol

File-based: create a job in `pending/`, a worker claims it by moving it to
`running/`, then moves it to `done/` or `failed/`. The worker whose move succeeds
owns the job. One job file per command, sortable names:

```text
001_geqrf_b200_case3.job
002_geqrf_b200_case7.job
```

Job file:

```text
job_id: 001_geqrf_b200_case3
priority: 10                       # lower runs first
agent: any_cuda_worker
hardware: b200
depends_on:
  - correctness_b200_case3
command: QR_HARDWARE=b200 QR_CASE_INDEX=3 NCU_SET=roofline ./scripts/ncu_qr.sh
log: logs/001_geqrf_b200_case3.log
claimed_at:                        # stamped on claim, with pid above
```

Rules: correctness failures don't retry automatically; cloud interruptions
(spot/ssh/preemption) may. Each completed job records the exact command it ran,
whether it passed, runtime, main artifact, and a next action.

`depends_on` blocks a job until the named jobs are in `done/`. Use it for
correctness-before-profile and baseline-before-candidate chains.

Summary file:

```text
job_id: 001_geqrf_b200_case3
state: done
hardware: b200
case: 3
command: QR_HARDWARE=b200 QR_CASE_INDEX=3 NCU_SET=roofline ./scripts/ncu_qr.sh
passed: true
runtime_s: 184
artifacts:
  - logs/001_geqrf_b200_case3.log
  - profiles/<report>.ncu-rep
next_action: profile candidate kernel for same case
```

**Stale locks:** if a worker dies mid-job, its file stays in `running/`. The
coordinator (or any idle worker) may move a `running/` job back to `pending/` if
its `claimed_at` is older than ~2x expected runtime and the pid is dead.

## Remote GPU Rule

Keep the GPU busy with one long-lived worker in `tmux`:

```bash
mkdir -p swarm/queue/{pending,running,done,failed} swarm/agents/cuda_worker logs
```

The worker loops: check inbox, claim the highest-priority pending job, update
`status.txt`, run the command with `tee` to the job log, write a summary, move
the job to `done/` or `failed/`, repeat.

Run only one active CUDA benchmark/profile job per GPU - concurrent jobs corrupt
wall time, profiler traces, and launch-count interpretation. On multi-GPU
instances, pin each worker with `CUDA_VISIBLE_DEVICES=<id>` or run a single
worker on one visible GPU. Do not add CUDA streams to kernels or submission code.

## Artifact Handoff

After a batch of jobs, package the evidence:

```bash
tar -czf qr-swarm-results-$(hostname)-$(date +%Y%m%d-%H%M%S).tgz \
  swarm results profiles logs
```

The coordinator summarizes completed work into:

```text
job | hardware | case | command | passed | runtime | main artifact | next action
```

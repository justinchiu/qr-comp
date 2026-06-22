# QR Swarm Dashboard

A small web UI that visualizes the swarm's state by reading the project's real
files — no database, no extra services.

It shows four panels:

- **Leaderboard** — per-hardware ranking of variants by wins (fastest passing
  variant per config) and geomean speedup vs the `python_geqrf` baseline.
- **Results vs baseline** — best run per `(hardware, n, case, batch, variant)`
  with speedup vs baseline, filterable by hardware / "beats baseline only".
- **Subagents** — each `swarm/agents/<name>/status.txt`, with stale detection
  (no update in >10 min, per [SWARM.md](../SWARM.md)).
- **Queue** — `swarm/queue/{pending,running,done,failed}/*.job` as columns.

## Data sources

| Panel        | Reads                                              |
| ------------ | -------------------------------------------------- |
| Leaderboard  | `results/*.csv` (derived)                          |
| Results      | `results/*.csv`                                    |
| Subagents    | `swarm/agents/*/status.txt`, `task.txt`, `inbox.txt` |
| Queue        | `swarm/queue/*/*.job`                              |

The server tolerates a missing `swarm/` (panels just show empty) and re-reads
the files on every request, so it reflects live worker activity. The frontend
polls `/api/all` every 5s.

## Run

```bash
# from the repo root, using the project interpreter (stdlib only, no pip deps)
uv run python dashboard/server.py --port 8787
# open http://localhost:8787/
```

To see the Subagents/Queue panels populated before a real worker is running:

```bash
uv run python dashboard/seed_demo.py   # writes sample files under swarm/
# remove with: rm -rf swarm
```

## Editing the frontend

The UI is TypeScript in `src/`, compiled to `static/*.js`. The compiled output
is committed so the server runs without Node. If you change the `.ts` files,
rebuild:

```bash
cd dashboard
npm install      # one-time: installs typescript locally
npm run build    # or: npm run watch
```

## API

`GET /api/all` returns everything (used by the UI). Individual endpoints:
`/api/leaderboard`, `/api/results`, `/api/queue`, `/api/agents`.

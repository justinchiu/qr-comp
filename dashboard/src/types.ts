// Shared types mirroring the JSON returned by dashboard/server.py.

export interface ResultRow {
  source: string;
  hardware: string;
  target_gpu: string;
  batch: string;
  n: string;
  case: string;
  cond: string;
  variant: string;
  panel_type: string;
  block_size: string;
  passed: boolean;
  mean_ms: number;
  std_ms: number | null;
  best_ms: number | null;
  speedup: number | null;
  is_baseline: boolean;
  baseline_ms: number | null;
}

export interface ResultsPayload {
  rows: ResultRow[];
  baseline: string;
  sources: string[];
}

export interface LeaderboardEntry {
  variant: string;
  wins: number;
  configs: number;
  geomean_speedup: number | null;
}

export interface LeaderboardBoard {
  hardware: string;
  entries: LeaderboardEntry[];
}

export interface LeaderboardPayload {
  boards: LeaderboardBoard[];
  baseline: string;
}

export interface Job {
  job_id: string;
  state: string;
  file: string;
  hardware?: string;
  priority?: string;
  agent?: string;
  command?: string;
  log?: string;
  claimed_at?: string;
  passed?: string;
  runtime_s?: string;
  next_action?: string;
  depends_on?: string[];
  [key: string]: unknown;
}

export type QueuePayload = Record<"pending" | "running" | "done" | "failed", Job[]>;

export interface Agent {
  name: string;
  state: string;
  task?: string;
  task_summary?: string;
  last_update?: string;
  pid?: string;
  cuda_visible_devices?: string;
  current_job?: string;
  latest_log?: string;
  inbox_messages?: number;
  [key: string]: unknown;
}

export interface AgentsPayload {
  agents: Agent[];
}

export interface AllPayload {
  results: ResultsPayload;
  leaderboard: LeaderboardPayload;
  queue: QueuePayload;
  agents: AgentsPayload["agents"];
}

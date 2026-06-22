// QR swarm dashboard frontend. Plain DOM, no framework. Polls /api/all and
// renders four panels: leaderboard, results vs baseline, agents, queue.
const REFRESH_MS = 5000;
function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className)
        node.className = className;
    if (text !== undefined)
        node.textContent = text;
    return node;
}
function clear(node) {
    while (node.firstChild)
        node.removeChild(node.firstChild);
}
function speedupClass(speedup) {
    if (speedup === null)
        return "";
    if (speedup >= 1.05)
        return "good";
    if (speedup <= 0.95)
        return "bad";
    return "neutral";
}
function fmt(value, digits = 3) {
    return value === null || value === undefined ? "–" : value.toFixed(digits);
}
// --------------------------------------------------------------------------- //
// Leaderboard panel
// --------------------------------------------------------------------------- //
function renderLeaderboard(boards, baseline) {
    const root = document.getElementById("leaderboard-body");
    clear(root);
    if (boards.length === 0) {
        root.appendChild(el("p", "empty", "No results yet."));
        return;
    }
    for (const board of boards) {
        root.appendChild(el("h3", "hw", board.hardware));
        const table = el("table");
        const head = el("tr");
        ["#", "variant", "wins", "configs", "geomean speedup"].forEach((h) => head.appendChild(el("th", undefined, h)));
        table.appendChild(head);
        board.entries.forEach((e, i) => {
            const tr = el("tr");
            if (e.variant === baseline)
                tr.classList.add("baseline-row");
            tr.appendChild(el("td", "rank", String(i + 1)));
            tr.appendChild(el("td", "variant", e.variant));
            tr.appendChild(el("td", undefined, String(e.wins)));
            tr.appendChild(el("td", "muted", String(e.configs)));
            const sp = el("td", speedupClass(e.geomean_speedup), `${fmt(e.geomean_speedup, 2)}×`);
            tr.appendChild(sp);
            table.appendChild(tr);
        });
        root.appendChild(table);
    }
}
// --------------------------------------------------------------------------- //
// Results vs baseline panel
// --------------------------------------------------------------------------- //
let resultsCache = [];
function renderResults(rows) {
    resultsCache = rows;
    const hwSel = document.getElementById("hw-filter");
    const hardwares = Array.from(new Set(rows.map((r) => r.hardware))).sort();
    const current = hwSel.value;
    clear(hwSel);
    hwSel.appendChild(new Option("all hardware", ""));
    hardwares.forEach((hw) => hwSel.appendChild(new Option(hw, hw)));
    if (hardwares.includes(current) || current === "")
        hwSel.value = current;
    const onlyImprovements = document.getElementById("only-wins").checked;
    drawResultsTable(hwSel.value, onlyImprovements);
}
function drawResultsTable(hw, onlyWins) {
    const root = document.getElementById("results-body");
    clear(root);
    let rows = resultsCache;
    if (hw)
        rows = rows.filter((r) => r.hardware === hw);
    if (onlyWins)
        rows = rows.filter((r) => r.speedup !== null && r.speedup > 1);
    if (rows.length === 0) {
        root.appendChild(el("p", "empty", "No matching rows."));
        return;
    }
    const table = el("table");
    const head = el("tr");
    ["hardware", "n", "case", "batch", "variant", "block", "mean ms", "baseline ms", "speedup", "ok"].forEach((h) => head.appendChild(el("th", undefined, h)));
    table.appendChild(head);
    for (const r of rows) {
        const tr = el("tr");
        if (r.is_baseline)
            tr.classList.add("baseline-row");
        tr.appendChild(el("td", "muted", r.hardware));
        tr.appendChild(el("td", undefined, r.n));
        tr.appendChild(el("td", undefined, r.case));
        tr.appendChild(el("td", undefined, r.batch));
        tr.appendChild(el("td", "variant", r.variant + (r.is_baseline ? " (baseline)" : "")));
        tr.appendChild(el("td", "muted", r.block_size || "–"));
        tr.appendChild(el("td", undefined, fmt(r.mean_ms)));
        tr.appendChild(el("td", "muted", fmt(r.baseline_ms)));
        const sp = el("td", speedupClass(r.speedup), r.speedup === null ? "–" : `${fmt(r.speedup, 2)}×`);
        tr.appendChild(sp);
        tr.appendChild(el("td", r.passed ? "good" : "bad", r.passed ? "✓" : "✗"));
        table.appendChild(tr);
    }
    root.appendChild(table);
}
// --------------------------------------------------------------------------- //
// Agents panel
// --------------------------------------------------------------------------- //
function staleClass(lastUpdate) {
    if (!lastUpdate)
        return "";
    const ts = Date.parse(lastUpdate);
    if (Number.isNaN(ts))
        return "";
    const ageMin = (Date.now() - ts) / 60000;
    return ageMin > 10 ? "stale" : "";
}
function renderAgents(agents) {
    const root = document.getElementById("agents-body");
    clear(root);
    document.getElementById("agents-count").textContent = String(agents.length);
    if (agents.length === 0) {
        root.appendChild(el("p", "empty", "No agents. Run dashboard/seed_demo.py or start a worker."));
        return;
    }
    for (const a of agents) {
        const card = el("div", `card agent state-${a.state}`);
        const header = el("div", "card-head");
        header.appendChild(el("span", "agent-name", a.name));
        const badge = el("span", `badge state-${a.state}`, a.state);
        if (staleClass(a.last_update))
            badge.classList.add("stale");
        header.appendChild(badge);
        card.appendChild(header);
        const task = a.task_summary || a.task || "";
        if (task)
            card.appendChild(el("div", "agent-task", task));
        const meta = el("div", "meta");
        if (a.current_job)
            meta.appendChild(el("span", "chip", `job ${a.current_job}`));
        if (a.cuda_visible_devices !== undefined)
            meta.appendChild(el("span", "chip", `gpu ${a.cuda_visible_devices}`));
        if (a.pid)
            meta.appendChild(el("span", "chip", `pid ${a.pid}`));
        if (a.inbox_messages)
            meta.appendChild(el("span", "chip", `📬 ${a.inbox_messages}`));
        if (a.last_update) {
            const c = el("span", `chip ${staleClass(a.last_update)}`, a.last_update);
            meta.appendChild(c);
        }
        card.appendChild(meta);
        root.appendChild(card);
    }
}
// --------------------------------------------------------------------------- //
// Queue panel
// --------------------------------------------------------------------------- //
const QUEUE_COLUMNS = [
    "pending",
    "running",
    "done",
    "failed",
];
function renderQueue(queue) {
    const root = document.getElementById("queue-body");
    clear(root);
    const total = QUEUE_COLUMNS.reduce((sum, k) => sum + (queue[k]?.length ?? 0), 0);
    document.getElementById("queue-count").textContent = String(total);
    for (const col of QUEUE_COLUMNS) {
        const jobs = queue[col] ?? [];
        const column = el("div", `queue-col col-${col}`);
        const head = el("div", "queue-col-head");
        head.appendChild(el("span", "col-name", col));
        head.appendChild(el("span", "col-count", String(jobs.length)));
        column.appendChild(head);
        if (jobs.length === 0) {
            column.appendChild(el("p", "empty small", "—"));
        }
        else {
            jobs.forEach((j) => column.appendChild(jobCard(j)));
        }
        root.appendChild(column);
    }
}
function jobCard(job) {
    const card = el("div", "card job");
    const head = el("div", "card-head");
    head.appendChild(el("span", "job-id", job.job_id));
    if (job.priority)
        head.appendChild(el("span", "chip", `p${job.priority}`));
    card.appendChild(head);
    if (job.hardware)
        card.appendChild(el("div", "chip hw-chip", job.hardware));
    if (job.command)
        card.appendChild(el("code", "cmd", job.command));
    const meta = el("div", "meta");
    if (job.passed)
        meta.appendChild(el("span", `chip ${job.passed === "true" ? "good" : "bad"}`, `passed ${job.passed}`));
    if (job.runtime_s)
        meta.appendChild(el("span", "chip", `${job.runtime_s}s`));
    if (Array.isArray(job.depends_on) && job.depends_on.length)
        meta.appendChild(el("span", "chip", `deps ${job.depends_on.length}`));
    if (meta.childNodes.length)
        card.appendChild(meta);
    if (job.next_action)
        card.appendChild(el("div", "next", `→ ${job.next_action}`));
    return card;
}
// --------------------------------------------------------------------------- //
// Polling loop
// --------------------------------------------------------------------------- //
async function refresh() {
    const status = document.getElementById("status");
    try {
        const res = await fetch("/api/all", { cache: "no-store" });
        if (!res.ok)
            throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderLeaderboard(data.leaderboard.boards, data.leaderboard.baseline);
        renderResults(data.results.rows);
        renderAgents(data.agents);
        renderQueue(data.queue);
        document.getElementById("sources").textContent =
            data.results.sources.join(", ") || "no csv files";
        status.textContent = `updated ${new Date().toLocaleTimeString()}`;
        status.className = "ok";
    }
    catch (err) {
        status.textContent = `error: ${err.message}`;
        status.className = "err";
    }
}
function init() {
    document.getElementById("hw-filter").addEventListener("change", () => refresh());
    document.getElementById("only-wins").addEventListener("change", () => refresh());
    document.getElementById("refresh-now").addEventListener("click", () => refresh());
    refresh();
    setInterval(refresh, REFRESH_MS);
}
init();
export {};
//# sourceMappingURL=app.js.map
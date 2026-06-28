"use strict";

const DATA = JSON.parse(document.getElementById("site-data").textContent);

const COLORS = {
  teal: "#0c7a84",
  blue: "#2b5fae",
  green: "#2c7d54",
  amber: "#b9720f",
  rose: "#c0344a",
  violet: "#6a4fbb",
  ink: "#111a24",
  muted: "#586676",
  faint: "#8995a4",
  line: "#d3dae3",
  lineSoft: "#e4e9ef",
  surface: "#ffffff",
  tint: "#f6f8fb"
};

/* 11 reasonably distinct categorical hues for programs */
const PROGRAM_PALETTE = [
  "#0c7a84", "#2b5fae", "#b9720f", "#c0344a", "#2c7d54", "#6a4fbb",
  "#1f9aa6", "#8a3fa0", "#b4543b", "#46708f", "#7d7a25"
];

const STATUS_COLORS = {
  Confirmed: COLORS.green,
  Promising: COLORS.amber,
  Negative: COLORS.rose,
  Open: COLORS.blue,
  Retired: COLORS.faint
};

const state = { search: "", program: "all", ready: "all", need: "all", sort: { key: "id", dir: 1 } };

const els = {
  search: document.getElementById("globalSearch"),
  program: document.getElementById("programFilter"),
  ready: document.getElementById("readyFilter"),
  need: document.getElementById("needFilter"),
  reset: document.getElementById("resetFilters"),
  filterStatus: document.getElementById("filterStatus"),
  tooltip: document.getElementById("tooltip"),
  drawer: document.getElementById("detailDrawer"),
  drawerScrim: document.getElementById("drawerScrim"),
  drawerClose: document.getElementById("drawerClose"),
  drawerEyebrow: document.getElementById("drawerEyebrow"),
  drawerTitle: document.getElementById("drawerTitle"),
  drawerContent: document.getElementById("drawerContent")
};

const programById = new Map(DATA.programs.map((p) => [p.id, p]));
const experimentById = new Map(DATA.experiments.map((e) => [e.id, e]));
const programColorMap = new Map(DATA.programs.map((p, i) => [p.id, PROGRAM_PALETTE[i % PROGRAM_PALETTE.length]]));
const programIndex = new Map(DATA.programs.map((p, i) => [p.id, i]));

/* ---------------------------------------------------------------- helpers */
function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function formatNumber(value) { return Number(value || 0).toLocaleString(); }
function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value < 1024) return { n: String(value), unit: "B" };
  const units = ["KB", "MB", "GB", "TB"];
  let scaled = value / 1024, unit = units[0];
  for (let i = 1; i < units.length && scaled >= 1024; i += 1) { scaled /= 1024; unit = units[i]; }
  return { n: scaled.toFixed(scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2), unit };
}
function programColor(id) { return programColorMap.get(id) || COLORS.muted; }
function titleForProgram(id) { return programById.get(id)?.title || id; }
function repoLink(path) { return path ? `${DATA.repo.github}/blob/main/${path}` : ""; }
function splitList(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  return String(value).split(";").map((s) => s.trim()).filter(Boolean);
}
function listText(items) { return (items || []).join(", "); }
function textBlob(item) {
  return Object.values(item).flatMap((v) => (Array.isArray(v) ? v : [v])).join(" ").toLowerCase();
}
function matchesSearch(item) { return !state.search || textBlob(item).includes(state.search.toLowerCase()); }
function matchesProgram(item, field = "programs") {
  if (state.program === "all") return true;
  return splitList(item[field]).includes(state.program);
}
function clear(el) { el.replaceChildren(); }

function svg(tag, attrs = {}, children = []) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const c of children) el.appendChild(c);
  return el;
}
function text(content, attrs = {}) {
  return svg("text", attrs, [document.createTextNode(String(content))]);
}

/* ---------------------------------------------------------------- tooltip */
function showTooltip(event, html) {
  els.tooltip.innerHTML = html;
  const pad = 12;
  let x = event.clientX, y = event.clientY;
  els.tooltip.style.left = `${Math.min(window.innerWidth - pad, Math.max(pad, x))}px`;
  els.tooltip.style.top = `${y}px`;
  els.tooltip.classList.add("visible");
}
function hideTooltip() { els.tooltip.classList.remove("visible"); }
function attachTooltip(el, html) {
  el.addEventListener("mousemove", (e) => showTooltip(e, html));
  el.addEventListener("mouseleave", hideTooltip);
}

/* ---------------------------------------------------------------- filters */
function filteredExperiments() {
  return DATA.experiments.filter((e) => {
    if (!matchesSearch(e)) return false;
    if (!matchesProgram(e)) return false;
    if (state.ready !== "all" && e.anchor_ready !== state.ready) return false;
    if (state.need !== "all" && !(e.needs || []).includes(state.need)) return false;
    return true;
  });
}
function filteredQueue() { return DATA.queue.filter((i) => matchesSearch(i) && matchesProgram(i)); }
function filteredClaims() {
  return DATA.claims.filter((c) => {
    if (!matchesSearch(c)) return false;
    if (state.program === "all") return true;
    return splitList(c.programs).includes(state.program);
  });
}
function filteredPrograms() {
  const s = state.search.toLowerCase();
  return DATA.programs.filter((p) => {
    if (state.program !== "all" && p.id !== state.program) return false;
    return !s || textBlob(p).includes(s);
  });
}
function filtersActive() {
  return state.search || state.program !== "all" || state.ready !== "all" || state.need !== "all";
}

function setProgram(id) {
  state.program = state.program === id ? "all" : id;
  els.program.value = state.program;
  render();
}

/* ---------------------------------------------------------------- drawer */
let lastFocused = null;
function openDrawer(eyebrow, title, lead, rows, extra = "") {
  lastFocused = document.activeElement;
  const detail = rows
    .filter((r) => r[1] !== undefined && r[1] !== "" && r[1] !== null)
    .map(([label, value]) => `<dt>${escapeHtml(label)}</dt><dd>${value}</dd>`)
    .join("");
  els.drawerEyebrow.textContent = eyebrow;
  els.drawerTitle.textContent = title;
  els.drawerContent.innerHTML = `
    ${lead ? `<p class="drawer-lead">${escapeHtml(lead)}</p>` : ""}
    <dl>${detail}</dl>
    ${extra}
  `;
  els.drawer.classList.add("open");
  els.drawerScrim.classList.add("open");
  els.drawer.setAttribute("aria-hidden", "false");
  els.drawerClose.focus();
}
function closeDrawer() {
  els.drawer.classList.remove("open");
  els.drawerScrim.classList.remove("open");
  els.drawer.setAttribute("aria-hidden", "true");
  if (lastFocused && lastFocused.focus) lastFocused.focus();
}

/* ---------------------------------------------------------------- header / hero */
function renderHero() {
  const s = DATA.summary;
  document.getElementById("heroStamp").textContent =
    `Generated ${DATA.generated_at} · ${formatNumber(s.experiments)} experiments indexed`;
  document.getElementById("heroLede").innerHTML =
    `<b>${formatNumber(s.experiments)} experiments</b> across <b>${formatNumber(s.programs)} research programs</b>, ` +
    `with <b>${formatNumber(s.claims)} shared claims</b> and <b>${formatNumber(s.future_proposals)} queued probes</b> — ` +
    `indexed, cross-linked, and explorable from one page.`;
  const repo = DATA.repo.github;
  document.getElementById("footerRepo").href = repo;
  document.getElementById("footerStamp").textContent = `Generated ${DATA.generated_at}`;
}

function renderFilters() {
  els.program.innerHTML = [
    '<option value="all">All programs</option>',
    ...DATA.programs.map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.title)}</option>`)
  ].join("");
  const needs = Array.from(new Set(DATA.experiments.flatMap((e) => e.needs || []))).sort();
  els.need.innerHTML = [
    '<option value="all">All curation needs</option>',
    ...needs.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`)
  ].join("");
}

function updateFilterStatus() {
  if (!filtersActive()) {
    els.filterStatus.textContent = "Filters apply across every panel below.";
    return;
  }
  const bits = [];
  if (state.program !== "all") bits.push(titleForProgram(state.program));
  if (state.ready === "yes") bits.push("anchor-ready");
  if (state.ready === "no") bits.push("needs curation");
  if (state.need !== "all") bits.push(`need: ${state.need}`);
  if (state.search) bits.push(`“${state.search}”`);
  els.filterStatus.textContent = `Filtered → ${bits.join(" · ")}`;
}

/* ---------------------------------------------------------------- metrics */
function renderMetrics() {
  const s = DATA.summary;
  const size = formatBytes(s.total_size_bytes);
  const metrics = [
    ["Experiments", formatNumber(s.experiments), "", "Self-contained experiment folders", COLORS.teal],
    ["Programs", formatNumber(s.programs), "", "Durable lines of inquiry", COLORS.blue],
    ["Anchor-ready", formatNumber(s.anchor_ready), "", "Reusable directly, no curation", COLORS.green],
    ["Needs curation", formatNumber(s.needs_curation), "", "Gaps before safe reuse", COLORS.amber],
    ["Future queue", formatNumber(s.future_proposals), "", "Structured next probes", COLORS.rose],
    ["Claims", formatNumber(s.claims), "", "Shared evidence statements", COLORS.violet],
    ["Files indexed", formatNumber(s.total_files), "", "Tracked corpus files", COLORS.teal],
    ["Tracked size", size.n, size.unit, "Repository-local footprint", COLORS.blue]
  ];
  document.getElementById("metricGrid").innerHTML = metrics
    .map(([label, value, unit, help, color]) => `
      <article class="metric-card" style="--accent:${color}">
        <p class="eyebrow">${escapeHtml(label)}</p>
        <span class="metric-value">${escapeHtml(value)}${unit ? `<span class="unit">${escapeHtml(unit)}</span>` : ""}</span>
        <p class="metric-help">${escapeHtml(help)}</p>
      </article>`)
    .join("");
}

/* ---------------------------------------------------------------- bar chart */
function barChart(targetId, rows, options = {}) {
  const target = document.getElementById(targetId);
  clear(target);
  if (!rows.length) { target.innerHTML = '<div class="empty-state">No data.</div>'; return; }
  const width = options.width || 560;
  const rowH = options.rowH || 30;
  const labelW = options.labelW ?? 150;
  const top = 6, bottom = 6, rightPad = 46;
  const height = top + bottom + rows.length * rowH;
  const maxValue = Math.max(1, ...rows.map((r) => Number(r.value || 0)));
  const plotW = width - labelW - rightPad;
  const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": options.label || "bar chart", preserveAspectRatio: "xMinYMin meet" });
  rows.forEach((row, i) => {
    const y = top + i * rowH;
    const value = Number(row.value || 0);
    const bw = Math.max(2, (plotW * value) / maxValue);
    const color = options.colors?.[row.id] || options.color || PROGRAM_PALETTE[i % PROGRAM_PALETTE.length];
    const label = row.label || row.id;
    chart.appendChild(text(label, { x: 0, y: y + rowH / 2 + 4, class: "chart-label" }));
    chart.appendChild(svg("rect", { x: labelW, y: y + rowH / 2 - 9, width: plotW, height: 18, rx: 5, fill: COLORS.tint }));
    const bar = svg("rect", { x: labelW, y: y + rowH / 2 - 9, width: bw, height: 18, rx: 5, fill: color });
    attachTooltip(bar, `<strong>${escapeHtml(label)}</strong><br><span class="tip-meta">${formatNumber(value)}</span>`);
    chart.appendChild(bar);
    chart.appendChild(text(formatNumber(value), { x: labelW + bw + 7, y: y + rowH / 2 + 4, class: "chart-value" }));
  });
  target.appendChild(chart);
}

/* ---------------------------------------------------------------- orbital corpus map */
function programNetwork() {
  const target = document.getElementById("programNetwork");
  clear(target);
  const W = 760, H = 470, cx = W / 2, cy = H / 2, ringR = 168;
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Corpus map: research programs by weight and readiness" });

  // ambient guide rings
  [ringR + 46, ringR, ringR - 60].forEach((r) => {
    chart.appendChild(svg("circle", { cx, cy, r, fill: "none", stroke: COLORS.lineSoft, "stroke-width": 1 }));
  });

  const maxCount = Math.max(1, ...DATA.programs.map((p) => p.experiment_count));
  const nodeR = (c) => 14 + (Math.sqrt(c) / Math.sqrt(maxCount)) * 24;
  const dim = state.program !== "all";

  // links first
  DATA.programs.forEach((p, i) => {
    const angle = -Math.PI / 2 + (i / DATA.programs.length) * Math.PI * 2;
    const x = cx + Math.cos(angle) * ringR, y = cy + Math.sin(angle) * ringR;
    const active = state.program === p.id;
    chart.appendChild(svg("line", {
      x1: cx, y1: cy, x2: x, y2: y,
      stroke: active ? programColor(p.id) : COLORS.line,
      "stroke-width": active ? 2.4 : 1 + (p.experiment_count / maxCount) * 1.6,
      opacity: dim && !active ? 0.25 : 0.8
    }));
  });

  // core
  chart.appendChild(svg("circle", { cx, cy, r: 50, fill: COLORS.surface, stroke: COLORS.ink, "stroke-width": 1.5 }));
  chart.appendChild(text(formatNumber(DATA.summary.experiments), { x: cx, y: cy - 2, "text-anchor": "middle", class: "chart-value", style: "font-size:24px" }));
  chart.appendChild(text("EXPERIMENTS", { x: cx, y: cy + 16, "text-anchor": "middle", class: "axis-label" }));

  // nodes: ring shows experiments, arc fill shows readiness ratio
  DATA.programs.forEach((p, i) => {
    const angle = -Math.PI / 2 + (i / DATA.programs.length) * Math.PI * 2;
    const x = cx + Math.cos(angle) * ringR, y = cy + Math.sin(angle) * ringR;
    const r = nodeR(p.experiment_count);
    const color = programColor(p.id);
    const active = state.program === p.id;
    const opacity = dim && !active ? 0.3 : 1;
    const g = svg("g", { tabindex: 0, role: "button", "aria-label": `${p.title}: ${p.experiment_count} experiments`, style: "cursor:pointer" });
    g.style.opacity = opacity;
    // base disk
    g.appendChild(svg("circle", { cx: x, cy: y, r, fill: COLORS.surface, stroke: color, "stroke-width": active ? 3 : 1.5 }));
    // readiness arc (ring)
    const ratio = p.experiment_count ? p.anchor_ready_count / p.experiment_count : 0;
    const rr = r - 4;
    const circ = 2 * Math.PI * rr;
    g.appendChild(svg("circle", {
      cx: x, cy: y, r: rr, fill: "none", stroke: color, "stroke-width": 4, opacity: 0.9,
      "stroke-dasharray": `${ratio * circ} ${circ}`, transform: `rotate(-90 ${x} ${y})`, "stroke-linecap": "round"
    }));
    g.appendChild(text(p.experiment_count, { x, y: y + 4, "text-anchor": "middle", class: "chart-value", style: "font-size:13px" }));
    attachTooltip(g, `<strong>${escapeHtml(p.title)}</strong><br><span class="tip-meta">${p.experiment_count} experiments · ${p.anchor_ready_count} ready · ${p.queue_count} queued · ${p.claim_count} claims</span>`);
    g.addEventListener("click", () => setProgram(p.id));
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setProgram(p.id); } });
    chart.appendChild(g);
  });

  target.appendChild(chart);
  renderProgramLegend();
}

function renderProgramLegend() {
  const legend = document.getElementById("programLegend");
  legend.innerHTML = DATA.programs
    .map((p) => {
      const dim = state.program !== "all" && state.program !== p.id;
      return `<button class="legend-item${dim ? " dim" : ""}" data-program="${escapeHtml(p.id)}" type="button" title="${escapeHtml(p.title)}">
        <span class="legend-swatch" style="background:${programColor(p.id)}"></span>${escapeHtml(p.title)} <b>${p.experiment_count}</b>
      </button>`;
    })
    .join("");
  legend.querySelectorAll("[data-program]").forEach((b) => {
    b.addEventListener("click", () => setProgram(b.getAttribute("data-program")));
  });
}

/* ---------------------------------------------------------------- readiness donut */
function donutChart() {
  const target = document.getElementById("readinessDonut");
  clear(target);
  const ready = DATA.summary.anchor_ready;
  const total = DATA.summary.experiments || 1;
  const needs = Math.max(0, total - ready);
  const pct = Math.round((ready / total) * 100);
  const W = 260, H = 230, cx = W / 2, cy = 104, radius = 72;
  const circ = 2 * Math.PI * radius;
  const readyArc = (ready / total) * circ;
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${pct}% anchor-ready` });
  chart.appendChild(svg("circle", { cx, cy, r: radius, fill: "none", stroke: COLORS.lineSoft, "stroke-width": 22 }));
  chart.appendChild(svg("circle", {
    cx, cy, r: radius, fill: "none", stroke: COLORS.green, "stroke-width": 22, "stroke-linecap": "round",
    "stroke-dasharray": `${readyArc} ${circ - readyArc}`, transform: `rotate(-90 ${cx} ${cy})`
  }));
  chart.appendChild(text(`${pct}%`, { x: cx, y: cy + 2, "text-anchor": "middle", class: "chart-value", style: "font-size:34px;font-family:var(--display)" }));
  chart.appendChild(text("anchor-ready", { x: cx, y: cy + 22, "text-anchor": "middle", class: "axis-label" }));
  // legend line
  chart.appendChild(svg("rect", { x: 36, y: H - 22, width: 10, height: 10, rx: 2, fill: COLORS.green }));
  chart.appendChild(text(`${formatNumber(ready)} ready`, { x: 52, y: H - 13, class: "chart-label" }));
  chart.appendChild(svg("rect", { x: 150, y: H - 22, width: 10, height: 10, rx: 2, fill: COLORS.lineSoft }));
  chart.appendChild(text(`${formatNumber(needs)} to curate`, { x: 166, y: H - 13, class: "chart-label" }));
  target.appendChild(chart);
}

/* ---------------------------------------------------------------- scatter -> files skyline */
function scatterPlot() {
  const target = document.getElementById("experimentScatter");
  clear(target);
  const rows = filteredExperiments().slice().sort((a, b) => b.total_files - a.total_files);
  if (!rows.length) { target.innerHTML = '<div class="empty-state">No experiments match.</div>'; return; }
  const W = 560, H = 200, top = 16, bottom = 26, left = 8, right = 8;
  const maxFiles = Math.max(1, ...rows.map((e) => e.total_files));
  const plotW = W - left - right, plotH = H - top - bottom;
  const n = rows.length;
  const bw = Math.max(1.5, plotW / n - 1);
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Files per experiment, sorted", preserveAspectRatio: "xMinYMin meet" });
  // baseline
  chart.appendChild(svg("line", { x1: left, y1: H - bottom, x2: W - right, y2: H - bottom, stroke: COLORS.line }));
  rows.forEach((e, i) => {
    const x = left + (i / n) * plotW;
    const h = Math.max(1.5, (e.total_files / maxFiles) * plotH);
    const color = e.anchor_ready === "yes" ? COLORS.green : COLORS.amber;
    const bar = svg("rect", { x, y: H - bottom - h, width: bw, height: h, rx: bw > 3 ? 1.5 : 0, fill: color, opacity: 0.85 });
    bar.style.cursor = "pointer";
    attachTooltip(bar, `<strong>${escapeHtml(e.id)}</strong><br><span class="tip-meta">${formatNumber(e.total_files)} files · ${e.anchor_ready === "yes" ? "ready" : "needs curation"}</span>`);
    bar.addEventListener("click", () => openExperiment(e));
    chart.appendChild(bar);
  });
  chart.appendChild(text(`${n} experiments, sorted by file count (max ${formatNumber(maxFiles)})`, { x: left, y: H - 8, class: "axis-label" }));
  // legend
  chart.appendChild(svg("rect", { x: W - right - 150, y: H - 16, width: 9, height: 9, rx: 2, fill: COLORS.green }));
  chart.appendChild(text("ready", { x: W - right - 137, y: H - 8, class: "chart-tick" }));
  chart.appendChild(svg("rect", { x: W - right - 92, y: H - 16, width: 9, height: 9, rx: 2, fill: COLORS.amber }));
  chart.appendChild(text("needs curation", { x: W - right - 79, y: H - 8, class: "chart-tick" }));
  target.appendChild(chart);
}

/* ---------------------------------------------------------------- program heatmap */
function programHeatmap() {
  const target = document.getElementById("programHeatmap");
  clear(target);
  const metrics = [
    { key: "experiment_count", label: "Total", color: COLORS.blue },
    { key: "anchor_ready_count", label: "Ready", color: COLORS.green },
    { key: "queue_count", label: "Queued", color: COLORS.amber },
    { key: "claim_count", label: "Claims", color: COLORS.violet }
  ];
  const W = 560, labelW = 230, cellW = (W - labelW) / metrics.length, rowH = 30, top = 30;
  const progs = DATA.programs.slice().sort((a, b) => b.experiment_count - a.experiment_count);
  const H = top + progs.length * rowH + 6;
  const maxes = {};
  metrics.forEach((m) => { maxes[m.key] = Math.max(1, ...progs.map((p) => p[m.key])); });
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Readiness by program", preserveAspectRatio: "xMinYMin meet" });
  metrics.forEach((m, ci) => {
    chart.appendChild(text(m.label, { x: labelW + ci * cellW + cellW / 2, y: 18, "text-anchor": "middle", class: "axis-label" }));
  });
  progs.forEach((p, ri) => {
    const y = top + ri * rowH;
    const active = state.program === p.id;
    if (active) chart.appendChild(svg("rect", { x: 0, y: y - 1, width: W, height: rowH, fill: COLORS.tint, rx: 5 }));
    chart.appendChild(svg("rect", { x: 0, y: y + rowH / 2 - 7, width: 6, height: 14, rx: 2, fill: programColor(p.id) }));
    const tlabel = text(p.title, { x: 14, y: y + rowH / 2 + 4, class: "chart-label" });
    chart.appendChild(tlabel);
    metrics.forEach((m, ci) => {
      const value = p[m.key];
      const intensity = value / maxes[m.key];
      const cellX = labelW + ci * cellW;
      const cell = svg("rect", {
        x: cellX + 3, y: y + 3, width: cellW - 6, height: rowH - 6, rx: 5,
        fill: m.color, opacity: 0.1 + intensity * 0.8, style: "cursor:pointer"
      });
      attachTooltip(cell, `<strong>${escapeHtml(p.title)}</strong><br><span class="tip-meta">${m.label}: ${formatNumber(value)}</span>`);
      cell.addEventListener("click", () => setProgram(p.id));
      chart.appendChild(cell);
      chart.appendChild(text(formatNumber(value), {
        x: cellX + cellW / 2, y: y + rowH / 2 + 4, "text-anchor": "middle",
        class: "chart-value", style: `fill:${intensity > 0.55 ? "#fff" : COLORS.ink}`
      }));
    });
  });
  target.appendChild(chart);
}

/* ---------------------------------------------------------------- claim matrix */
function claimGraph() {
  const target = document.getElementById("claimGraph");
  clear(target);
  const claims = DATA.claims;
  const progs = DATA.programs;
  if (!claims.length) { target.innerHTML = '<div class="empty-state">No claims yet.</div>'; return; }
  const W = 560, labelW = 240, top = 56, rowH = 30, colW = (W - labelW) / claims.length;
  const H = top + progs.length * rowH + 8;
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Claims linked to programs matrix", preserveAspectRatio: "xMinYMin meet" });
  // column headers (claim ids, rotated status dot above)
  claims.forEach((c, ci) => {
    const x = labelW + ci * colW + colW / 2;
    chart.appendChild(svg("circle", { cx: x, cy: 18, r: 6, fill: STATUS_COLORS[c.status] || COLORS.violet }));
    const t = text(c.id, { x, y: 44, "text-anchor": "middle", class: "chart-value", style: "cursor:pointer" });
    t.addEventListener("click", () => openClaim(c));
    attachTooltip(t, `<strong>${escapeHtml(c.id)}: ${escapeHtml(c.title)}</strong><br><span class="tip-meta">${escapeHtml(c.status)}</span>`);
    chart.appendChild(t);
  });
  progs.forEach((p, ri) => {
    const y = top + ri * rowH;
    chart.appendChild(svg("rect", { x: 0, y: y + rowH / 2 - 7, width: 6, height: 14, rx: 2, fill: programColor(p.id) }));
    chart.appendChild(text(p.title, { x: 14, y: y + rowH / 2 + 4, class: "chart-label" }));
    claims.forEach((c, ci) => {
      const linked = splitList(c.programs).includes(p.id);
      const x = labelW + ci * colW + colW / 2;
      if (linked) {
        const dot = svg("circle", { cx: x, cy: y + rowH / 2, r: 6.5, fill: STATUS_COLORS[c.status] || COLORS.violet, style: "cursor:pointer" });
        attachTooltip(dot, `<strong>${escapeHtml(c.id)}</strong> ↔ <strong>${escapeHtml(p.title)}</strong><br><span class="tip-meta">${escapeHtml(c.title)}</span>`);
        dot.addEventListener("click", () => openClaim(c));
        chart.appendChild(dot);
      } else {
        chart.appendChild(svg("circle", { cx: x, cy: y + rowH / 2, r: 2, fill: COLORS.line }));
      }
    });
  });
  target.appendChild(chart);
  renderClaimStatusLegend();
}

function renderClaimStatusLegend() {
  const host = document.getElementById("claimStatusLegend");
  if (!host) return;
  const order = ["Confirmed", "Promising", "Open", "Negative", "Retired"];
  const present = new Set(DATA.claims.map((c) => c.status));
  host.innerHTML = order.filter((s) => present.has(s)).map((s) => {
    const n = DATA.claims.filter((c) => c.status === s).length;
    return `<span class="legend-item" style="cursor:default"><span class="legend-swatch" style="background:${STATUS_COLORS[s] || COLORS.violet}"></span>${escapeHtml(s)} <b>${n}</b></span>`;
  }).join("");
}

/* ---------------------------------------------------------------- programs */
function renderPrograms() {
  const rows = filteredPrograms();
  const host = document.getElementById("programCards");
  if (!rows.length) { host.innerHTML = '<div class="empty-state">No programs match this filter.</div>'; return; }
  host.innerHTML = rows.map((p) => `
    <article class="program-card" tabindex="0" data-program="${escapeHtml(p.id)}" style="--accent:${programColor(p.id)}">
      <p class="card-id">${escapeHtml(p.id)}</p>
      <h3>${escapeHtml(p.title)}</h3>
      <p class="focus">${escapeHtml(p.focus)}</p>
      <div class="stat-row">
        <div class="stat"><span class="n">${formatNumber(p.experiment_count)}</span><span class="k">exp</span></div>
        <div class="stat"><span class="n" style="color:${COLORS.green}">${formatNumber(p.anchor_ready_count)}</span><span class="k">ready</span></div>
        <div class="stat"><span class="n" style="color:${COLORS.amber}">${formatNumber(p.queue_count)}</span><span class="k">queued</span></div>
        <div class="stat"><span class="n" style="color:${COLORS.violet}">${formatNumber(p.claim_count)}</span><span class="k">claims</span></div>
      </div>
    </article>`).join("");
  host.querySelectorAll("[data-program]").forEach((card) => {
    const id = card.getAttribute("data-program");
    card.addEventListener("click", () => openProgram(programById.get(id)));
    card.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openProgram(programById.get(id)); } });
  });
}

function openProgram(p) {
  if (!p) return;
  const extra = `<a class="text-link" href="${repoLink(p.path)}">Open charter on GitHub <span class="arr">→</span></a>`;
  openDrawer("Research program", p.title, p.excerpt || p.focus, [
    ["Program id", `<span class="mono-val">${escapeHtml(p.id)}</span>`],
    ["Focus", escapeHtml(p.focus)],
    ["Experiments", `${formatNumber(p.experiment_count)} (${formatNumber(p.anchor_ready_count)} anchor-ready)`],
    ["Queued probes", formatNumber(p.queue_count)],
    ["Claims", formatNumber(p.claim_count)],
    ["Seed tags", (p.seed_tags || []).length ? (p.seed_tags || []).map((t) => `<span class="pill">${escapeHtml(t)}</span>`).join(" ") : ""]
  ], `<div style="margin-top:8px"><button class="text-link" type="button" id="drawerFilterBtn">Filter the atlas to this program <span class="arr">→</span></button></div>${extra ? `<div style="margin-top:12px">${extra}</div>` : ""}`);
  const btn = document.getElementById("drawerFilterBtn");
  if (btn) btn.addEventListener("click", () => { closeDrawer(); setProgram(p.id); document.getElementById("experiments").scrollIntoView({ behavior: "smooth" }); });
}

/* ---------------------------------------------------------------- experiments table */
const TABLE_COLS = [
  { key: "id", label: "Experiment" },
  { key: "programs", label: "Programs" },
  { key: "anchor_ready", label: "Ready" },
  { key: "run_surface", label: "Run surface" },
  { key: "needs", label: "Needs" }
];
const TABLE_LIMIT = 200;

function sortExperiments(rows) {
  const { key, dir } = state.sort;
  const val = (e) => {
    if (key === "programs") return e.programs.map(titleForProgram).join(", ");
    if (key === "needs") return (e.needs || []).length;
    if (key === "anchor_ready") return e.anchor_ready === "yes" ? 1 : 0;
    return String(e[key] ?? "");
  };
  return rows.slice().sort((a, b) => {
    const av = val(a), bv = val(b);
    if (typeof av === "number" && typeof bv === "number") return (av - bv) * dir;
    return String(av).localeCompare(String(bv)) * dir;
  });
}

function renderExperiments() {
  const all = filteredExperiments();
  const rows = sortExperiments(all);
  const shown = rows.slice(0, TABLE_LIMIT);
  document.getElementById("experimentCount").innerHTML = `<b>${formatNumber(all.length)}</b> of ${formatNumber(DATA.experiments.length)} shown`;
  const caret = (key) => state.sort.key === key ? `<span class="sort-caret">${state.sort.dir > 0 ? "▲" : "▼"}</span>` : "";
  const head = TABLE_COLS.map((c) => `<th data-sort="${c.key}">${c.label}${caret(c.key)}</th>`).join("");
  const body = shown.map((e) => `
    <tr data-experiment="${escapeHtml(e.id)}">
      <td><span class="cell-id">${escapeHtml(e.id)}</span><span class="cell-title">${escapeHtml(e.title)}</span></td>
      <td class="col-progs" title="${escapeHtml(e.programs.map(titleForProgram).join(", "))}"><span class="prog-dots">${e.programs.map((id) => `<span class="prog-dot" style="background:${programColor(id)}"></span>`).join("")}${e.programs.length > 1 ? `<span class="prog-count">${e.programs.length}</span>` : ""}</span></td>
      <td><span class="status-tag ${e.anchor_ready === "yes" ? "ready" : "notready"}">${e.anchor_ready === "yes" ? "ready" : "curate"}</span></td>
      <td class="col-needs">${escapeHtml(e.run_surface || "—")}</td>
      <td class="col-needs">${escapeHtml((e.needs || []).join(", ") || "none")}</td>
    </tr>`).join("");
  const note = all.length > TABLE_LIMIT ? `<div class="table-toolbar">Showing first ${TABLE_LIMIT} of ${formatNumber(all.length)}. Narrow with search or filters.</div>` : "";
  document.getElementById("experimentTable").innerHTML = all.length
    ? `${note}<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`
    : '<div class="empty-state">No experiments match the active filters.<span class="mono">try resetting filters</span></div>';
  document.querySelectorAll("#experimentTable [data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.getAttribute("data-sort");
      if (state.sort.key === key) state.sort.dir *= -1; else state.sort = { key, dir: 1 };
      renderExperiments();
    });
  });
  document.querySelectorAll("#experimentTable [data-experiment]").forEach((tr) => {
    tr.addEventListener("click", () => openExperiment(experimentById.get(tr.getAttribute("data-experiment"))));
  });
}

function openExperiment(e) {
  if (!e) return;
  openDrawer("Experiment", e.title, e.summary, [
    ["Id", `<span class="mono-val">${escapeHtml(e.id)}</span>`],
    ["Programs", e.programs.map((id) => `<span class="pill dot" style="--pill-color:${programColor(id)}">${escapeHtml(titleForProgram(id))}</span>`).join(" ")],
    ["Status", `<span class="status-tag ${e.anchor_ready === "yes" ? "ready" : "notready"}">${e.anchor_ready === "yes" ? "anchor-ready" : "needs curation"}</span>`],
    ["Run surface", e.run_surface ? `<span class="mono-val">${escapeHtml(e.run_surface)}</span>` : ""],
    ["Smoke command", e.smoke_command ? `<code>${escapeHtml(e.smoke_command)}</code>` : ""],
    ["Needs", (e.needs || []).length ? (e.needs || []).map((n) => `<span class="pill">${escapeHtml(n)}</span>`).join(" ") : "none"],
    ["Tags", (e.tags || []).length ? (e.tags || []).map((t) => `<span class="pill">${escapeHtml(t)}</span>`).join(" ") : ""],
    ["Artifacts", listText(e.recognized_artifacts) ? escapeHtml(listText(e.recognized_artifacts)) : ""],
    ["Files", `${formatNumber(e.total_files)} · ${formatBytes(e.total_size_bytes).n} ${formatBytes(e.total_size_bytes).unit}`]
  ], `<div class="drawer-section-label">Open</div><div style="display:flex;gap:16px;flex-wrap:wrap">
      ${e.primary_readme ? `<a class="text-link" href="${repoLink(e.primary_readme)}">README <span class="arr">→</span></a>` : ""}
      ${e.primary_report ? `<a class="text-link" href="${repoLink(e.primary_report)}">Report <span class="arr">→</span></a>` : ""}
      <a class="text-link" href="${repoLink(e.path)}">Folder <span class="arr">→</span></a>
    </div>`);
}

/* ---------------------------------------------------------------- queue board */
const QUEUE_META = {
  P0: { color: COLORS.rose, meaning: "do next" },
  P1: { color: COLORS.amber, meaning: "soon" },
  P2: { color: COLORS.blue, meaning: "later" }
};
function renderQueue() {
  const rows = filteredQueue();
  document.getElementById("queueCount").innerHTML = `<b>${formatNumber(rows.length)}</b> of ${formatNumber(DATA.queue.length)} shown`;
  const priorities = ["P0", "P1", "P2"];
  document.getElementById("queueBoard").innerHTML = priorities.map((pr) => {
    const items = rows.filter((i) => i.priority === pr);
    const meta = QUEUE_META[pr] || { color: COLORS.muted, meaning: "" };
    const cards = items.length ? items.map((item) => `
      <article class="queue-card" data-queue="${escapeHtml(item.id)}" style="border-left:3px solid ${meta.color}">
        <div class="meta-line"><span>${escapeHtml(item.status)}</span><span class="effort-chip">${escapeHtml(item.effort)}</span></div>
        <h3>${escapeHtml(item.title)}</h3>
        <p class="question">${escapeHtml(item.question)}</p>
        <div class="pill-row">${(item.programs || []).slice(0, 3).map((id) => `<span class="pill dot" style="--pill-color:${programColor(id)}">${escapeHtml(titleForProgram(id))}</span>`).join("")}</div>
      </article>`).join("") : '<div class="empty-state" style="padding:20px 8px">No matching proposals.</div>';
    return `<div class="queue-column" style="--col-color:${meta.color}">
      <div class="queue-col-head"><h3>${pr}</h3><span class="count">${items.length}</span><span class="meaning">${meta.meaning}</span></div>
      ${cards}
    </div>`;
  }).join("");
  document.querySelectorAll("[data-queue]").forEach((card) => {
    card.addEventListener("click", () => openQueue(DATA.queue.find((i) => i.id === card.getAttribute("data-queue"))));
  });
}

function openQueue(item) {
  if (!item) return;
  const meta = QUEUE_META[item.priority] || {};
  openDrawer(`Queue · ${item.priority} ${meta.meaning ? "(" + meta.meaning + ")" : ""}`, item.title, item.question, [
    ["Queue id", `<span class="mono-val">${escapeHtml(item.id)}</span>`],
    ["Priority", `${escapeHtml(item.priority)} · ${escapeHtml(item.status)} · ${escapeHtml(item.effort)} effort`],
    ["Programs", (item.programs || []).map((id) => `<span class="pill dot" style="--pill-color:${programColor(id)}">${escapeHtml(titleForProgram(id))}</span>`).join(" ")],
    ["Hypothesis", escapeHtml(item.hypothesis)],
    ["Minimal protocol", escapeHtml(item.minimal_protocol)],
    ["Success signal", escapeHtml(item.success_signal)],
    ["Failure signal", escapeHtml(item.failure_signal)],
    ["Expected artifacts", listText(item.expected_artifacts) ? escapeHtml(listText(item.expected_artifacts)) : ""],
    ["Next step", escapeHtml(item.next_step)]
  ], item.source ? `<div class="drawer-section-label">Open</div><a class="text-link" href="${repoLink(item.source)}">Source on GitHub <span class="arr">→</span></a>` : "");
}

/* ---------------------------------------------------------------- claims */
function renderClaims() {
  const claims = filteredClaims();
  const host = document.getElementById("claimCards");
  if (!claims.length) { host.innerHTML = '<div class="empty-state">No claims match this filter.</div>'; return; }
  host.innerHTML = claims.map((c) => {
    const color = STATUS_COLORS[c.status] || COLORS.violet;
    return `<article class="claim-card" data-claim="${escapeHtml(c.id)}" style="--accent:${color}">
      <div class="claim-top">
        <span class="claim-id">${escapeHtml(c.id)}</span>
        <span class="status-chip" style="--status-color:${color}">${escapeHtml(c.status)}</span>
      </div>
      <h3>${escapeHtml(c.title)}</h3>
      <p class="claim-summary">${escapeHtml(c.summary)}</p>
      <div class="pill-row">${splitList(c.programs).map((id) => `<span class="pill dot" style="--pill-color:${programColor(id)}">${escapeHtml(titleForProgram(id))}</span>`).join("")}</div>
    </article>`;
  }).join("");
  host.querySelectorAll("[data-claim]").forEach((card) => {
    card.addEventListener("click", () => openClaim(DATA.claims.find((c) => c.id === card.getAttribute("data-claim"))));
  });
}

function openClaim(c) {
  if (!c) return;
  const color = STATUS_COLORS[c.status] || COLORS.violet;
  openDrawer(`Claim ${c.id}`, c.title, c.summary, [
    ["Status", `<span class="status-chip" style="--status-color:${color}">${escapeHtml(c.status)}</span>`],
    ["Programs", splitList(c.programs).map((id) => `<span class="pill dot" style="--pill-color:${programColor(id)}">${escapeHtml(titleForProgram(id))}</span>`).join(" ")],
    ["Evidence", escapeHtml(c.evidence)],
    ["Implication", escapeHtml(c.implication)]
  ], `<div class="drawer-section-label">Open</div><a class="text-link" href="${repoLink("knowledge/claims/index.md")}">Claim ledger on GitHub <span class="arr">→</span></a>`);
}

/* ---------------------------------------------------------------- narrative */
function renderNarrative() {
  const items = [
    ["Synthesis", DATA.narrative.synthesis, "knowledge/synthesis.md"],
    ["Roadmap", DATA.narrative.roadmap, "knowledge/research_roadmap.md"],
    ["Patterns", DATA.narrative.patterns, "knowledge/patterns.md"]
  ];
  document.getElementById("narrativeCards").innerHTML = items.map(([title, body, path]) => `
    <article class="narrative-card">
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(body || "Open the source document for current notes.")}</p>
      <a class="text-link" href="${repoLink(path)}">Open source <span class="arr">→</span></a>
    </article>`).join("");
}

/* ---------------------------------------------------------------- charts orchestration */
function renderCharts() {
  programNetwork();
  donutChart();
  scatterPlot();
  programHeatmap();
  claimGraph();
  barChart("needsChart", DATA.charts.needs, { label: "curation needs", labelW: 170, color: COLORS.rose });
  barChart("queuePriorityChart", DATA.charts.queue_priority, {
    label: "queue priorities", labelW: 50, rowH: 34,
    colors: { P0: COLORS.rose, P1: COLORS.amber, P2: COLORS.blue }
  });
  barChart("runSurfaceChart", DATA.charts.run_surfaces, { label: "run surfaces", labelW: 160, rowH: 30, color: COLORS.teal });
  barChart("artifactKindChart", DATA.charts.artifact_kinds, { label: "artifact manifest kinds", labelW: 120, color: COLORS.violet });
  barChart("extensionChart", DATA.charts.extensions, { label: "file extension counts", labelW: 70, rowH: 24, color: COLORS.blue });
}

/* ---------------------------------------------------------------- render */
function render() {
  updateFilterStatus();
  renderPrograms();
  renderExperiments();
  renderQueue();
  renderClaims();
  renderCharts();
}

/* ---------------------------------------------------------------- scroll spy */
function setupScrollSpy() {
  const links = Array.from(document.querySelectorAll(".topnav a"));
  const map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const sections = Array.from(document.querySelectorAll("main section[id]"));
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        links.forEach((a) => a.classList.remove("active"));
        const link = map.get(entry.target.id);
        if (link) link.classList.add("active");
      }
    });
  }, { rootMargin: "-45% 0px -50% 0px", threshold: 0 });
  sections.forEach((s) => obs.observe(s));
}

/* ---------------------------------------------------------------- events */
function resetFilters() {
  state.search = ""; state.program = "all"; state.ready = "all"; state.need = "all";
  els.search.value = ""; els.program.value = "all"; els.ready.value = "all"; els.need.value = "all";
  render();
}

function bindEvents() {
  els.search.addEventListener("input", (e) => { state.search = e.target.value.trim(); render(); });
  els.program.addEventListener("change", (e) => { state.program = e.target.value; render(); });
  els.ready.addEventListener("change", (e) => { state.ready = e.target.value; render(); });
  els.need.addEventListener("change", (e) => { state.need = e.target.value; render(); });
  els.reset.addEventListener("click", resetFilters);
  els.drawerClose.addEventListener("click", closeDrawer);
  els.drawerScrim.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDrawer();
    if (e.key === "/" && document.activeElement !== els.search) { e.preventDefault(); els.search.focus(); }
  });
}

renderHero();
renderFilters();
renderMetrics();
renderNarrative();
bindEvents();
setupScrollSpy();
render();

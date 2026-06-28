"use strict";

const DATA = JSON.parse(document.getElementById("site-data").textContent);

const COLORS = {
  teal: "#0c7a84",
  blue: "#2b5fae",
  green: "#1f8a4d",
  amber: "#b9720f",
  amberText: "#8a5400",
  rose: "#c0344a",
  violet: "#6a4fbb",
  ink: "#111a24",
  inkInv: "#f3f6fa",
  muted: "#586676",
  faint: "#5c6979",
  faintInv: "#9aa7b8",
  line: "#d3dae3",
  lineSoft: "#e4e9ef",
  lineInv: "#39475a",
  surface: "#ffffff",
  tint: "#f6f8fb"
};

/* Status colors are the SATURATED signal set (green/amber/rose/blue + retired gray).
   The program palette below is a muted CATEGORICAL set deliberately distinct from these,
   so a hue never means both "a status" and "a program". */
const STATUS_COLORS = {
  Confirmed: COLORS.green,
  Promising: COLORS.amber,
  Negative: COLORS.rose,
  Open: COLORS.blue,
  Retired: "#7c8794"
};

/* 11 muted, mutually distinct categorical hues for programs (lower saturation than the
   status signals; identity is reinforced by node initials + the legend, not color alone). */
const PROGRAM_PALETTE = [
  "#3d6fa6", "#2a8c82", "#6d5bb0", "#b3578a", "#a8762f", "#5e9440",
  "#3f93a8", "#a7644a", "#7d8a36", "#9c4f9c", "#566b86"
];

/* Lowercase minor words for display title-case; fix a few corpus spellings. */
const MINOR_WORDS = new Set(["and", "or", "for", "of", "to", "the", "a", "an", "in", "on", "vs", "with"]);
function humanizeTitle(value) {
  return String(value || "")
    .replace(/\bPosttraining\b/g, "Post-training")
    .split(/(\s+)/)
    .map((word, i) => {
      if (/^\s+$/.test(word)) return word;
      const lower = word.toLowerCase();
      if (i > 0 && MINOR_WORDS.has(lower)) return lower;
      return word;
    })
    .join("");
}

/* Map raw kebab/snake tokens to readable phrases for end users. */
const TOKEN_LABELS = {
  "add-smoke-command": "Add smoke-test command",
  "add-experiment-log": "Add experiment log",
  "document-run-path": "Document run path",
  "add-artifact-manifest": "Add artifact manifest",
  "replace-generated-readme": "Replace generated README",
  "review-program-assignment": "Review program assignment",
  "source-or-analysis": "Source / analysis only",
  "documented-scripts": "Documented scripts",
  "documented-command": "Documented run command",
  "scripts-undocumented": "Scripts, undocumented"
};
function humanizeToken(value) {
  if (!value) return value;
  if (TOKEN_LABELS[value]) return TOKEN_LABELS[value];
  return String(value).replace(/[-_]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
function humanizeStatus(value) {
  if (!value) return value;
  const s = String(value).toLowerCase().replace(/[-_]/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/* Short program code (initials) for compact contexts like map nodes. */
function programCode(p) {
  const words = humanizeTitle(p.title).split(/\s+/).filter((w) => !MINOR_WORDS.has(w.toLowerCase()));
  return words.slice(0, 2).map((w) => w[0]).join("").toUpperCase() || p.id.slice(0, 2).toUpperCase();
}

const state = { search: "", program: "all", ready: "all", need: "all", sort: { key: "id", dir: 1 } };

const els = {
  search: document.getElementById("globalSearch"),
  program: document.getElementById("programFilter"),
  ready: document.getElementById("readyFilter"),
  need: document.getElementById("needFilter"),
  reset: document.getElementById("resetFilters"),
  searchTop: document.getElementById("globalSearchTop"),
  programTop: document.getElementById("programFilterTop"),
  readyTop: document.getElementById("readyFilterTop"),
  needTop: document.getElementById("needFilterTop"),
  resetTop: document.getElementById("resetFiltersTop"),
  filterRail: document.getElementById("filterRail"),
  filterChips: document.getElementById("filterChips"),
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
/* one canonical program order (by size) shared across map, legend, heatmap, claim matrix */
const PROGRAMS_BY_SIZE = DATA.programs.slice().sort((a, b) => b.experiment_count - a.experiment_count);

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
function titleForProgram(id) {
  const p = programById.get(id);
  return p ? humanizeTitle(p.title) : humanizeToken(id); // unknown ids (e.g. candidate programs) -> readable
}
function programPill(id) {
  return `<button class="pill dot pill-link" type="button" data-program-pill="${escapeHtml(id)}" style="--pill-color:${programColor(id)}" title="Filter to ${escapeHtml(titleForProgram(id))}">${escapeHtml(titleForProgram(id))}</button>`;
}
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

/* Add click + Enter/Space activation to a non-button interactive element. */
function onActivate(el, fn) {
  el.addEventListener("click", fn);
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fn(e); }
  });
}

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
  // keyboard/touch parity: show on focus relative to the element's box
  el.addEventListener("focus", () => {
    const r = el.getBoundingClientRect();
    showTooltip({ clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 }, html);
  });
  el.addEventListener("blur", hideTooltip);
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
/* Per-program {count, ready} computed from a given experiment set (for filter-reactive overview). */
function programStats(experiments) {
  const m = new Map(DATA.programs.map((p) => [p.id, { count: 0, ready: 0 }]));
  experiments.forEach((e) => (e.programs || []).forEach((id) => {
    const s = m.get(id);
    if (s) { s.count += 1; if (e.anchor_ready === "yes") s.ready += 1; }
  }));
  return m;
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

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function syncControls() {
  [els.program, els.programTop].forEach((c) => c && (c.value = state.program));
  [els.ready, els.readyTop].forEach((c) => c && (c.value = state.ready));
  [els.need, els.needTop].forEach((c) => c && (c.value = state.need));
  [els.search, els.searchTop].forEach((c) => c && (c.value = state.search));
}

function setProgram(id) {
  state.program = state.program === id ? "all" : id;
  syncControls();
  render();
}
function setReady(value) { state.ready = value; syncControls(); render(); }
function setNeed(value) { state.need = value; syncControls(); render(); }
function setSearch(value) { state.search = value; syncControls(); render(); }

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
  els.drawer.removeAttribute("inert");
  els.drawerScrim.classList.add("open");
  els.drawer.setAttribute("aria-hidden", "false");
  setBackgroundInert(true);
  els.drawerClose.focus();
}
function closeDrawer() {
  if (!els.drawer.classList.contains("open")) return;
  els.drawer.classList.remove("open");
  els.drawer.setAttribute("inert", "");
  els.drawerScrim.classList.remove("open");
  els.drawer.setAttribute("aria-hidden", "true");
  setBackgroundInert(false);
  if (lastFocused && lastFocused.focus) lastFocused.focus();
}
function setBackgroundInert(on) {
  ["main", ".topbar", ".site-footer", "#filterRail"].forEach((sel) => {
    const node = document.querySelector(sel);
    if (node) { if (on) node.setAttribute("inert", ""); else node.removeAttribute("inert"); }
  });
}
function trapDrawerFocus(event) {
  if (event.key !== "Tab" || !els.drawer.classList.contains("open")) return;
  const focusables = els.drawer.querySelectorAll('a[href], button:not([disabled]), input, select, [tabindex]:not([tabindex="-1"])');
  if (!focusables.length) return;
  const first = focusables[0], last = focusables[focusables.length - 1];
  if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
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
  const programOpts = [
    '<option value="all">All programs</option>',
    ...DATA.programs.map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(titleForProgram(p.id))}</option>`)
  ].join("");
  const needs = Array.from(new Set(DATA.experiments.flatMap((e) => e.needs || []))).sort();
  const needOpts = [
    '<option value="all">All needs</option>',
    ...needs.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(humanizeToken(n))}</option>`)
  ].join("");
  [els.program, els.programTop].forEach((c) => c && (c.innerHTML = programOpts));
  [els.need, els.needTop].forEach((c) => c && (c.innerHTML = needOpts));
}

function activeChips() {
  const chips = [];
  if (state.program !== "all") chips.push({ label: titleForProgram(state.program), clear: () => setProgram(state.program) });
  if (state.ready === "yes") chips.push({ label: "Anchor-ready", clear: () => setReady("all") });
  if (state.ready === "no") chips.push({ label: "Needs curation", clear: () => setReady("all") });
  if (state.need !== "all") chips.push({ label: humanizeToken(state.need), clear: () => setNeed("all") });
  if (state.search) chips.push({ label: `“${state.search}”`, clear: () => setSearch("") });
  return chips;
}

function chipMarkup(chips, withClearAll) {
  return chips
    .map((c, i) => `<button class="filter-chip" type="button" data-chip="${i}">${escapeHtml(c.label)}<span class="chip-x" aria-hidden="true">✕</span><span class="sr-only"> — remove filter</span></button>`)
    .join("") + (withClearAll ? '<button class="filter-chip clear-all" type="button" data-chip="reset">Clear all</button>' : "");
}
function wireChips(host, chips) {
  host.querySelectorAll("[data-chip]").forEach((b) => {
    const key = b.getAttribute("data-chip");
    b.addEventListener("click", () => (key === "reset" ? resetFilters() : chips[Number(key)].clear()));
  });
}

function updateFilterStatus() {
  const chips = activeChips();
  const active = chips.length > 0;
  els.filterStatus.textContent = active
    ? "Scoping the atlas — remove a chip to widen."
    : "Filters scope every panel below.";
  if (els.reset) els.reset.hidden = !active;
  // rail chips (with clear-all) and hero chips (compact)
  if (els.filterChips) {
    els.filterChips.innerHTML = active ? chipMarkup(chips, true) : '<span class="chips-empty">No filters active</span>';
    wireChips(els.filterChips, chips);
  }
  const heroChips = document.getElementById("filterChipsHero");
  if (heroChips) {
    heroChips.innerHTML = active ? chipMarkup(chips, false) : "";
    wireChips(heroChips, chips);
  }
}

/* ---------------------------------------------------------------- metrics */
function gotoSection(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth" });
}

const METRICS = [
  { label: "Experiments", key: "experiments", help: "Self-contained experiment folders", color: COLORS.teal, action: () => gotoSection("experiments") },
  { label: "Programs", key: "programs", help: "Durable lines of inquiry", color: COLORS.ink, action: () => gotoSection("programs") },
  { label: "Anchor-ready", key: "anchor_ready", help: "Citeable and reusable as-is", color: COLORS.green, action: () => { setReady("yes"); gotoSection("readiness"); } },
  { label: "Needs curation", key: "needs_curation", help: "Not yet anchor-ready", color: COLORS.amber, action: () => { setReady("no"); gotoSection("readiness"); } },
  { label: "Future queue", key: "future_proposals", help: "Structured next probes", color: COLORS.blue, action: () => gotoSection("queue") },
  { label: "Claims", key: "claims", help: "Shared evidence statements", color: COLORS.violet, action: () => gotoSection("claims") },
  { label: "Files indexed", key: "total_files", help: "Tracked corpus files", color: COLORS.ink, action: () => gotoSection("artifacts") },
  { label: "Tracked size", key: "total_size_bytes", help: "Repository-local footprint", color: COLORS.ink, isBytes: true, action: () => gotoSection("artifacts") }
];

function renderMetrics() {
  const s = DATA.summary;
  document.getElementById("metricGrid").innerHTML = METRICS
    .map((m, i) => {
      const raw = s[m.key];
      const value = m.isBytes ? formatBytes(raw).n : formatNumber(raw);
      const unit = m.isBytes ? formatBytes(raw).unit : "";
      return `
      <button class="metric-card" type="button" data-metric="${i}" style="--accent:${m.color};--i:${i}">
        <span class="eyebrow">${escapeHtml(m.label)}</span>
        <span class="metric-value">${escapeHtml(value)}${unit ? `<span class="unit">${escapeHtml(unit)}</span>` : ""}</span>
        <span class="metric-help">${escapeHtml(m.help)}<span class="metric-go" aria-hidden="true">→</span></span>
      </button>`;
    })
    .join("");
  document.querySelectorAll("[data-metric]").forEach((card) => {
    card.addEventListener("click", () => METRICS[Number(card.getAttribute("data-metric"))].action());
  });
}

/* ---------------------------------------------------------------- bar chart */
function barChart(targetId, rows, options = {}) {
  const target = document.getElementById(targetId);
  clear(target);
  if (!rows.length) { target.innerHTML = `<div class="empty-state" style="padding:24px 16px">${escapeHtml(options.empty || "No data.")}</div>`; return; }
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
function programNetwork(fstats, fcount) {
  const target = document.getElementById("programNetwork");
  clear(target);
  const stats = fstats || programStats(DATA.experiments);
  const W = 760, H = 446, cx = W / 2, cy = H / 2 - 6, ringR = 156;
  const progs = PROGRAMS_BY_SIZE;
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Corpus map: research programs sized by experiments, ring shows percent anchor-ready" });

  [ringR + 46, ringR, ringR - 56].forEach((r) => {
    chart.appendChild(svg("circle", { cx, cy, r, fill: "none", stroke: COLORS.lineInv, "stroke-width": 1, opacity: 0.5 }));
  });

  const maxCount = Math.max(1, ...progs.map((p) => p.experiment_count));
  const nodeR = (c) => 16 + (Math.sqrt(c) / Math.sqrt(maxCount)) * 24;
  const filtering = state.program !== "all" || state.ready !== "all" || state.need !== "all" || !!state.search;

  progs.forEach((p, i) => {
    const angle = -Math.PI / 2 + (i / progs.length) * Math.PI * 2;
    const x = cx + Math.cos(angle) * ringR, y = cy + Math.sin(angle) * ringR;
    const inView = (stats.get(p.id)?.count || 0) > 0;
    const active = state.program === p.id;
    chart.appendChild(svg("line", {
      x1: cx, y1: cy, x2: x, y2: y,
      stroke: active ? programColor(p.id) : COLORS.lineInv,
      "stroke-width": active ? 2.4 : 1 + (p.experiment_count / maxCount) * 1.4,
      opacity: filtering && !inView ? 0.12 : active ? 0.9 : 0.5
    }));
  });

  chart.appendChild(svg("circle", { cx, cy, r: 46, fill: "rgba(255,255,255,0.04)", stroke: COLORS.lineInv, "stroke-width": 1.5 }));
  chart.appendChild(text(formatNumber(fcount == null ? DATA.summary.experiments : fcount), { x: cx, y: cy - 1, "text-anchor": "middle", class: "chart-value-inv", style: "font-size:23px;font-family:var(--display)" }));
  chart.appendChild(text(filtering ? "IN VIEW" : "EXPERIMENTS", { x: cx, y: cy + 16, "text-anchor": "middle", class: "axis-label-inv" }));

  progs.forEach((p, i) => {
    const angle = -Math.PI / 2 + (i / progs.length) * Math.PI * 2;
    const x = cx + Math.cos(angle) * ringR, y = cy + Math.sin(angle) * ringR;
    const r = nodeR(p.experiment_count);
    const color = programColor(p.id);
    const active = state.program === p.id;
    const s = stats.get(p.id) || { count: 0, ready: 0 };
    const inView = s.count > 0;
    const ratio = s.count ? s.ready / s.count : 0;
    const opacity = filtering && !inView ? 0.22 : 1;
    const g = svg("g", { tabindex: 0, role: "button", class: "map-node",
      "aria-label": `${titleForProgram(p.id)}: ${s.count} experiments, ${Math.round(ratio * 100)} percent anchor-ready. Activate to filter.`,
      style: `cursor:pointer; opacity:${opacity}; --i:${i}` });
    g.appendChild(svg("circle", { cx: x, cy: y, r, fill: color, stroke: active ? "#fff" : "rgba(255,255,255,0.25)", "stroke-width": active ? 3 : 1.5 }));
    const rr = r + 5, circ = 2 * Math.PI * rr;
    g.appendChild(svg("circle", { cx: x, cy: y, r: rr, fill: "none", stroke: "#d99a3a", "stroke-width": 3, opacity: 0.8 }));
    g.appendChild(svg("circle", {
      cx: x, cy: y, r: rr, fill: "none", stroke: "#46d39a", "stroke-width": 3,
      "stroke-dasharray": `${ratio * circ} ${circ}`, transform: `rotate(-90 ${x} ${y})`
    }));
    g.appendChild(text(programCode(p), { x, y: y + 4, "text-anchor": "middle", class: "node-code", style: `font-size:${r > 26 ? 13 : 11}px` }));
    attachTooltip(g, `<strong>${escapeHtml(titleForProgram(p.id))}</strong><br><span class="tip-meta">${s.count} experiments · ${Math.round(ratio * 100)}% ready · ${p.queue_count} queued · ${p.claim_count} claims</span>`);
    onActivate(g, () => setProgram(p.id));
    chart.appendChild(g);
  });

  chart.appendChild(text("disk = experiments    green ring = % anchor-ready, amber = to curate", { x: cx, y: H - 6, "text-anchor": "middle", class: "axis-label-inv" }));

  target.appendChild(chart);
  renderProgramLegend(stats);
}

function renderProgramLegend(stats) {
  const legend = document.getElementById("programLegend");
  const filtering = state.program !== "all" || state.ready !== "all" || state.need !== "all" || !!state.search;
  legend.innerHTML = PROGRAMS_BY_SIZE
    .map((p) => {
      const active = state.program === p.id;
      const count = stats ? (stats.get(p.id)?.count ?? p.experiment_count) : p.experiment_count;
      const dim = filtering && !active && count === 0;
      return `<button class="legend-item${dim ? " dim" : ""}${active ? " active" : ""}" data-program="${escapeHtml(p.id)}" type="button" aria-pressed="${active}" title="Filter to ${escapeHtml(titleForProgram(p.id))}">
        <span class="legend-code" style="background:${programColor(p.id)}">${escapeHtml(programCode(p))}</span>${escapeHtml(titleForProgram(p.id))} <b>${formatNumber(count)}</b>
      </button>`;
    })
    .join("");
  legend.querySelectorAll("[data-program]").forEach((b) => {
    b.addEventListener("click", () => setProgram(b.getAttribute("data-program")));
  });
}

/* ---------------------------------------------------------------- readiness donut */
function donutChart(experiments) {
  const target = document.getElementById("readinessDonut");
  clear(target);
  const exp = experiments || DATA.experiments;
  const total = exp.length || 1;
  const ready = exp.filter((e) => e.anchor_ready === "yes").length;
  const needs = Math.max(0, total - ready);
  const pct = Math.round((ready / total) * 100);
  const W = 260, H = 232, cx = W / 2, cy = 104, radius = 72;
  const circ = 2 * Math.PI * radius;
  const readyArc = (ready / total) * circ;
  const track = "#c8d0da"; // visible "to curate" tone (was invisible lineSoft)
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${pct}% anchor-ready in this view (${ready} of ${total})` });
  chart.appendChild(svg("circle", { cx, cy, r: radius, fill: "none", stroke: track, "stroke-width": 22 }));
  const arc = svg("circle", {
    cx, cy, r: radius, fill: "none", stroke: COLORS.green, "stroke-width": 22, class: "donut-arc",
    "stroke-dasharray": `${readyArc} ${circ - readyArc}`, transform: `rotate(-90 ${cx} ${cy})`,
    style: `--dash:${readyArc}`
  });
  chart.appendChild(arc);
  chart.appendChild(text(`${pct}%`, { x: cx, y: cy + 2, "text-anchor": "middle", class: "chart-value", style: "font-size:34px;font-family:var(--display)" }));
  chart.appendChild(text("anchor-ready", { x: cx, y: cy + 22, "text-anchor": "middle", class: "axis-label" }));
  chart.appendChild(svg("rect", { x: 30, y: H - 20, width: 10, height: 10, rx: 2, fill: COLORS.green }));
  chart.appendChild(text(`${formatNumber(ready)} ready`, { x: 46, y: H - 11, class: "chart-label" }));
  chart.appendChild(svg("rect", { x: 144, y: H - 20, width: 10, height: 10, rx: 2, fill: track }));
  chart.appendChild(text(`${formatNumber(needs)} to curate`, { x: 160, y: H - 11, class: "chart-label" }));
  target.appendChild(chart);
}

/* ---------------------------------------------------------------- scatter -> files skyline */
function scatterPlot() {
  const target = document.getElementById("experimentScatter");
  clear(target);
  const rows = filteredExperiments().slice().sort((a, b) => b.total_files - a.total_files);
  if (!rows.length) { target.innerHTML = '<div class="empty-state" style="padding:24px 16px">No experiments in this view.</div>'; return; }
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
    const bar = svg("rect", { x, y: H - bottom - h, width: bw, height: h, rx: bw > 3 ? 1.5 : 0, fill: color, opacity: 0.85,
      tabindex: 0, role: "button", "aria-label": `${e.id}: ${e.total_files} files, ${e.anchor_ready === "yes" ? "ready" : "needs curation"}`, style: "cursor:pointer" });
    // widen the hit/focus target for hairline bars
    const hit = svg("rect", { x: x - 1, y: top, width: Math.max(bw + 2, 4), height: plotH, fill: "transparent", style: "cursor:pointer" });
    attachTooltip(bar, `<strong>${escapeHtml(e.id)}</strong><br><span class="tip-meta">${formatNumber(e.total_files)} files · ${e.anchor_ready === "yes" ? "ready" : "needs curation"}</span>`);
    onActivate(bar, () => openExperiment(e));
    hit.addEventListener("click", () => openExperiment(e));
    chart.appendChild(hit);
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
/* Approximate text width for IBM Plex Mono so labels never overprint cells (offline-safe). */
function ellipsize(str, maxWidth, fontPx) {
  const adv = fontPx * 0.605; // mono advance ratio
  const maxChars = Math.floor(maxWidth / adv);
  if (str.length <= maxChars) return str;
  return str.slice(0, Math.max(1, maxChars - 1)).trimEnd() + "…";
}

/* choose ink/white text by the luminance of a hex fill blended with white at `alpha` */
function readableText(hex, alpha) {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
  const mix = (c) => c * alpha + 255 * (1 - alpha);
  const lum = (0.2126 * mix(r) + 0.7152 * mix(g) + 0.0722 * mix(b)) / 255;
  return lum < 0.56 ? "#ffffff" : COLORS.ink;
}

function programHeatmap(fstats) {
  const target = document.getElementById("programHeatmap");
  clear(target);
  const stats = fstats || programStats(DATA.experiments);
  // Exp + %Ready reflect the filtered view; Queued/Claims are program-level attributes.
  const metrics = [
    { key: "exp", label: "Exp", color: "#566b86", mode: "count" },
    { key: "ready", label: "% Ready", color: COLORS.green, mode: "ratio" },
    { key: "queue_count", label: "Queued", color: COLORS.amber, mode: "count" },
    { key: "claim_count", label: "Claims", color: COLORS.violet, mode: "count" }
  ];
  const W = 620, labelW = 270, gutter = 12, fontPx = 9.5;
  const cellW = (W - labelW) / metrics.length, rowH = 30, top = 32;
  const progs = PROGRAMS_BY_SIZE;
  const rowOf = (p) => { const s = stats.get(p.id) || { count: 0, ready: 0 }; return { exp: s.count, ready: s.ready, queue_count: p.queue_count, claim_count: p.claim_count, expTotal: s.count }; };
  const data = progs.map((p) => ({ p, v: rowOf(p) }));
  const H = top + progs.length * rowH + 6;
  const maxes = {};
  metrics.forEach((m) => { maxes[m.key] = Math.max(1, ...data.map((d) => d.v[m.key])); });
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Program scorecard: experiments, percent ready, queued, claims", preserveAspectRatio: "xMinYMin meet" });
  metrics.forEach((m, ci) => {
    chart.appendChild(text(m.label, { x: labelW + ci * cellW + cellW / 2, y: 18, "text-anchor": "middle", class: "axis-label" }));
  });
  data.forEach(({ p, v }, ri) => {
    const y = top + ri * rowH;
    const active = state.program === p.id;
    if (active) chart.appendChild(svg("rect", { x: 0, y: y - 1, width: W, height: rowH, fill: COLORS.tint, rx: 5 }));
    chart.appendChild(svg("rect", { x: 0, y: y + rowH / 2 - 7, width: 5, height: 14, rx: 2, fill: programColor(p.id) }));
    const full = titleForProgram(p.id);
    const label = text(ellipsize(full, labelW - gutter - 14, fontPx), { x: 12, y: y + rowH / 2 + 3.5, class: "chart-label", style: `font-size:${fontPx}px` });
    label.appendChild(svg("title", {}, [document.createTextNode(full)]));
    chart.appendChild(label);
    metrics.forEach((m, ci) => {
      const value = v[m.key];
      const ratio = m.mode === "ratio" ? value / Math.max(1, v.expTotal) : value / maxes[m.key];
      const alpha = 0.1 + ratio * 0.8;
      const cellX = labelW + ci * cellW;
      const cell = svg("rect", {
        x: cellX + 3, y: y + 3, width: cellW - 6, height: rowH - 6, rx: 5,
        fill: m.color, opacity: alpha, tabindex: 0, role: "button",
        "aria-label": `${full}, ${m.label}: ${m.mode === "ratio" ? Math.round(ratio * 100) + " percent" : value}. Activate to filter.`,
        style: "cursor:pointer"
      });
      const valLabel = m.mode === "ratio" ? `${Math.round(ratio * 100)}% (${formatNumber(value)}/${formatNumber(v.expTotal)})` : formatNumber(value);
      attachTooltip(cell, `<strong>${escapeHtml(full)}</strong><br><span class="tip-meta">${m.label}: ${valLabel}</span>`);
      onActivate(cell, () => setProgram(p.id));
      chart.appendChild(cell);
      const shown = m.mode === "ratio" ? (v.expTotal ? `${Math.round(ratio * 100)}%` : "—") : formatNumber(value);
      chart.appendChild(text(shown, {
        x: cellX + cellW / 2, y: y + rowH / 2 + 4, "text-anchor": "middle",
        class: "chart-value", style: `fill:${readableText(m.color, alpha)}`
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
  const progs = PROGRAMS_BY_SIZE;
  if (!claims.length) { target.innerHTML = '<div class="empty-state">No claims yet.</div>'; return; }
  const W = 620, labelW = 270, gutter = 12, fontPx = 9.5, top = 56, rowH = 30, colW = (W - labelW) / claims.length;
  const H = top + progs.length * rowH + 8;
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Claims linked to programs matrix", preserveAspectRatio: "xMinYMin meet" });
  // column headers (claim ids, status dot above)
  claims.forEach((c, ci) => {
    const x = labelW + ci * colW + colW / 2;
    chart.appendChild(svg("circle", { cx: x, cy: 18, r: 6, fill: STATUS_COLORS[c.status] || COLORS.violet }));
    const t = text(c.id, { x, y: 44, "text-anchor": "middle", class: "chart-value",
      tabindex: 0, role: "button", "aria-label": `Claim ${c.id}: ${c.title}, ${humanizeStatus(c.status)}`, style: "cursor:pointer" });
    onActivate(t, () => openClaim(c));
    attachTooltip(t, `<strong>${escapeHtml(c.id)}: ${escapeHtml(c.title)}</strong><br><span class="tip-meta">${escapeHtml(humanizeStatus(c.status))}</span>`);
    chart.appendChild(t);
  });
  progs.forEach((p, ri) => {
    const y = top + ri * rowH;
    chart.appendChild(svg("rect", { x: 0, y: y + rowH / 2 - 7, width: 5, height: 14, rx: 2, fill: programColor(p.id) }));
    const full = titleForProgram(p.id);
    const lbl = text(ellipsize(full, labelW - gutter - 14, fontPx), { x: 12, y: y + rowH / 2 + 3.5, class: "chart-label", style: `font-size:${fontPx}px` });
    lbl.appendChild(svg("title", {}, [document.createTextNode(full)]));
    chart.appendChild(lbl);
    claims.forEach((c, ci) => {
      const linked = splitList(c.programs).includes(p.id);
      const x = labelW + ci * colW + colW / 2;
      if (linked) {
        const dot = svg("circle", { cx: x, cy: y + rowH / 2, r: 6.5, fill: STATUS_COLORS[c.status] || COLORS.violet,
          tabindex: 0, role: "button", "aria-label": `${c.id} relates to ${titleForProgram(p.id)}: ${c.title}`, style: "cursor:pointer" });
        attachTooltip(dot, `<strong>${escapeHtml(c.id)}</strong> ↔ <strong>${escapeHtml(titleForProgram(p.id))}</strong><br><span class="tip-meta">${escapeHtml(c.title)}</span>`);
        onActivate(dot, () => openClaim(c));
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
  host.innerHTML = rows.map((p) => {
    const active = state.program === p.id;
    return `
    <article class="program-card${active ? " active" : ""}" tabindex="0" role="button" aria-pressed="${active}"
      aria-label="Filter the atlas to ${escapeHtml(titleForProgram(p.id))}" data-program="${escapeHtml(p.id)}" style="--accent:${programColor(p.id)}">
      <div class="card-head">
        <span class="card-code" style="background:${programColor(p.id)}">${escapeHtml(programCode(p))}</span>
        <h4>${escapeHtml(titleForProgram(p.id))}</h4>
      </div>
      <p class="focus">${escapeHtml(p.focus)}</p>
      <div class="stat-row">
        <div class="stat"><span class="n">${formatNumber(p.experiment_count)}</span><span class="k">exp</span></div>
        <div class="stat"><span class="n" style="color:${COLORS.green}">${formatNumber(p.anchor_ready_count)}</span><span class="k">ready</span></div>
        <div class="stat"><span class="n" style="color:${COLORS.amberText}">${formatNumber(p.queue_count)}</span><span class="k">queued</span></div>
        <div class="stat"><span class="n" style="color:${COLORS.violet}">${formatNumber(p.claim_count)}</span><span class="k">claims</span></div>
      </div>
      <div class="card-actions">
        <span class="card-cta">${active ? "Filtering — click to clear" : "Filter to this program"}</span>
        <button class="text-link details-btn" type="button" data-details="${escapeHtml(p.id)}">Charter <span class="arr">→</span></button>
      </div>
    </article>`;
  }).join("");
  host.querySelectorAll("[data-program]").forEach((card) => {
    const id = card.getAttribute("data-program");
    onActivate(card, () => { setProgram(id); if (state.program === id) gotoSection("experiments"); });
  });
  host.querySelectorAll("[data-details]").forEach((btn) => {
    btn.addEventListener("click", (e) => { e.stopPropagation(); openProgram(programById.get(btn.getAttribute("data-details"))); });
  });
}

function openProgram(p) {
  if (!p) return;
  openDrawer("Research program", titleForProgram(p.id), p.excerpt || p.focus, [
    ["Program id", `<span class="mono-val">${escapeHtml(p.id)}</span>`],
    ["Focus", escapeHtml(p.focus)],
    ["Experiments", `${formatNumber(p.experiment_count)} total · ${formatNumber(p.anchor_ready_count)} anchor-ready`],
    ["Queued probes", formatNumber(p.queue_count)],
    ["Claims", formatNumber(p.claim_count)],
    ["Seed tags", (p.seed_tags || []).length ? (p.seed_tags || []).map((t) => `<span class="pill">${escapeHtml(t)}</span>`).join(" ") : ""]
  ], `<div class="drawer-actions">
        <button class="btn-primary" type="button" id="drawerFilterBtn">Filter the atlas to this program</button>
        <a class="text-link" href="${repoLink(p.path)}">Open charter on GitHub <span class="arr">→</span></a>
      </div>`);
  const btn = document.getElementById("drawerFilterBtn");
  if (btn) btn.addEventListener("click", () => { closeDrawer(); setProgram(p.id); gotoSection("experiments"); });
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
  const head = TABLE_COLS.map((c) => {
    const isSorted = state.sort.key === c.key;
    const ariaSort = isSorted ? (state.sort.dir > 0 ? "ascending" : "descending") : "none";
    const caret = isSorted ? (state.sort.dir > 0 ? "▲" : "▼") : "↕";
    return `<th data-sort="${c.key}" aria-sort="${ariaSort}"><button type="button" class="th-btn">${c.label}<span class="sort-caret${isSorted ? " on" : ""}">${caret}</span></button></th>`;
  }).join("");
  const body = shown.map((e) => `
    <tr data-experiment="${escapeHtml(e.id)}" tabindex="0" aria-label="${escapeHtml(e.id)} — open details">
      <td data-label="Experiment"><span class="cell-id">${escapeHtml(e.id)}</span><span class="cell-title">${escapeHtml(e.title)}</span></td>
      <td class="col-progs" data-label="Programs" title="${escapeHtml(e.programs.map(titleForProgram).join(", "))}"><span class="prog-dots">${e.programs.map((id) => `<span class="prog-dot" style="background:${programColor(id)}"></span>`).join("")}${e.programs.length > 1 ? `<span class="prog-count">${e.programs.length}</span>` : ""}</span></td>
      <td data-label="Ready"><span class="status-tag ${e.anchor_ready === "yes" ? "ready" : "notready"}">${e.anchor_ready === "yes" ? "Ready" : "Needs curation"}</span></td>
      <td class="col-mono" data-label="Run surface">${escapeHtml(humanizeToken(e.run_surface) || "—")}</td>
      <td class="col-mono" data-label="Needs">${(e.needs || []).length ? escapeHtml((e.needs || []).map(humanizeToken).join(", ")) : '<span class="none">none</span>'}</td>
    </tr>`).join("");
  const note = all.length > TABLE_LIMIT ? `<div class="table-toolbar">Showing first ${TABLE_LIMIT} of ${formatNumber(all.length)} — narrow with search or filters.</div>` : "";
  document.getElementById("experimentTable").innerHTML = all.length
    ? `${note}<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`
    : emptyState("No experiments match the active filters.");
  document.querySelectorAll("#experimentTable [data-sort]").forEach((th) => {
    th.querySelector(".th-btn").addEventListener("click", () => {
      const key = th.getAttribute("data-sort");
      if (state.sort.key === key) state.sort.dir *= -1; else state.sort = { key, dir: 1 };
      renderExperiments();
    });
  });
  document.querySelectorAll("#experimentTable [data-experiment]").forEach((tr) => {
    onActivate(tr, () => openExperiment(experimentById.get(tr.getAttribute("data-experiment"))));
  });
  wireEmptyReset();
}

function emptyState(message) {
  return `<div class="empty-state">${escapeHtml(message)}<button class="text-link empty-reset" type="button">Reset filters <span class="arr">→</span></button></div>`;
}
function wireEmptyReset() {
  document.querySelectorAll(".empty-reset").forEach((b) => b.addEventListener("click", resetFilters));
}

function openExperiment(e) {
  if (!e) return;
  const hasSmoke = e.smoke_command && !["no", "none", ""].includes(String(e.smoke_command).toLowerCase());
  openDrawer("Experiment", e.title, e.summary, [
    ["Id", `<span class="mono-val">${escapeHtml(e.id)}</span>`],
    ["Programs", e.programs.map((id) => programPill(id)).join(" ")],
    ["Status", `<span class="status-tag ${e.anchor_ready === "yes" ? "ready" : "notready"}">${e.anchor_ready === "yes" ? "Anchor-ready" : "Needs curation"}</span>`],
    ["Run surface", e.run_surface ? `<span class="mono-val">${escapeHtml(humanizeToken(e.run_surface))}</span>` : ""],
    ["Smoke test", hasSmoke ? `<code>${escapeHtml(e.smoke_command)}</code>` : '<span class="muted-val">none yet</span>'],
    ["Needs", (e.needs || []).length ? (e.needs || []).map((n) => `<span class="pill">${escapeHtml(humanizeToken(n))}</span>`).join(" ") : '<span class="muted-val">none</span>'],
    ["Tags", (e.tags || []).length ? (e.tags || []).map((t) => `<span class="pill">${escapeHtml(t)}</span>`).join(" ") : ""],
    ["Artifacts", listText(e.recognized_artifacts) ? escapeHtml(listText(e.recognized_artifacts)) : ""],
    ["Files", `${formatNumber(e.total_files)} · ${formatBytes(e.total_size_bytes).n} ${formatBytes(e.total_size_bytes).unit}`]
  ], `<div class="drawer-section-label">Open</div><div class="drawer-links">
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
      <article class="queue-card" data-queue="${escapeHtml(item.id)}" tabindex="0" role="button" aria-label="${escapeHtml(item.title)}" style="border-left:3px solid ${meta.color}">
        <div class="meta-line"><span>${escapeHtml(humanizeStatus(item.status))}</span><span class="effort-chip">${escapeHtml(humanizeStatus(item.effort))} effort</span></div>
        <h4>${escapeHtml(item.title)}</h4>
        <p class="question">${escapeHtml(item.question)}</p>
        <div class="pill-row">${(item.programs || []).slice(0, 3).map((id) => programPill(id)).join("")}</div>
      </article>`).join("") : '<div class="queue-empty">No matching proposals.</div>';
    return `<div class="queue-column" style="--col-color:${meta.color}">
      <div class="queue-col-head"><h3>${pr}</h3><span class="count">${items.length}</span><span class="meaning">${meta.meaning}</span></div>
      ${cards}
    </div>`;
  }).join("");
  document.querySelectorAll("[data-queue]").forEach((card) => {
    onActivate(card, () => openQueue(DATA.queue.find((i) => i.id === card.getAttribute("data-queue"))));
  });
}

function openQueue(item) {
  if (!item) return;
  const meta = QUEUE_META[item.priority] || {};
  openDrawer(`Queue · ${item.priority}${meta.meaning ? " · " + meta.meaning : ""}`, item.title, item.question, [
    ["Queue id", `<span class="mono-val">${escapeHtml(item.id)}</span>`],
    ["Priority", `${escapeHtml(item.priority)} · ${escapeHtml(humanizeStatus(item.status))} · ${escapeHtml(humanizeStatus(item.effort))} effort`],
    ["Programs", (item.programs || []).map((id) => programPill(id)).join(" ")],
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
    return `<article class="claim-card" data-claim="${escapeHtml(c.id)}" tabindex="0" role="button" aria-label="Claim ${escapeHtml(c.id)}: ${escapeHtml(c.title)}" style="--accent:${color}">
      <div class="claim-top">
        <span class="claim-id">${escapeHtml(c.id)}</span>
        <span class="status-chip" style="--status-color:${color}">${escapeHtml(humanizeStatus(c.status))}</span>
      </div>
      <h4>${escapeHtml(c.title)}</h4>
      <p class="claim-summary">${escapeHtml(c.summary)}</p>
      <div class="pill-row">${splitList(c.programs).map((id) => programPill(id)).join("")}</div>
    </article>`;
  }).join("");
  host.querySelectorAll("[data-claim]").forEach((card) => {
    onActivate(card, () => openClaim(DATA.claims.find((c) => c.id === card.getAttribute("data-claim"))));
  });
}

function openClaim(c) {
  if (!c) return;
  const color = STATUS_COLORS[c.status] || COLORS.violet;
  openDrawer(`Claim ${c.id}`, c.title, c.summary, [
    ["Status", `<span class="status-chip" style="--status-color:${color}">${escapeHtml(humanizeStatus(c.status))}</span>`],
    ["Programs", splitList(c.programs).map((id) => programPill(id)).join(" ")],
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
/* Count a field across a set of experiments -> sorted [{id,label,value}] rows. */
function countRows(items, getValues, opts = {}) {
  const counts = new Map();
  items.forEach((item) => {
    let vals = getValues(item);
    if (!Array.isArray(vals)) vals = vals ? [vals] : [];
    vals.forEach((v) => counts.set(v, (counts.get(v) || 0) + 1));
  });
  const rows = Array.from(counts, ([id, value]) => ({ id, value, label: humanizeToken(id) }));
  rows.sort((a, b) => b.value - a.value);
  return opts.limit ? rows.slice(0, opts.limit) : rows;
}

function renderCharts() {
  const fexp = filteredExperiments();
  const fstats = programStats(fexp);
  programNetwork(fstats, fexp.length);
  donutChart(fexp);
  scatterPlot();
  programHeatmap(fstats);
  claimGraph();
  // explorer charts react to the active filters (consistent with the scatter + table)
  const exp = fexp;
  const needRows = countRows(exp, (e) => e.needs || []);
  barChart("needsChart", needRows.length ? needRows : DATA.charts.needs.map((r) => ({ ...r, label: humanizeToken(r.id) })), { label: "curation needs", labelW: 190, color: COLORS.rose, empty: "No curation gaps in this view." });
  const runRows = countRows(exp, (e) => e.run_surface);
  barChart("runSurfaceChart", runRows, { label: "run surfaces", labelW: 170, rowH: 30, color: COLORS.teal, empty: "No experiments in this view." });
  const prioOrder = { P0: 0, P1: 1, P2: 2 };
  const prioRows = DATA.charts.queue_priority.slice().sort((a, b) => (prioOrder[a.id] ?? 9) - (prioOrder[b.id] ?? 9));
  barChart("queuePriorityChart", prioRows, {
    label: "queue priorities", labelW: 50, rowH: 34,
    colors: { P0: COLORS.rose, P1: COLORS.amber, P2: COLORS.blue }
  });
  barChart("artifactKindChart", DATA.charts.artifact_kinds.map((r) => ({ ...r, label: humanizeToken(r.id) })), { label: "artifact manifest kinds", labelW: 130, color: COLORS.violet });
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
  syncControls();
  render();
}

let searchTimer = null;
function onSearchInput(value) {
  state.search = value.trim();
  syncControls();
  clearTimeout(searchTimer);
  searchTimer = setTimeout(render, 110);
}

function bindEvents() {
  [els.search, els.searchTop].forEach((c) => c && c.addEventListener("input", (e) => onSearchInput(e.target.value)));
  [els.program, els.programTop].forEach((c) => c && c.addEventListener("change", (e) => { state.program = e.target.value; syncControls(); render(); }));
  [els.ready, els.readyTop].forEach((c) => c && c.addEventListener("change", (e) => { state.ready = e.target.value; syncControls(); render(); }));
  [els.need, els.needTop].forEach((c) => c && c.addEventListener("change", (e) => { state.need = e.target.value; syncControls(); render(); }));
  [els.reset, els.resetTop].forEach((c) => c && c.addEventListener("click", resetFilters));

  els.drawerClose.addEventListener("click", closeDrawer);
  els.drawerScrim.addEventListener("click", closeDrawer);

  // cross-linking: any program pill anywhere filters the atlas
  document.addEventListener("click", (e) => {
    const pill = e.target.closest("[data-program-pill]");
    if (!pill) return;
    e.stopPropagation();
    const id = pill.getAttribute("data-program-pill");
    if (els.drawer.classList.contains("open")) closeDrawer();
    state.program = id; syncControls(); render();
    gotoSection("experiments");
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDrawer();
    trapDrawerFocus(e);
    const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName);
    if (e.key === "/" && !typing) { e.preventDefault(); (els.search || els.searchTop)?.focus(); }
  });
}

/* Restrained load reveal + one signature moment (corpus map / donut draw-in). */
function setupMotion() {
  if (prefersReducedMotion()) return; // honor the global guard; leave everything visible
  document.documentElement.classList.add("js-motion");
  // Only the signature surfaces get a one-shot reveal. Nothing is ever left hidden:
  // the hide rules are scoped to .play-once, which is removed shortly after firing.
  if (!("IntersectionObserver" in window)) return;
  const targets = [document.querySelector(".panel-signature"), document.getElementById("readinessDonut")].filter(Boolean);
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add("play-once");
      setTimeout(() => e.target.classList.remove("play-once"), 1100);
      obs.unobserve(e.target);
    });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.04 });
  targets.forEach((t) => obs.observe(t));
}

/* Reveal the sticky filter rail once the hero controls scroll out of view. */
function setupFilterRail() {
  if (!els.filterRail) return;
  const hero = document.querySelector(".global-controls");
  if (!hero || !("IntersectionObserver" in window)) { return; }
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      els.filterRail.classList.toggle("visible", !entry.isIntersecting && entry.boundingClientRect.top < 0);
    });
  }, { threshold: 0 });
  obs.observe(hero);
}

renderHero();
renderFilters();
renderMetrics();
renderNarrative();
bindEvents();
setupScrollSpy();
setupFilterRail();
setupMotion();
render();

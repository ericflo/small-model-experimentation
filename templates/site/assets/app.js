"use strict";

const DATA = JSON.parse(document.getElementById("site-data").textContent);

const COLORS = {
  teal: "#0a666e",
  blue: "#2b5fae",
  green: "#15703b",
  amber: "#b9720f",
  queue: "#c2410c", // future-queue brand (also QUEUE_META.P0) — one hue for the queue concept everywhere
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

/* A second, non-color channel for claim status so the evidence matrix reads without color
   vision (Confirmed-green vs Negative-red are near-isoluminant): each status maps to a glyph. */
const STATUS_GLYPH = {
  Confirmed: "✓",
  Promising: "+",
  Negative: "✕",
  Open: "?",
  Retired: "–"
};

/* "Negative" in the source means a recorded negative RESULT (an approach tested and found wanting),
   not that the claim itself is refuted — spell that out so the label can't be misread. */
const STATUS_LABEL = { Negative: "Negative result" };
function statusLabel(status) { return STATUS_LABEL[status] || humanizeStatus(status); }

/* 11 mutually distinct categorical hues for programs, hue-spaced so no two collide and
   none lands on the reserved status green; all dark enough for white-on-fill AA (code badges)
   and text-on-white AA (CTAs). Identity is reinforced by 2-letter initials + the legend. */
const PROGRAM_PALETTE = [
  "#3a6ea5", "#14807f", "#634bbf", "#a83a7e", "#9c6418", "#8f5226",
  "#2c7088", "#a8443a", "#6f7218", "#7a3f9c", "#4a6178"
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

/* Clean a corpus title for display: drop the generator's trailing " Experiment",
   then apply display title-casing. */
function cleanTitle(value) {
  return humanizeTitle(String(value || "").replace(/\s+Experiment$/i, ""));
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
/* candidate programs are proposed lines named by queue probes but not yet in the corpus */
const candidateById = new Map((DATA.candidate_programs || []).map((c) => [c.id, c]));
function isEstablishedProgram(id) { return programById.has(id); }

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
  const p = programById.get(id) || candidateById.get(id);
  return p ? humanizeTitle(p.title) : humanizeTitle(humanizeToken(id));
}
function programPill(id) {
  const name = titleForProgram(id);
  if (!isEstablishedProgram(id)) {
    // proposed program — not yet in the corpus, so it's informational, not a filter
    return `<span class="pill proposed" title="Proposed program — not yet in the corpus">${escapeHtml(name)}<span class="proposed-tag">proposed</span></span>`;
  }
  return `<button class="pill dot pill-link" type="button" data-program-pill="${escapeHtml(id)}" style="--pill-color:${programColor(id)}" title="Filter to ${escapeHtml(name)}">${escapeHtml(name)}</button>`;
}
function repoLink(path) { return path ? `${DATA.repo.github}/blob/main/${path}` : ""; }
function splitList(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  return String(value).split(";").map((s) => s.trim()).filter(Boolean);
}
function listText(items) { return (items || []).join(", "); }
function plural(n, word) { return `${formatNumber(n)} ${word}${Number(n) === 1 ? "" : "s"}`; }
function textBlob(item) {
  const base = Object.values(item).flatMap((v) => (Array.isArray(v) ? v : [v])).join(" ");
  // index the humanized program names + readiness status so people can search by what they see
  const progNames = item.programs ? splitList(item.programs).map(titleForProgram).join(" ") : "";
  const status = item.anchor_ready ? (item.anchor_ready === "yes" ? "anchor-ready" : "not anchor-ready needs curation") : "";
  return `${base} ${progNames} ${status}`.toLowerCase();
}
const normSep = (s) => s.replace(/[^a-z0-9]+/gi, " "); // commas, hyphens, slashes, dots -> spaces
function matchesSearch(item) {
  const q = normSep(state.search.trim().toLowerCase());
  if (!q) return true;
  const blob = normSep(textBlob(item));
  return q.split(/\s+/).filter(Boolean).every((t) => blob.includes(t)); // token-AND
}
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
  const tip = els.tooltip;
  tip.innerHTML = html;
  tip.classList.add("visible");
  const pad = 12;
  const rect = tip.getBoundingClientRect();
  const halfW = rect.width / 2;
  const x = Math.min(window.innerWidth - halfW - pad, Math.max(halfW + pad, event.clientX));
  const below = event.clientY - rect.height - 16 < pad; // flip below when no room above
  tip.classList.toggle("below", below);
  tip.style.left = `${x}px`;
  tip.style.top = `${event.clientY + (below ? 16 : 0)}px`;
}
function hideTooltip() { els.tooltip.classList.remove("visible"); }
let lastTooltipFocusAt = -1e9; // guards the scroll-hide against the scroll-into-view a focus triggers
function attachTooltip(el, html) {
  el.addEventListener("mousemove", (e) => showTooltip(e, html));
  el.addEventListener("mouseleave", hideTooltip);
  // keyboard/touch parity: show on focus relative to the element's box. Read the rect on the next
  // frame so it reflects the post-scroll-into-view position when focus moves an off-screen node.
  el.addEventListener("focus", () => {
    lastTooltipFocusAt = performance.now();
    requestAnimationFrame(() => {
      const r = el.getBoundingClientRect();
      showTooltip({ clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 }, html);
    });
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
  return DATA.programs.filter((p) => {
    if (state.program !== "all" && p.id !== state.program) return false;
    return matchesSearch(p); // same matcher as every other panel (normalize + token-AND)
  });
}
function filtersActive() {
  return !!state.search.trim() || state.program !== "all" || state.ready !== "all" || state.need !== "all";
}

function prefersReducedMotion() {
  return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function syncControls() {
  [els.program, els.programTop].forEach((c) => c && (c.value = state.program));
  [els.ready, els.readyTop].forEach((c) => c && (c.value = state.ready));
  [els.need, els.needTop].forEach((c) => c && (c.value = state.need));
  // never overwrite the field the user is typing in (it strips spaces / moves the caret)
  [els.search, els.searchTop].forEach((c) => { if (c && c !== document.activeElement) c.value = state.search; });
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
let openingFromHistory = false; // true while opening from popstate/deep-link (don't push)
let drawerPushed = false; // did opening this drawer add a history entry we should pop on close?
function openDrawer(eyebrow, title, lead, rows, extra = "", hash = "") {
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
  // make the open entity a shareable URL + let Back/Forward close/reopen it.
  // switching entities replaces (one drawer entry total) so a single Back always closes.
  if (hash && !openingFromHistory) {
    try {
      const url = `${location.pathname}${location.search}#${hash}`;
      if (drawerPushed) history.replaceState({ drawer: 1 }, "", url);
      else history.pushState({ drawer: 1 }, "", url);
      drawerPushed = true;
    } catch (_) {}
  }
}
/* user-initiated close (X / Esc / scrim): pop the history entry so no dead trail accrues.
   Clear the flag BEFORE history.back() so a repeat key/double-click can't double-pop off-page. */
function dismissDrawer() {
  if (drawerPushed) { drawerPushed = false; try { history.back(); return; } catch (_) {} }
  closeDrawer();
}
function closeDrawer() {
  if (!els.drawer.classList.contains("open")) return;
  els.drawer.classList.remove("open");
  els.drawer.setAttribute("inert", "");
  els.drawerScrim.classList.remove("open");
  els.drawer.setAttribute("aria-hidden", "true");
  setBackgroundInert(false);
  if (lastFocused && lastFocused.focus) lastFocused.focus();
  // never leave a stale entity hash (covers deep-link opens that didn't push)
  if (location.hash && /^#(exp|claim|queue|program)\//.test(location.hash)) {
    try { history.replaceState(null, "", `${location.pathname}${location.search}`); } catch (_) {}
  }
  drawerPushed = false;
}
function openEntityFromHash(h) {
  const slash = h.indexOf("/");
  if (slash < 0) return;
  const kind = h.slice(0, slash), id = h.slice(slash + 1);
  const section = { exp: "experiments", claim: "claims", queue: "queue", program: "programs" }[kind];
  let opened = false;
  if (kind === "exp" && experimentById.has(id)) { openExperiment(experimentById.get(id)); opened = true; }
  else if (kind === "claim") { const c = DATA.claims.find((x) => x.id === id); if (c) { openClaim(c); opened = true; } }
  else if (kind === "queue") { const i = DATA.queue.find((x) => x.id === id); if (i) { openQueue(i); opened = true; } }
  else if (kind === "program" && programById.has(id)) { openProgram(programById.get(id)); opened = true; }
  // orient the reader: leave the owning section in view behind the drawer
  if (opened && section) gotoSection(section);
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
  // Lede stays evergreen (the live counts live in the metric ledger, which reacts to filters).
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
    '<option value="all">All tasks</option>',
    ...needs.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(humanizeToken(n))}</option>`)
  ].join("");
  [els.program, els.programTop].forEach((c) => c && (c.innerHTML = programOpts));
  [els.need, els.needTop].forEach((c) => c && (c.innerHTML = needOpts));
}

function activeChips() {
  const chips = [];
  if (state.program !== "all") chips.push({ label: titleForProgram(state.program), clear: () => setProgram(state.program) });
  if (state.ready === "yes") chips.push({ label: "Anchor-ready", clear: () => setReady("all") });
  if (state.ready === "no") chips.push({ label: "Not anchor-ready", clear: () => setReady("all") });
  if (state.need !== "all") chips.push({ label: humanizeToken(state.need), clear: () => setNeed("all") });
  if (state.search.trim()) chips.push({ label: `“${state.search.trim()}”`, clear: () => setSearch("") });
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
    b.addEventListener("click", () => {
      if (key === "reset") { resetFilters(); return; }
      chips[Number(key)].clear(); // chip element is destroyed on re-render — park focus somewhere stable
      restoreFilterFocus();
    });
  });
}
/* park keyboard focus on a stable control after a filter chip / reset destroys the active element */
function restoreFilterFocus() {
  const rail = els.filterRail && els.filterRail.classList.contains("visible");
  const target = rail ? (els.resetTop || els.searchTop) : (els.search || els.searchTop);
  if (target && target.focus) target.focus();
}

function updateFilterStatus() {
  const chips = activeChips();
  const active = chips.length > 0;
  els.filterStatus.textContent = active
    ? "Scoping the atlas — remove a chip to widen."
    : "Filters scope every panel below (Artifacts stays corpus-wide).";
  [els.reset, els.resetTop].forEach((b) => b && (b.hidden = !active));
  if (els.filterRail) els.filterRail.classList.toggle("has-active", active); // mobile: show only when filtering
  // rail chips (with clear-all) and hero chips (compact)
  if (els.filterChips) {
    // no "Clear all" here: it would be appended inside the clip-overflow rail (so it's the control
    // that vanishes once chips wrap) and it duplicates the always-visible "Reset" button beside it
    els.filterChips.innerHTML = active ? chipMarkup(chips, false) : '<span class="chips-empty">No filters active</span>';
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
  { label: "Experiments", key: "experiments", help: "Self-contained experiments", color: COLORS.teal, action: () => gotoSection("experiments") },
  // when filtered, "N spanned in view" is a multi-program reach — route to the map, whose legend
  // visualizes that spanned set; #programs hard-filters to the single selected program (would show 1)
  { label: "Programs", key: "programs", help: "Durable lines of inquiry", color: COLORS.ink, action: () => gotoSection(state.program !== "all" ? "map" : "programs") },
  { label: "Anchor-ready", key: "anchor_ready", help: "Citable and buildable today", color: COLORS.green, filters: true, readyVal: "yes", action: () => { setReady(state.ready === "yes" ? "all" : "yes"); gotoSection("readiness"); } },
  { label: "Not anchor-ready", key: "needs_curation", help: "Has pending curation tasks", color: COLORS.amber, filters: true, readyVal: "no", action: () => { setReady(state.ready === "no" ? "all" : "no"); gotoSection("readiness"); } },
  { label: "Future queue", key: "future_proposals", help: "Structured next probes", color: COLORS.queue, action: () => gotoSection("queue") },
  { label: "Claims", key: "claims", help: "Shared evidence statements", color: COLORS.violet, action: () => gotoSection("claims") },
  // corpus-wide: these describe the whole tree (the Artifacts charts are non-reactive), so the
  // headline stays the whole-tree total even under a filter — no "of N", matching the destination
  { label: "Files indexed", key: "total_files", help: "Tracked corpus files", color: COLORS.ink, corpusWide: true, action: () => gotoSection("artifacts") },
  { label: "Tracked size", key: "total_size_bytes", help: "Repository-local footprint", color: COLORS.ink, isBytes: true, corpusWide: true, action: () => gotoSection("artifacts") }
];

/* live value for a metric over the current filter scope */
function metricValue(key) {
  const fexp = filteredExperiments();
  switch (key) {
    case "experiments": return fexp.length;
    case "programs": return new Set(fexp.flatMap((e) => e.programs || [])).size;
    case "anchor_ready": return fexp.filter((e) => e.anchor_ready === "yes").length;
    case "needs_curation": return fexp.filter((e) => e.anchor_ready !== "yes").length;
    case "future_proposals": return filteredQueue().length;
    case "claims": return filteredClaims().length;
    case "total_files": return fexp.reduce((a, e) => a + Number(e.total_files || 0), 0);
    case "total_size_bytes": return fexp.reduce((a, e) => a + Number(e.total_size_bytes || 0), 0);
    default: return 0;
  }
}

function renderMetrics() {
  document.getElementById("metricGrid").innerHTML = METRICS
    .map((m, i) => `
      <button class="metric-card${m.filters ? " is-filter" : ""}" type="button" data-metric="${i}"${m.readyVal ? ' aria-pressed="false"' : ""} style="--accent:${m.color};--i:${i}">
        <span class="eyebrow">${escapeHtml(m.label)}</span>
        <span class="metric-value" data-mv="${i}"></span>
        <span class="metric-help">${escapeHtml(m.help)}<span class="metric-go" aria-hidden="true">${m.filters ? "filter →" : "view →"}</span></span>
      </button>`)
    .join("");
  document.querySelectorAll("[data-metric]").forEach((card) => {
    card.addEventListener("click", () => METRICS[Number(card.getAttribute("data-metric"))].action());
  });
  updateMetrics();
}

/* Update the metric values to the filtered scope; show "of total" when filtered. */
function updateMetrics() {
  const s = DATA.summary;
  const filtering = filtersActive();
  METRICS.forEach((m, i) => {
    const node = document.querySelector(`[data-mv="${i}"]`);
    if (!node) return;
    const total = m.key === "programs" ? DATA.programs.length : s[m.key];
    const v = (filtering && !m.corpusWide) ? metricValue(m.key) : total;
    const value = m.isBytes ? formatBytes(v) : { n: formatNumber(v), unit: "" };
    const showOf = filtering && !m.corpusWide && v !== total;
    // "Programs" under a program filter is a multi-tag reach, not "N of 11" — reframe it
    const ofText = m.key === "programs" && state.program !== "all"
      ? "spanned in view"
      : `of ${escapeHtml(m.isBytes ? formatBytes(total).n + " " + formatBytes(total).unit : formatNumber(total))}`;
    node.innerHTML = `${escapeHtml(value.n)}${value.unit ? `<span class="unit">${escapeHtml(value.unit)}</span>` : ""}` +
      (showOf ? `<span class="metric-of">${ofText}</span>` : "");
    if (m.readyVal) {
      const card = document.querySelector(`[data-metric="${i}"]`);
      if (card) card.setAttribute("aria-pressed", String(state.ready === m.readyVal));
    }
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
  // fold the data into the label so screen readers get the distribution, not just "bar chart"
  const summary = rows.slice(0, 8).map((r) => `${r.label || r.id} ${formatNumber(r.value)}`).join(", ");
  const ariaLabel = `${options.label || "bar chart"}: ${summary}`;
  const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: typeof options.onBar === "function" ? "group" : "img", "aria-label": ariaLabel, preserveAspectRatio: "xMinYMin meet" });
  rows.forEach((row, i) => {
    const y = top + i * rowH;
    const value = Number(row.value || 0);
    const bw = Math.max(2, (plotW * value) / maxValue);
    const color = options.colors?.[row.id] || options.color || PROGRAM_PALETTE[i % PROGRAM_PALETTE.length];
    const label = row.label || row.id;
    chart.appendChild(text(label, { x: 0, y: y + rowH / 2 + 4, class: "chart-label" }));
    chart.appendChild(svg("rect", { x: labelW, y: y + rowH / 2 - 9, width: plotW, height: 18, rx: 5, fill: COLORS.tint }));
    const interactive = typeof options.onBar === "function";
    const barAttrs = { x: labelW, y: y + rowH / 2 - 9, width: bw, height: 18, rx: 5, fill: color };
    if (interactive) {
      Object.assign(barAttrs, { tabindex: 0, role: "button", "data-bar": row.id, "aria-label": `${label}: ${value}. Activate to filter.`, style: "cursor:pointer" });
    }
    const bar = svg("rect", barAttrs);
    attachTooltip(bar, `<strong>${escapeHtml(label)}</strong><br><span class="tip-meta">${formatNumber(value)}</span>`);
    if (interactive) {
      // widen the hit target across the whole row for easy clicking/focus
      const hit = svg("rect", { x: labelW, y, width: plotW, height: rowH, fill: "transparent", style: "cursor:pointer" });
      hit.addEventListener("click", () => options.onBar(row.id));
      chart.appendChild(hit);
      onActivate(bar, () => options.onBar(row.id));
    }
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
  const fqByProg = {}; filteredQueue().forEach((i) => (i.programs || []).forEach((id) => { fqByProg[id] = (fqByProg[id] || 0) + 1; }));
  const fcByProg = {}; filteredClaims().forEach((c) => splitList(c.programs).forEach((id) => { fcByProg[id] = (fcByProg[id] || 0) + 1; }));
  const W = 760, H = 446, cx = W / 2, cy = H / 2 - 6, ringR = 156;
  const progs = PROGRAMS_BY_SIZE;
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Corpus map: research programs sized by experiments, ring shows percent anchor-ready" });

  [ringR + 46, ringR, ringR - 56].forEach((r) => {
    chart.appendChild(svg("circle", { cx, cy, r, fill: "none", stroke: COLORS.lineInv, "stroke-width": 1, opacity: 0.5 }));
  });

  const filtering = state.program !== "all" || state.ready !== "all" || state.need !== "all" || !!state.search;
  // disk size encodes experiments IN THE CURRENT VIEW so it never contradicts the legend/center
  const sizeOf = (p) => filtering ? (stats.get(p.id)?.count || 0) : p.experiment_count;
  const maxCount = Math.max(1, ...progs.map(sizeOf));
  // true sqrt-area scale (area ∝ count) with a small legibility clamp
  const nodeR = (c) => (c <= 0 ? 9 : Math.max(11, Math.sqrt(c / maxCount) * 41));

  progs.forEach((p, i) => {
    const angle = -Math.PI / 2 + (i / progs.length) * Math.PI * 2;
    const x = cx + Math.cos(angle) * ringR, y = cy + Math.sin(angle) * ringR;
    const inView = (stats.get(p.id)?.count || 0) > 0;
    const active = state.program === p.id;
    chart.appendChild(svg("line", {
      x1: cx, y1: cy, x2: x, y2: y,
      stroke: active ? programColor(p.id) : COLORS.lineInv,
      "stroke-width": active ? 2.4 : 1 + (sizeOf(p) / maxCount) * 1.4,
      opacity: filtering && !inView ? 0.12 : active ? 0.9 : 0.5
    }));
  });

  chart.appendChild(svg("circle", { cx, cy, r: 46, fill: "rgba(255,255,255,0.04)", stroke: COLORS.lineInv, "stroke-width": 1.5 }));
  chart.appendChild(text(formatNumber(fcount == null ? DATA.summary.experiments : fcount), { x: cx, y: cy - 1, "text-anchor": "middle", class: "chart-value-inv", style: "font-size:23px;font-family:var(--display)" }));
  chart.appendChild(text(filtering ? "IN VIEW" : "EXPERIMENTS", { x: cx, y: cy + 16, "text-anchor": "middle", class: "axis-label-inv" }));

  progs.forEach((p, i) => {
    const angle = -Math.PI / 2 + (i / progs.length) * Math.PI * 2;
    const x = cx + Math.cos(angle) * ringR, y = cy + Math.sin(angle) * ringR;
    const r = nodeR(sizeOf(p));
    const color = programColor(p.id);
    const active = state.program === p.id;
    const s = stats.get(p.id) || { count: 0, ready: 0 };
    const inView = s.count > 0;
    const ratio = s.count ? s.ready / s.count : 0;
    const opacity = filtering && !inView ? 0.22 : 1;
    const g = svg("g", { tabindex: 0, role: "button", class: "map-node", "data-mapnode": p.id,
      "aria-label": `${titleForProgram(p.id)}: ${inView ? `${s.count} experiments, ${Math.round(ratio * 100)} percent anchor-ready` : "no experiments in this view"}. Activate to filter.`,
      style: `cursor:pointer; opacity:${opacity}; --i:${i}` });
    g.appendChild(svg("circle", { cx: x, cy: y, r, fill: color, stroke: active ? "#fff" : "rgba(255,255,255,0.25)", "stroke-width": active ? 3 : 1.5 }));
    // readiness ring: a faint track with a GREEN ARC whose LENGTH = % anchor-ready.
    // Encoding readiness as arc length over a luminance-contrasting track (green on faint
    // grey) makes it readable without color vision and positionally, and the unfilled track
    // reads as the still-to-curate remainder. Programs with zero experiments in view get a
    // neutral dashed ring — no false "0% ready" amber alarm contradicting the scorecard "—".
    const rr = r + 7, circ = 2 * Math.PI * rr;
    if (inView) {
      g.appendChild(svg("circle", { cx: x, cy: y, r: rr, fill: "none", stroke: "rgba(255,255,255,0.16)", "stroke-width": 5 }));
      if (ratio > 0.001) g.appendChild(svg("circle", {
        cx: x, cy: y, r: rr, fill: "none", stroke: "#3fbf86", "stroke-width": 5,
        "stroke-dasharray": `${ratio * circ} ${circ}`, "stroke-linecap": ratio > 0.985 ? "butt" : "round",
        transform: `rotate(-90 ${x} ${y})`
      }));
    } else {
      g.appendChild(svg("circle", { cx: x, cy: y, r: rr, fill: "none", stroke: COLORS.lineInv, "stroke-width": 2, "stroke-dasharray": "2 4" }));
    }
    g.appendChild(text(programCode(p), { x, y: y + 4, "text-anchor": "middle", class: "node-code", style: `font-size:${r > 26 ? 13 : 11}px` }));
    const qN = fqByProg[p.id] || 0, cN = fcByProg[p.id] || 0;
    attachTooltip(g, `<strong>${escapeHtml(titleForProgram(p.id))}</strong><br><span class="tip-meta">${inView ? `${plural(s.count, "experiment")} · ${Math.round(ratio * 100)}% ready · ${plural(qN, "probe")} · ${plural(cN, "claim")}` : "no experiments in this view"}</span>`);
    onActivate(g, () => setProgram(p.id));
    chart.appendChild(g);
  });

  chart.appendChild(text("disk = experiments (may span programs)  ·  green ring arc = % anchor-ready", { x: cx, y: H - 6, "text-anchor": "middle", class: "axis-label-inv" }));

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
  if (!exp.length) { target.innerHTML = '<div class="empty-state" style="padding:40px 16px">No experiments in this view.</div>'; return; }
  const total = exp.length;
  const ready = exp.filter((e) => e.anchor_ready === "yes").length;
  const needs = Math.max(0, total - ready);
  const pct = Math.round((ready / total) * 100);
  // horizontal fuel gauge (reuses the bar vocabulary; no stock pie beside the bespoke map)
  const amber = "#b9720f";
  const W = 320, H = 168, padX = 8, barY = 78, barH = 30, barW = W - padX * 2;
  const readyW = (ready / total) * barW;
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${pct} percent anchor-ready in this view: ${ready} ready, ${needs} not ready`, preserveAspectRatio: "xMinYMin meet" });
  chart.appendChild(text(`${pct}%`, { x: padX, y: 50, class: "chart-value", style: "font-size:46px;font-family:var(--display);letter-spacing:-0.03em" }));
  chart.appendChild(text("ANCHOR-READY", { x: padX + 2, y: 70, class: "axis-label" }));
  // gauge: amber base (not-ready) + green fill (ready) + ticks, wrapped so the reveal wipes the
  // whole gauge as one unit (the ready/not-ready ratio stays truthful at every frame, unlike
  // animating the green fill width alone)
  const gaugeG = svg("g", { class: "gauge-wipe" });
  gaugeG.appendChild(svg("rect", { x: padX, y: barY, width: barW, height: barH, rx: 7, fill: amber }));
  gaugeG.appendChild(svg("rect", { x: padX, y: barY, width: readyW, height: barH, rx: 7, fill: COLORS.green }));
  [0.25, 0.5, 0.75].forEach((t) => gaugeG.appendChild(svg("line", { x1: padX + barW * t, y1: barY, x2: padX + barW * t, y2: barY + barH, stroke: "rgba(255,255,255,0.5)", "stroke-width": 1 })));
  chart.appendChild(gaugeG);
  // legend
  chart.appendChild(svg("rect", { x: padX, y: H - 18, width: 11, height: 11, rx: 2, fill: COLORS.green }));
  chart.appendChild(text(`${formatNumber(ready)} ready`, { x: padX + 17, y: H - 9, class: "chart-label" }));
  chart.appendChild(svg("rect", { x: padX + 116, y: H - 18, width: 11, height: 11, rx: 2, fill: amber }));
  chart.appendChild(text(`${formatNumber(needs)} not ready`, { x: padX + 133, y: H - 9, class: "chart-label" }));
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
  const readyN = rows.filter((e) => e.anchor_ready === "yes").length;
  // single image with a summary label; the table is the keyboard path (no 155 tab stops)
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Files per experiment for ${n} experiments, sorted high to low; ${readyN} ready, ${n - readyN} not ready. Use the table to open an experiment.`, preserveAspectRatio: "xMinYMin meet" });
  chart.appendChild(svg("line", { x1: left, y1: H - bottom, x2: W - right, y2: H - bottom, stroke: COLORS.line }));
  rows.forEach((e, i) => {
    const x = left + (i / n) * plotW;
    const h = Math.max(1.5, (e.total_files / maxFiles) * plotH);
    const color = e.anchor_ready === "yes" ? COLORS.green : COLORS.amber;
    const bar = svg("rect", { x, y: H - bottom - h, width: bw, height: h, rx: bw > 3 ? 1.5 : 0, fill: color, opacity: 0.85, style: "cursor:pointer" });
    const hit = svg("rect", { x: x - 1, y: top, width: Math.max(bw + 2, 4), height: plotH, fill: "transparent", style: "cursor:pointer" });
    attachTooltip(bar, `<strong>${escapeHtml(e.id)}</strong><br><span class="tip-meta">${formatNumber(e.total_files)} files · ${e.anchor_ready === "yes" ? "ready" : "not ready"}</span>`);
    hit.addEventListener("click", () => openExperiment(e));
    chart.appendChild(hit);
    chart.appendChild(bar);
  });
  chart.appendChild(text(`${n} experiments, sorted by file count (max ${formatNumber(maxFiles)})`, { x: left, y: H - 8, class: "axis-label" }));
  // legend
  chart.appendChild(svg("rect", { x: W - right - 150, y: H - 16, width: 9, height: 9, rx: 2, fill: COLORS.green }));
  chart.appendChild(text("ready", { x: W - right - 137, y: H - 8, class: "chart-tick" }));
  chart.appendChild(svg("rect", { x: W - right - 92, y: H - 16, width: 9, height: 9, rx: 2, fill: COLORS.amber }));
  chart.appendChild(text("not ready", { x: W - right - 79, y: H - 8, class: "chart-tick" }));
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

function programHeatmap(fstats) {
  const target = document.getElementById("programHeatmap");
  clear(target);
  const stats = fstats || programStats(DATA.experiments);
  // every column reflects the filtered view
  const metrics = [
    { key: "exp", label: "Exp", color: "#455468", mode: "count" },
    { key: "ready", label: "% Ready", color: COLORS.green, mode: "ratio" },
    { key: "queue", label: "Queued", color: COLORS.queue, mode: "count" },
    { key: "claims", label: "Claims", color: COLORS.violet, mode: "count" }
  ];
  const W = 620, labelW = 270, gutter = 12, fontPx = 9.5;
  const cellW = (W - labelW) / metrics.length, rowH = 30, top = 32;
  const progs = PROGRAMS_BY_SIZE;
  const fqByProg = {}; filteredQueue().forEach((i) => (i.programs || []).forEach((id) => { fqByProg[id] = (fqByProg[id] || 0) + 1; }));
  const fcByProg = {}; filteredClaims().forEach((c) => splitList(c.programs).forEach((id) => { fcByProg[id] = (fcByProg[id] || 0) + 1; }));
  const rowOf = (p) => { const s = stats.get(p.id) || { count: 0, ready: 0 }; return { exp: s.count, ready: s.ready, queue: fqByProg[p.id] || 0, claims: fcByProg[p.id] || 0, expTotal: s.count }; };
  const data = progs.map((p) => ({ p, v: rowOf(p) }));
  const H = top + progs.length * rowH + 6;
  const maxes = {};
  metrics.forEach((m) => { maxes[m.key] = Math.max(1, ...data.map((d) => d.v[m.key])); });
  // Queued/Claims can't honor a readiness/need filter; mark rows with 0 in-view experiments
  const scopeNote = state.ready !== "all" || state.need !== "all";
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H + (scopeNote ? 16 : 0)}`, role: "group", "aria-label": "Program scorecard: experiments, percent ready, queued, claims", preserveAspectRatio: "xMinYMin meet" });
  metrics.forEach((m, ci) => {
    chart.appendChild(text(m.label, { x: labelW + ci * cellW + cellW / 2, y: 18, "text-anchor": "middle", class: "axis-label" }));
  });
  const READY_GREEN = "#15703b", TRACK = "#e7ecf1", MUTED_TRACK = "#eef1f4";
  data.forEach(({ p, v }, ri) => {
    const y = top + ri * rowH;
    const active = state.program === p.id;
    const outOfView = v.expTotal === 0;
    const rowG = svg("g", {});
    if (active) chart.appendChild(svg("rect", { x: 0, y: y - 1, width: W, height: rowH, fill: COLORS.tint, rx: 5 }));
    rowG.appendChild(svg("rect", { x: 0, y: y + rowH / 2 - 7, width: 5, height: 14, rx: 2, fill: programColor(p.id) }));
    const full = titleForProgram(p.id);
    const label = text(ellipsize(full, labelW - gutter - 14, fontPx), { x: 12, y: y + rowH / 2 + 3.5, class: "chart-label", style: `font-size:${fontPx}px` });
    label.appendChild(svg("title", {}, [document.createTextNode(full)]));
    rowG.appendChild(label);
    metrics.forEach((m, ci) => {
      const value = v[m.key];
      const cellX = labelW + ci * cellW, trackX = cellX + 3, trackW = cellW - 6;
      const isRatio = m.mode === "ratio";
      // Queued/Claims for an out-of-view program under a readiness/need filter -> "—"
      const suppressed = scopeNote && outOfView && (m.key === "queue" || m.key === "claims");
      const muted = (isRatio && outOfView) || suppressed;
      const ratio = isRatio ? value / Math.max(1, v.expTotal) : 0;
      // EVERY column is an inline LENGTH gauge now: %Ready fills by ratio, counts by value/colMax.
      // Length (not fill darkness) carries magnitude, so 43 vs 37 separate at a glance and all
      // four columns share one chart grammar.
      const frac = muted ? 0 : (isRatio ? ratio : value / maxes[m.key]);
      const shown = muted ? "—" : (isRatio ? `${Math.round(ratio * 100)}%` : formatNumber(value));
      const cell = svg("rect", {
        x: trackX, y: y + 3, width: trackW, height: rowH - 6, rx: 5,
        fill: muted ? MUTED_TRACK : TRACK, tabindex: 0, role: "button", "data-cell": p.id, "data-col": m.key,
        "aria-label": `${full}, ${m.label}: ${muted ? "not in view" : shown}. Activate to filter.`,
        style: "cursor:pointer"
      });
      attachTooltip(cell, `<strong>${escapeHtml(full)}</strong><br><span class="tip-meta">${m.label}: ${muted ? "not in current view" : (isRatio ? `${Math.round(ratio * 100)}% (${formatNumber(value)}/${formatNumber(v.expTotal)})` : formatNumber(value))}</span>`);
      onActivate(cell, () => setProgram(p.id));
      rowG.appendChild(cell);
      if (frac > 0.001) {
        rowG.appendChild(svg("rect", { x: trackX, y: y + 3, width: Math.max(3, trackW * Math.min(1, frac)), height: rowH - 6, rx: 5,
          fill: isRatio ? READY_GREEN : m.color, style: "pointer-events:none" }));
      }
      // ink digits with a real white halo (paint-order:stroke) -> AA on the light track AND on the
      // saturated fill, so legibility never depends on where the gauge edge falls under the text
      rowG.appendChild(text(shown, { x: cellX + cellW / 2, y: y + rowH / 2 + 4, "text-anchor": "middle", class: "chart-value",
        style: `fill:${muted ? COLORS.faint : COLORS.ink}; paint-order:stroke; stroke:#ffffff; stroke-width:${muted ? 0 : 3}px; stroke-linejoin:round` }));
    });
    chart.appendChild(rowG);
  });
  if (scopeNote) chart.appendChild(text("Queued & Claims reflect program + search scope only", { x: W, y: H + 10, "text-anchor": "end", class: "axis-label" }));
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
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Claims linked to programs matrix" });
  const visibleClaims = new Set(filteredClaims().map((c) => c.id));
  // when any filter is active, dim claims outside the view (incl. the zero-match case)
  const dimClaim = (c) => filtersActive() && !visibleClaims.has(c.id);
  // column headers (claim ids, status dot above)
  claims.forEach((c, ci) => {
    const x = labelW + ci * colW + colW / 2;
    const op = dimClaim(c) ? 0.25 : 1;
    chart.appendChild(svg("circle", { cx: x, cy: 18, r: 7, fill: STATUS_COLORS[c.status] || COLORS.violet, opacity: op }));
    chart.appendChild(text(STATUS_GLYPH[c.status] || "·", { x, y: 21.5, "text-anchor": "middle", opacity: op, style: "fill:#fff;font-size:9px;font-weight:700;font-family:var(--sans);pointer-events:none" }));
    const t = text(c.id, { x, y: 46, "text-anchor": "middle", class: "chart-value", opacity: op,
      tabindex: 0, role: "button", "aria-label": `Claim ${c.id}: ${c.title}, ${statusLabel(c.status)}`, style: "cursor:pointer" });
    onActivate(t, () => openClaim(c));
    attachTooltip(t, `<strong>${escapeHtml(c.id)}: ${escapeHtml(c.title)}</strong><br><span class="tip-meta">${escapeHtml(statusLabel(c.status))}</span>`);
    chart.appendChild(t);
  });
  progs.forEach((p, ri) => {
    const y = top + ri * rowH;
    const activeRow = state.program === p.id;
    if (activeRow) chart.appendChild(svg("rect", { x: 0, y: y - 1, width: W, height: rowH, fill: COLORS.tint, rx: 5 }));
    const rowDim = state.program !== "all" && !activeRow ? 0.4 : 1;
    chart.appendChild(svg("rect", { x: 0, y: y + rowH / 2 - 7, width: 5, height: 14, rx: 2, fill: programColor(p.id), opacity: rowDim }));
    const full = titleForProgram(p.id);
    const lbl = text(ellipsize(full, labelW - gutter - 14, fontPx), { x: 12, y: y + rowH / 2 + 3.5, class: "chart-label", opacity: rowDim, style: `font-size:${fontPx}px` });
    lbl.appendChild(svg("title", {}, [document.createTextNode(full)]));
    chart.appendChild(lbl);
    claims.forEach((c, ci) => {
      const linked = splitList(c.programs).includes(p.id);
      const x = labelW + ci * colW + colW / 2;
      const op = Math.min(rowDim, dimClaim(c) ? 0.25 : 1);
      if (linked) {
        const dot = svg("circle", { cx: x, cy: y + rowH / 2, r: 7, fill: STATUS_COLORS[c.status] || COLORS.violet, opacity: op,
          tabindex: 0, role: "button", "aria-label": `${c.id} (${statusLabel(c.status)}) relates to ${titleForProgram(p.id)}: ${c.title}`, style: "cursor:pointer" });
        attachTooltip(dot, `<strong>${escapeHtml(c.id)}</strong> ↔ <strong>${escapeHtml(titleForProgram(p.id))}</strong><br><span class="tip-meta">${escapeHtml(c.title)}</span>`);
        onActivate(dot, () => openClaim(c));
        chart.appendChild(dot);
        chart.appendChild(text(STATUS_GLYPH[c.status] || "·", { x, y: y + rowH / 2 + 3.5, "text-anchor": "middle", opacity: op, style: "fill:#fff;font-size:9px;font-weight:700;font-family:var(--sans);pointer-events:none" }));
      } else {
        chart.appendChild(svg("circle", { cx: x, cy: y + rowH / 2, r: 2, fill: COLORS.line, opacity: op }));
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
    return `<span class="legend-item" style="cursor:default"><span class="legend-swatch legend-glyph" style="background:${STATUS_COLORS[s] || COLORS.violet}">${STATUS_GLYPH[s] || ""}</span>${escapeHtml(statusLabel(s))} <b>${n}</b></span>`;
  }).join("");
}

/* ---------------------------------------------------------------- programs */
function renderPrograms() {
  const rows = filteredPrograms();
  const host = document.getElementById("programCards");
  if (!rows.length) { host.innerHTML = emptyState("No programs match the active filters."); wireEmptyReset(); return; }
  host.innerHTML = rows.map((p) => {
    const active = state.program === p.id;
    return `
    <article class="program-card${active ? " active" : ""}" data-program="${escapeHtml(p.id)}" style="--accent:${programColor(p.id)}">
      <div class="card-head">
        <span class="card-code" style="background:${programColor(p.id)}">${escapeHtml(programCode(p))}</span>
        <h3>${escapeHtml(titleForProgram(p.id))}</h3>
      </div>
      <p class="focus">${escapeHtml(p.focus)}</p>
      <div class="stat-row">
        <div class="stat"><span class="n">${formatNumber(p.experiment_count)}</span><span class="k">exp</span></div>
        <div class="stat"><span class="n" style="color:${COLORS.green}">${formatNumber(p.anchor_ready_count)}</span><span class="k">ready</span></div>
        <div class="stat"><span class="n" style="color:${COLORS.queue}">${formatNumber(p.queue_count)}</span><span class="k">queued</span></div>
        <div class="stat"><span class="n" style="color:${COLORS.violet}">${formatNumber(p.claim_count)}</span><span class="k">claims</span></div>
      </div>
      <div class="card-actions">
        <button class="card-cta-btn" type="button" data-filter="${escapeHtml(p.id)}" aria-pressed="${active}">${active ? "Filtering — clear" : "Filter to this program"}</button>
        <button class="text-link details-btn" type="button" data-details="${escapeHtml(p.id)}">Details <span class="arr">→</span></button>
      </div>
    </article>`;
  }).join("");
  const filterTo = (id) => { setProgram(id); if (state.program === id) gotoSection("experiments"); };
  host.querySelectorAll("[data-filter]").forEach((btn) => btn.addEventListener("click", () => filterTo(btn.getAttribute("data-filter"))));
  host.querySelectorAll("[data-details]").forEach((btn) => btn.addEventListener("click", () => openProgram(programById.get(btn.getAttribute("data-details")))));
  // whole card is a mouse shortcut to filter; buttons handle themselves
  host.querySelectorAll(".program-card").forEach((card) => {
    card.addEventListener("click", (e) => { if (e.target.closest("button")) return; filterTo(card.getAttribute("data-program")); });
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
      </div>`, `program/${p.id}`);
  const btn = document.getElementById("drawerFilterBtn");
  if (btn) btn.addEventListener("click", () => { closeDrawer(); setProgram(p.id); restoreFilterFocus(); gotoSection("experiments"); });
}

/* ---------------------------------------------------------------- experiments table */
const TABLE_COLS = [
  { key: "id", label: "Experiment" },
  { key: "programs", label: "Programs" },
  { key: "anchor_ready", label: "Status" },
  { key: "run_surface", label: "Run surface" },
  { key: "needs", label: "Curation tasks" }
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
    <tr data-experiment="${escapeHtml(e.id)}">
      <td data-label="Experiment"><button class="row-open" type="button" aria-label="${escapeHtml(e.id)} — open details"><span class="cell-id">${escapeHtml(e.id)}</span><span class="cell-title">${escapeHtml(cleanTitle(e.title))}</span></button></td>
      <td class="col-progs" data-label="Programs"><span class="prog-chips">${e.programs.map((id) => `<button class="prog-chip pill-link" type="button" data-program-pill="${escapeHtml(id)}" style="background:${programColor(id)}" title="Filter to ${escapeHtml(titleForProgram(id))}" aria-label="Filter to ${escapeHtml(titleForProgram(id))}">${escapeHtml(programCode(programById.get(id) || { id, title: titleForProgram(id) }))}</button>`).join("")}</span></td>
      <td data-label="Status"><span class="status-tag ${e.anchor_ready === "yes" ? "ready" : "notready"}">${e.anchor_ready === "yes" ? "Anchor-ready" : "Not anchor-ready"}</span></td>
      <td class="col-enum" data-label="Run surface">${escapeHtml(humanizeToken(e.run_surface) || "—")}</td>
      <td class="col-mono" data-label="Curation tasks">${(e.needs || []).length ? escapeHtml((e.needs || []).map(humanizeToken).join(", ")) : '<span class="none">none</span>'}</td>
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
      // renderExperiments re-creates the header; keep focus on the column the user sorted
      document.querySelector(`[data-sort="${key}"] .th-btn`)?.focus();
    });
  });
  const openRow = (el) => openExperiment(experimentById.get(el.closest("[data-experiment]").getAttribute("data-experiment")));
  document.querySelectorAll("#experimentTable .row-open").forEach((btn) => btn.addEventListener("click", () => openRow(btn)));
  document.querySelectorAll("#experimentTable [data-experiment]").forEach((tr) => {
    tr.addEventListener("click", (e) => { if (e.target.closest(".row-open") || e.target.closest("[data-program-pill]")) return; openRow(tr); });
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
  openDrawer("Experiment", cleanTitle(e.title), e.summary, [
    ["Id", `<span class="mono-val">${escapeHtml(e.id)}</span>`],
    ["Programs", e.programs.map((id) => programPill(id)).join(" ")],
    ["Status", `<span class="status-tag ${e.anchor_ready === "yes" ? "ready" : "notready"}">${e.anchor_ready === "yes" ? "Anchor-ready" : "Not anchor-ready"}</span>`],
    ["Run surface", e.run_surface ? `<span class="mono-val">${escapeHtml(humanizeToken(e.run_surface))}</span>` : ""],
    ["Smoke test", hasSmoke ? `<code>${escapeHtml(e.smoke_command)}</code>` : '<span class="muted-val">none yet</span>'],
    ["Curation tasks", (e.needs || []).length ? (e.needs || []).map((n) => `<span class="pill">${escapeHtml(humanizeToken(n))}</span>`).join(" ") : '<span class="muted-val">none</span>'],
    ["Tags", (e.tags || []).length ? (e.tags || []).map((t) => `<span class="pill">${escapeHtml(t)}</span>`).join(" ") : ""],
    ["Artifacts", listText(e.recognized_artifacts) ? escapeHtml(listText(e.recognized_artifacts)) : ""],
    ["Files", `${formatNumber(e.total_files)} · ${formatBytes(e.total_size_bytes).n} ${formatBytes(e.total_size_bytes).unit}`]
  ], `<div class="drawer-section-label">Open</div><div class="drawer-links">
      ${e.primary_readme ? `<a class="text-link" href="${repoLink(e.primary_readme)}">README <span class="arr">→</span></a>` : ""}
      ${e.primary_report ? `<a class="text-link" href="${repoLink(e.primary_report)}">Report <span class="arr">→</span></a>` : ""}
      <a class="text-link" href="${repoLink(e.path)}">Folder <span class="arr">→</span></a>
    </div>`, `exp/${e.id}`);
}

/* ---------------------------------------------------------------- queue board */
/* priority = one warm urgency ramp (hot -> calm); headings are ink, ramp is on the spine/bar only */
const QUEUE_META = {
  P0: { color: "#c2410c", meaning: "do next" },
  P1: { color: "#dd8a3e", meaning: "soon" },
  P2: { color: "#e8c69c", meaning: "later" }
};
function renderQueue() {
  const rows = filteredQueue();
  document.getElementById("queueCount").innerHTML = `<b>${formatNumber(rows.length)}</b> of ${formatNumber(DATA.queue.length)} shown`;
  if (!rows.length) {
    document.getElementById("queueBoard").innerHTML = emptyState("No queued probes match the active filters.");
    wireEmptyReset();
    return;
  }
  const priorities = ["P0", "P1", "P2"];
  document.getElementById("queueBoard").innerHTML = priorities.map((pr) => {
    const items = rows.filter((i) => i.priority === pr);
    const meta = QUEUE_META[pr] || { color: COLORS.muted, meaning: "" };
    const cards = items.length ? items.map((item) => `
      <article class="queue-card" data-queue="${escapeHtml(item.id)}" style="border-left:4px solid ${meta.color}">
        <div class="meta-line"><span>${escapeHtml(humanizeStatus(item.status))}</span><span class="effort-chip">${escapeHtml(humanizeStatus(item.effort))} effort</span></div>
        <h4><button class="card-open" type="button" data-open="${escapeHtml(item.id)}">${escapeHtml(cleanTitle(item.title))}</button></h4>
        <p class="question">${escapeHtml(item.question)}</p>
        <div class="pill-row">${(item.programs || []).slice(0, 3).map((id) => programPill(id)).join("")}</div>
      </article>`).join("") : '<div class="queue-empty">No matching probes.</div>';
    return `<div class="queue-column" style="--col-color:${meta.color}">
      <div class="queue-col-head"><h3>${pr}</h3><span class="count">${items.length}</span><span class="meaning">${meta.meaning}</span></div>
      ${cards}
    </div>`;
  }).join("");
  const openQ = (id) => openQueue(DATA.queue.find((i) => i.id === id));
  document.querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", () => openQ(b.getAttribute("data-open"))));
  document.querySelectorAll(".queue-card").forEach((card) => {
    card.addEventListener("click", (e) => { if (e.target.closest("button")) return; openQ(card.getAttribute("data-queue")); });
  });
}

function openQueue(item) {
  if (!item) return;
  const meta = QUEUE_META[item.priority] || {};
  openDrawer(`Queue · ${item.priority}${meta.meaning ? " · " + meta.meaning : ""}`, cleanTitle(item.title), item.question, [
    ["Queue id", `<span class="mono-val">${escapeHtml(item.id)}</span>`],
    ["Priority", `${escapeHtml(item.priority)} · ${escapeHtml(humanizeStatus(item.status))} · ${escapeHtml(humanizeStatus(item.effort))} effort`],
    ["Programs", (item.programs || []).map((id) => programPill(id)).join(" ")],
    ["Hypothesis", escapeHtml(item.hypothesis)],
    ["Minimal protocol", escapeHtml(item.minimal_protocol)],
    ["Success signal", escapeHtml(item.success_signal)],
    ["Failure signal", escapeHtml(item.failure_signal)],
    ["Expected artifacts", listText(item.expected_artifacts) ? escapeHtml(listText(item.expected_artifacts)) : ""],
    ["Next step", escapeHtml(item.next_step)]
  ], item.source ? `<div class="drawer-section-label">Open</div><a class="text-link" href="${repoLink(item.source)}">Source on GitHub <span class="arr">→</span></a>` : "", `queue/${item.id}`);
}

/* ---------------------------------------------------------------- claims */
function renderClaims() {
  const claims = filteredClaims();
  const host = document.getElementById("claimCards");
  if (!claims.length) { host.innerHTML = emptyState("No claims match the active filters."); wireEmptyReset(); return; }
  host.innerHTML = claims.map((c) => {
    const color = STATUS_COLORS[c.status] || COLORS.violet;
    return `<article class="claim-card" data-claim="${escapeHtml(c.id)}" style="--accent:${color}">
      <div class="claim-top">
        <span class="claim-id">${escapeHtml(c.id)}</span>
        <span class="status-chip has-glyph" style="--status-color:${color}"><span class="status-glyph-badge" style="background:${color}">${STATUS_GLYPH[c.status] || ""}</span>${escapeHtml(statusLabel(c.status))}</span>
      </div>
      <h4><button class="card-open" type="button" data-open-claim="${escapeHtml(c.id)}">${escapeHtml(c.title)}</button></h4>
      <p class="claim-summary">${escapeHtml(c.summary)}</p>
      <div class="pill-row">${splitList(c.programs).map((id) => programPill(id)).join("")}</div>
    </article>`;
  }).join("");
  const openC = (id) => openClaim(DATA.claims.find((c) => c.id === id));
  host.querySelectorAll("[data-open-claim]").forEach((b) => b.addEventListener("click", () => openC(b.getAttribute("data-open-claim"))));
  host.querySelectorAll(".claim-card").forEach((card) => {
    card.addEventListener("click", (e) => { if (e.target.closest("button")) return; openC(card.getAttribute("data-claim")); });
  });
}

function openClaim(c) {
  if (!c) return;
  const color = STATUS_COLORS[c.status] || COLORS.violet;
  openDrawer(`Claim ${c.id}`, c.title, c.summary, [
    ["Status", `<span class="status-chip has-glyph" style="--status-color:${color}"><span class="status-glyph-badge" style="background:${color}">${STATUS_GLYPH[c.status] || ""}</span>${escapeHtml(statusLabel(c.status))}</span>`],
    ["Programs", splitList(c.programs).map((id) => programPill(id)).join(" ")],
    ["Evidence", escapeHtml(c.evidence)],
    ["Implication", escapeHtml(c.implication)]
  ], `<div class="drawer-section-label">Open</div><a class="text-link" href="${repoLink("knowledge/claims/index.md")}">Claim ledger on GitHub <span class="arr">→</span></a>`, `claim/${c.id}`);
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
      <p class="narrative-path">${escapeHtml(path)}</p>
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(body || "Open the source document for current notes.")}</p>
      <a class="text-link" href="${repoLink(path)}">Open document <span class="arr">→</span></a>
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
  barChart("needsChart", needRows.length ? needRows : DATA.charts.needs.map((r) => ({ ...r, label: humanizeToken(r.id) })), {
    // neutral slate (matches the other magnitude charts) — keeps rose exclusively the negative-result signal
    label: "curation tasks", labelW: 190, color: "#455468", empty: "No curation tasks in this view.",
    onBar: (id) => { setNeed(id); gotoSection("experiments"); }
  });
  const runRows = countRows(exp, (e) => e.run_surface);
  barChart("runSurfaceChart", runRows, { label: "run surfaces", labelW: 170, rowH: 30, color: COLORS.teal, empty: "No experiments in this view." });
  // priority mix reacts to the active filter (consistent with the queue board)
  const fq = filteredQueue();
  const prioCounts = { P0: 0, P1: 0, P2: 0 };
  fq.forEach((i) => { if (prioCounts[i.priority] != null) prioCounts[i.priority] += 1; });
  const prioRows = ["P0", "P1", "P2"].map((id) => ({ id, value: prioCounts[id] }));
  barChart("queuePriorityChart", prioRows, {
    label: "queue priorities", labelW: 50, rowH: 34, empty: "No queued probes in this view.",
    colors: { P0: QUEUE_META.P0.color, P1: QUEUE_META.P1.color, P2: QUEUE_META.P2.color }
  });
  // descriptive (non-status) magnitude charts use one neutral hue so status colors stay meaningful
  barChart("artifactKindChart", DATA.charts.artifact_kinds.map((r) => ({ ...r, label: humanizeToken(r.id) })), { label: "artifact manifest kinds", labelW: 130, color: "#455468" });
  barChart("extensionChart", DATA.charts.extensions, { label: "file extension counts", labelW: 70, rowH: 24, color: "#455468" });
}

/* ---------------------------------------------------------------- render */
/* capture a selector for the focused control so we can restore it after a re-render */
function focusKey() {
  const ae = document.activeElement;
  if (!ae || ae === document.body) return null;
  const prog = ae.getAttribute && ae.getAttribute("data-program");
  if (prog) return ae.closest("#programLegend") ? `#programLegend [data-program="${prog}"]` : `#programCards [data-program="${prog}"]`;
  const metric = ae.getAttribute && ae.getAttribute("data-metric");
  if (metric) return `[data-metric="${metric}"]`;
  const mapnode = ae.getAttribute && ae.getAttribute("data-mapnode");
  if (mapnode) return `[data-mapnode="${mapnode}"]`;
  const cell = ae.getAttribute && ae.getAttribute("data-cell");
  if (cell) { const col = ae.getAttribute("data-col"); return `#programHeatmap [data-cell="${cell}"]${col ? `[data-col="${col}"]` : ""}`; }
  const bar = ae.getAttribute && ae.getAttribute("data-bar");
  if (bar) return `[data-bar="${bar}"]`;
  const sortTh = ae.closest && ae.closest("[data-sort]");
  if (sortTh) return `[data-sort="${sortTh.getAttribute("data-sort")}"] .th-btn`;
  return null;
}

function render() {
  hideTooltip(); // re-render destroys the hovered/focused chart node; don't strand its tooltip
  const fk = focusKey();
  updateFilterStatus();
  updateMetrics();
  renderPrograms();
  renderExperiments();
  renderQueue();
  renderClaims();
  renderCharts();
  syncURL();
  if (fk) { const el = document.querySelector(fk); if (el) el.focus(); }
}

/* Mirror filter state into ?query so a filtered view is shareable/deep-linkable
   (the section #hash is left untouched). Wrapped for file:// where history may throw. */
function syncURL() {
  const params = new URLSearchParams();
  if (state.program !== "all") params.set("program", state.program);
  if (state.ready !== "all") params.set("ready", state.ready);
  if (state.need !== "all") params.set("need", state.need);
  if (state.search) params.set("q", state.search);
  const qs = params.toString();
  try {
    history.replaceState(null, "", `${location.pathname}${qs ? "?" + qs : ""}${location.hash}`);
  } catch (_) { /* file:// or sandboxed: ignore */ }
}

function seedStateFromURL() {
  let params;
  try { params = new URLSearchParams(location.search); } catch (_) { return; }
  const program = params.get("program");
  if (program && (program === "all" || programById.has(program))) state.program = program;
  const ready = params.get("ready");
  if (ready === "yes" || ready === "no") state.ready = ready;
  const need = params.get("need");
  const knownNeeds = new Set(DATA.experiments.flatMap((e) => e.needs || []));
  if (need && knownNeeds.has(need)) state.need = need; // ignore stale ?need= values
  const q = params.get("q");
  if (q) state.search = q;
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
        if (link) {
          link.classList.add("active");
          // on mobile the topnav is a horizontally-scrollable strip; keep the active item in view
          // so the "you are here" indicator isn't stranded off-screen-right
          const nav = link.parentElement;
          if (nav && nav.scrollWidth > nav.clientWidth + 4) {
            link.scrollIntoView({ inline: "nearest", block: "nearest", behavior: prefersReducedMotion() ? "auto" : "smooth" });
          }
        }
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
  restoreFilterFocus(); // the Reset control hides itself on clear — don't strand focus on body
}

let searchTimer = null;
function onSearchInput(value) {
  state.search = value; // keep raw (incl. spaces); trimmed only at compare time
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

  els.drawerClose.addEventListener("click", dismissDrawer);
  els.drawerScrim.addEventListener("click", dismissDrawer);

  // cross-linking: any program pill anywhere filters the atlas
  document.addEventListener("click", (e) => {
    const pill = e.target.closest("[data-program-pill]");
    if (!pill) return;
    e.stopPropagation();
    const id = pill.getAttribute("data-program-pill");
    if (els.drawer.classList.contains("open")) closeDrawer();
    state.program = id; syncControls(); render();
    restoreFilterFocus(); // the activated pill is destroyed by re-render — park focus on a stable control
    gotoSection("experiments");
  });

  // Back/Forward is authoritative for BOTH the filter scope and the open entity
  window.addEventListener("popstate", () => {
    state.program = "all"; state.ready = "all"; state.need = "all"; state.search = "";
    seedStateFromURL();
    syncControls();
    render();
    const h = decodeURIComponent(location.hash.replace(/^#/, ""));
    if (/^(exp|claim|queue|program)\//.test(h)) {
      openingFromHistory = true; openEntityFromHash(h); openingFromHistory = false;
    } else if (els.drawer.classList.contains("open")) {
      drawerPushed = false; // the entry was already popped by this navigation
      closeDrawer();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !e.repeat) {
      if (els.drawer.classList.contains("open")) dismissDrawer();
      else if (els.tooltip.classList.contains("visible")) hideTooltip(); // WCAG 1.4.13: dismissible without moving focus
    }
    trapDrawerFocus(e);
    const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName);
    const drawerOpen = els.drawer.classList.contains("open");
    if (e.key === "/" && !typing && !drawerOpen) {
      e.preventDefault();
      const rail = els.filterRail && els.filterRail.classList.contains("visible");
      (rail ? els.searchTop : els.search || els.searchTop)?.focus();
    }
  });
  // a focus-anchored tooltip is position:fixed; drop it on a genuine scroll so it can't strand
  // mid-page — but ignore the scroll-into-view that a focus itself triggers (or it would hide the
  // tooltip the focus just showed)
  window.addEventListener("scroll", () => {
    if (!els.tooltip.classList.contains("visible")) return;
    if (performance.now() - lastTooltipFocusAt < 250) return;
    hideTooltip();
  }, { passive: true });
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

/* keep the --header-h / --rail-h tokens accurate so sticky panels + anchor jumps clear the chrome */
function setupChromeSizing() {
  const topbar = document.querySelector(".topbar");
  // Measure the filter rail's RENDERED height even while it is hidden (it is display:none until
  // scrolled, and wraps to ~2 rows on mobile). A nav-jump always ends with the rail visible, so
  // .band scroll-margin must reserve its real height at every breakpoint or mobile headings land
  // under the sticky chrome. visibility:hidden + position:fixed makes this measure flash-free.
  const measureRail = () => {
    const rail = els.filterRail;
    if (!rail) return null;
    if (rail.classList.contains("visible")) return rail.offsetHeight || null;
    rail.style.visibility = "hidden";
    rail.classList.add("visible");
    const h = rail.offsetHeight;
    rail.classList.remove("visible");
    rail.style.visibility = "";
    return h || null;
  };
  const apply = () => {
    if (topbar && topbar.offsetHeight) document.documentElement.style.setProperty("--header-h", `${topbar.offsetHeight}px`);
    const rh = measureRail();
    if (rh) document.documentElement.style.setProperty("--rail-h", `${rh}px`);
  };
  apply();
  window.addEventListener("resize", apply);
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
seedStateFromURL();
syncControls();
bindEvents();
setupScrollSpy();
setupFilterRail();
setupChromeSizing();
setupMotion();
render();

// deep-link: open the entity drawer if the URL points at one
(() => {
  const h = decodeURIComponent(location.hash.replace(/^#/, ""));
  if (/^(exp|claim|queue|program)\//.test(h)) { openingFromHistory = true; openEntityFromHash(h); openingFromHistory = false; }
  else if (h && document.getElementById(h)) {
    // plain section anchor (#queue, #experiments, …): the browser's one-time fragment jump ran
    // against empty containers before render() injected content, landing at the top. Re-align to
    // the now-populated section so fresh-load / shared / bookmarked section URLs work.
    requestAnimationFrame(() => document.getElementById(h).scrollIntoView({ behavior: "auto", block: "start" }));
  }
})();

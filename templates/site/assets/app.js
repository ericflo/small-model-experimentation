"use strict";

/* Progressive enhancement for the research log. Every feature here is an
   enhancement: pages read fine with JavaScript disabled. Untrusted strings
   (search index, data files) only ever reach the DOM via textContent. */

const ROOT = document.body.dataset.root || "";

/* ------------------------------------------------------------ site search */

(function initSearch() {
  const input = document.getElementById("site-search");
  const list = document.getElementById("search-results");
  if (!input || !list) return;
  let index = null;
  let active = -1;

  async function ensureIndex() {
    if (index) return index;
    try {
      const res = await fetch(`${ROOT}data/search-index.json`);
      index = await res.json();
    } catch (err) {
      index = [];
    }
    return index;
  }

  function close() {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    active = -1;
  }

  function scoreHit(item, tokens) {
    // every token must match somewhere; title matches rank above metadata,
    // metadata above body text
    let total = 0;
    const t = item.t.toLowerCase();
    const x = (item.x || "").toLowerCase();
    const b = (item.b || "").toLowerCase();
    for (const token of tokens) {
      if (t.includes(token)) total += 3;
      else if (x.includes(token)) total += 2;
      else if (b.includes(token)) total += 1;
      else return 0;
    }
    return total;
  }

  function render(hits, query) {
    list.replaceChildren();
    hits.slice(0, 10).forEach((hit) => {
      const li = document.createElement("li");
      li.setAttribute("role", "option");
      const a = document.createElement("a");
      a.href = ROOT + hit.u;
      const kind = document.createElement("span");
      kind.className = "kind";
      kind.textContent = hit.k;
      const title = document.createElement("span");
      title.textContent = hit.t;
      const text = document.createElement("span");
      text.className = "hit-text";
      text.textContent = (hit.d ? `${hit.d} — ` : "") + hit.x.slice(0, 140);
      a.append(kind, title, text);
      li.appendChild(a);
      list.appendChild(li);
    });
    const meta = document.createElement("li");
    meta.className = "search-meta";
    if (hits.length === 0) {
      meta.textContent = `No matches for “${query}”`;
    } else {
      const expCount = hits.filter((hit) => hit.k === "experiment").length;
      const label = `${hits.length} match${hits.length === 1 ? "" : "es"}`;
      if (expCount > 0) {
        const a = document.createElement("a");
        a.href = `${ROOT}experiments/?q=${encodeURIComponent(query)}`;
        a.textContent = `${label} — open ${expCount} experiment${expCount === 1 ? "" : "s"} in the explorer →`;
        meta.appendChild(a);
      } else {
        meta.textContent = `${label} (no experiments — claims, programs, and queue hits are listed above)`;
      }
    }
    list.appendChild(meta);
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
    active = -1;
  }

  async function run() {
    const query = input.value.trim().toLowerCase();
    if (query.length < 2) return close();
    const idx = await ensureIndex();
    const tokens = query.split(/\s+/).filter(Boolean);
    const hits = idx
      .map((item) => ({ item, score: scoreHit(item, tokens) }))
      .filter((entry) => entry.score > 0)
      .sort((a, b) => b.score - a.score || (b.item.d || "").localeCompare(a.item.d || ""))
      .map((entry) => entry.item);
    render(hits, query);
  }

  let timer = 0;
  input.addEventListener("input", () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(run, 120);
  });
  input.addEventListener("focus", ensureIndex);
  input.addEventListener("keydown", (event) => {
    const links = list.querySelectorAll("a");
    if (event.key === "Escape") { close(); input.blur(); }
    else if (event.key === "ArrowDown" && links.length) {
      event.preventDefault();
      active = Math.min(active + 1, links.length - 1);
      links.forEach((a, i) => a.classList.toggle("active", i === active));
    } else if (event.key === "ArrowUp" && links.length) {
      event.preventDefault();
      active = Math.max(active - 1, 0);
      links.forEach((a, i) => a.classList.toggle("active", i === active));
    } else if (event.key === "Enter" && links.length) {
      event.preventDefault();
      // no arrow selection yet: Enter opens the top hit
      links[Math.max(active, 0)].click();
    }
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".site-search")) close();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) {
      event.preventDefault();
      input.focus();
    }
  });
})();

/* --------------------------------------------------------------- explorer */

(function initExplorer() {
  const listEl = document.getElementById("explorer");
  if (!listEl) return;
  const search = document.getElementById("explorer-search");
  const program = document.getElementById("explorer-program");
  const track = document.getElementById("explorer-track");
  const outcome = document.getElementById("explorer-outcome");
  const sort = document.getElementById("explorer-sort");
  const count = document.getElementById("explorer-count");
  const empty = document.getElementById("explorer-empty");
  const tabs = Array.from(document.querySelectorAll("[data-status-tab]"));
  const items = Array.from(listEl.querySelectorAll(".explorer-item"));
  let statusTab = "finished"; // primary path: latest finished, newest first

  // deep text (full README/report vocabulary) rides in the shared search index
  // so the explorer page itself stays small; loaded on first keystroke
  let deepText = null;
  async function ensureDeepText() {
    if (deepText) return deepText;
    deepText = new Map();
    try {
      const res = await fetch(`${ROOT}data/search-index.json`);
      const idx = await res.json();
      idx.forEach((entry) => {
        if (entry.k === "experiment" && entry.b) deepText.set(entry.u, entry.b);
      });
    } catch (err) { /* fall back to card text alone */ }
    items.forEach((item) => {
      const link = item.querySelector("h3 a");
      if (!link) return;
      const url = link.getAttribute("href").replace(/^\.\.\//, "");
      const extra = deepText.get(url);
      if (extra) item.dataset.deep = extra;
    });
    return deepText;
  }

  // date-range filter set from ?day= / ?week= (activity chart bars)
  let rangeFilter = null; // {from, to, label}

  function weekEnd(startIso) {
    const d = new Date(startIso + "T00:00:00Z");
    d.setUTCDate(d.getUTCDate() + 6);
    return d.toISOString().slice(0, 10);
  }

  function apply() {
    const tokens = (search.value || "").trim().toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    items.forEach((item) => {
      // a search spans every status; only the tabs themselves scope by status
      const okStatus = statusTab === "all" || tokens.length > 0 || item.dataset.status === statusTab;
      const okOutcome = !outcome || !outcome.value || item.dataset.outcome === outcome.value;
      const programs = (item.dataset.programs || "").split(/\s+/);
      const okProgram = !program.value || programs.includes(program.value);
      const okTrack = !track.value || item.dataset.track === track.value;
      const hay = (item.dataset.text || "") + " " + (item.dataset.deep || "");
      const okSearch = tokens.every((token) => hay.includes(token));
      const date = item.dataset.date || "";
      const okRange = !rangeFilter || (date >= rangeFilter.from && date <= rangeFilter.to);
      const show = okStatus && okOutcome && okProgram && okTrack && okSearch && okRange;
      item.hidden = !show;
      if (show) shown += 1;
    });
    const key = sort.value;
    // rank = build order (matches the prev/next pager chain on detail pages)
    const rank = (el) => Number(el.dataset.rank || 0);
    const sorted = items.slice().sort((a, b) => {
      if (key === "title") return (a.dataset.title || "").localeCompare(b.dataset.title || "");
      if (key === "figs") return Number(b.dataset.figs || 0) - Number(a.dataset.figs || 0) || rank(a) - rank(b);
      if (key === "date-asc") return (a.dataset.date || "9999").localeCompare(b.dataset.date || "9999") || rank(b) - rank(a);
      return (b.dataset.date || "").localeCompare(a.dataset.date || "") || rank(a) - rank(b);
    });
    sorted.forEach((item) => listEl.appendChild(item));
    const scoped = statusTab !== "all" && tokens.length === 0;
    const scope = scoped ? ` ${statusTab === "in-progress" ? "in progress" : statusTab}` : "";
    count.textContent = `${shown}${scope}` + (rangeFilter ? ` · ${rangeFilter.label}` : "");
    if (empty) empty.hidden = shown !== 0;
    const emptyScope = document.getElementById("empty-scope");
    if (emptyScope) emptyScope.textContent = scoped ? `in the ${statusTab === "in-progress" ? "In progress" : "Finished"} tab — try “All”` : "";
    const reset = document.getElementById("filter-reset");
    if (reset) {
      reset.hidden = !(search.value || program.value || track.value || (outcome && outcome.value) || rangeFilter || statusTab !== "finished");
    }
  }

  function setTab(value, focus) {
    statusTab = value;
    tabs.forEach((t) => {
      const on = t.dataset.statusTab === value;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1; // roving tabindex: one stop, arrows move within
    });
    if (focus) { const sel = tabs.find((t) => t.dataset.statusTab === value); if (sel) sel.focus(); }
    apply();
  }
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => setTab(t.dataset.statusTab));
    t.addEventListener("keydown", (e) => {
      let j = null;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % tabs.length;
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i - 1 + tabs.length) % tabs.length;
      else if (e.key === "Home") j = 0;
      else if (e.key === "End") j = tabs.length - 1;
      if (j !== null) { e.preventDefault(); setTab(tabs[j].dataset.statusTab, true); }
    });
  });

  let timer = 0;
  search.addEventListener("input", () => {
    window.clearTimeout(timer);
    ensureDeepText().then(apply);
    timer = window.setTimeout(apply, 120);
  });
  [program, track, sort, outcome].forEach((el) => el && el.addEventListener("change", apply));
  function resetFilters() {
    search.value = "";
    program.value = "";
    track.value = "";
    if (outcome) outcome.value = "";
    sort.value = "date";
    rangeFilter = null;
    window.history.replaceState(null, "", window.location.pathname);
    setTab("finished");
  }
  ["explorer-clear", "filter-reset"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", resetFilters);
  });
  const params = new URLSearchParams(window.location.search);
  const dayParam = params.get("day");
  const weekParam = params.get("week");
  if (dayParam && /^\d{4}-\d{2}-\d{2}$/.test(dayParam)) {
    rangeFilter = { from: dayParam, to: dayParam, label: dayParam };
    statusTab = "all"; // the activity bars count every run, so date-drilldown spans all statuses
  } else if (weekParam && /^\d{4}-\d{2}-\d{2}$/.test(weekParam)) {
    rangeFilter = { from: weekParam, to: weekEnd(weekParam), label: `week of ${weekParam}` };
    statusTab = "all";
  }
  if (params.get("q")) {
    search.value = params.get("q");
    statusTab = "all"; // a search from anywhere should span every experiment, not just finished
    ensureDeepText().then(apply);
  }
  if (params.get("program")) program.value = params.get("program");
  const statusParam = params.get("status");
  if (["finished", "in-progress", "all"].includes(statusParam)) statusTab = statusParam;
  const outcomeParam = params.get("outcome");
  if (outcome && outcomeParam) outcome.value = outcomeParam;
  const sortParam = params.get("sort");
  if (sortParam && Array.from(sort.options).some((option) => option.value === sortParam)) sort.value = sortParam;
  setTab(statusTab); // sets aria-selected + roving tabindex (one focusable tab) + applies filters
})();

/* ----------------------------------------------------------- claims filter */

(function initClaims() {
  const board = document.getElementById("claims-board");
  if (!board) return;
  const search = document.getElementById("claims-search");
  const count = document.getElementById("claims-count");
  const empty = document.getElementById("claims-empty");
  const filters = Array.from(document.querySelectorAll(".claim-filter"));
  const cards = Array.from(board.querySelectorAll(".claim-card"));
  const groups = Array.from(board.querySelectorAll(".claim-group"));
  let status = "";

  function apply() {
    const tokens = (search.value || "").trim().toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    cards.forEach((c) => {
      const okStatus = !status || c.dataset.status === status;
      const hay = c.dataset.text || "";
      const show = okStatus && tokens.every((t) => hay.includes(t));
      c.hidden = !show;
      if (show) shown += 1;
    });
    groups.forEach((g) => { g.hidden = !g.querySelector(".claim-card:not([hidden])"); });
    count.textContent = `${shown}${status ? " " + status.toLowerCase() : ""} of ${cards.length}`;
    if (empty) empty.hidden = shown !== 0;
  }
  filters.forEach((f) => f.addEventListener("click", () => {
    status = f.dataset.claimStatus;
    filters.forEach((o) => o.setAttribute("aria-pressed", String(o === f)));
    apply();
  }));
  let timer = 0;
  search.addEventListener("input", () => { window.clearTimeout(timer); timer = window.setTimeout(apply, 120); });
  const q = new URLSearchParams(window.location.search).get("q");
  if (q) search.value = q;
  apply();
})();

/* -------------------------------------------------------- sortable tables */

(function initTables() {
  document.querySelectorAll("table[data-sortable]").forEach((table) => {
    const body = table.tBodies[0];
    if (!body || body.rows.length < 2) return;
    Array.from(table.tHead ? table.tHead.rows[0].cells : []).forEach((th, col) => {
      th.tabIndex = 0;
      th.title = "Sort by this column";
      const sortBy = () => {
        const dir = th.classList.contains("sort-asc") ? -1 : 1;
        table.querySelectorAll("th").forEach((other) => {
          other.classList.remove("sort-asc", "sort-desc");
          other.removeAttribute("aria-sort");
        });
        th.classList.add(dir === 1 ? "sort-asc" : "sort-desc");
        th.setAttribute("aria-sort", dir === 1 ? "ascending" : "descending");
        const rows = Array.from(body.rows);
        const value = (row) => (row.cells[col] ? row.cells[col].textContent.trim() : "");
        const numeric = rows.every((row) => {
          const v = value(row).replace(/[%,+~≈]/g, "");
          return v === "" || !Number.isNaN(parseFloat(v));
        });
        rows.sort((a, b) => {
          const va = value(a);
          const vb = value(b);
          if (numeric) {
            const na = parseFloat(va.replace(/[%,+~≈]/g, ""));
            const nb = parseFloat(vb.replace(/[%,+~≈]/g, ""));
            return dir * ((Number.isNaN(na) ? -Infinity : na) - (Number.isNaN(nb) ? -Infinity : nb));
          }
          return dir * va.localeCompare(vb);
        });
        rows.forEach((row) => body.appendChild(row));
      };
      th.addEventListener("click", sortBy);
      th.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); sortBy(); }
      });
    });
  });
})();

/* ---------------------------------------------------------------- lightbox */

(function initLightbox() {
  const box = document.getElementById("lightbox");
  if (!box) return;
  const img = box.querySelector("img");
  const caption = box.querySelector("figcaption");
  // gallery anchors, plus inline doc images that are NOT already inside one
  const targets = Array.from(document.querySelectorAll(".fig-link, .doc-section img, .doc-page img"))
    .filter((el) => el.tagName !== "IMG" || !el.closest(".fig-link"));
  const entries = [];
  targets.forEach((el) => {
    const src = el.tagName === "IMG" ? el.getAttribute("src") : el.getAttribute("href");
    if (!src || !/\.(png|svg|jpe?g|gif|webp)(\?|$)/i.test(src)) return;
    const label = el.tagName === "IMG" ? (el.getAttribute("alt") || src) : (el.querySelector("img") ? el.querySelector("img").alt || src : src);
    const index = entries.length;
    entries.push({ src, label });
    el.addEventListener("click", (event) => {
      event.preventDefault();
      open(index);
    });
    if (el.tagName === "IMG") el.style.cursor = "zoom-in";
  });
  if (!entries.length) return;
  let current = 0;
  let lastFocus = null;

  function open(index) {
    lastFocus = document.activeElement;
    current = index;
    show();
    box.hidden = false;
    document.body.style.overflow = "hidden";
    box.querySelector(".lightbox-close").focus();
  }
  function show() {
    const entry = entries[current];
    img.src = entry.src;
    img.alt = entry.label;
    caption.textContent = `${entry.label} (${current + 1}/${entries.length})`;
    box.querySelector(".lightbox-prev").style.visibility = entries.length > 1 ? "visible" : "hidden";
    box.querySelector(".lightbox-next").style.visibility = entries.length > 1 ? "visible" : "hidden";
  }
  function close() {
    box.hidden = true;
    document.body.style.overflow = "";
    if (lastFocus) lastFocus.focus();
  }
  box.querySelector(".lightbox-close").addEventListener("click", close);
  box.querySelector(".lightbox-prev").addEventListener("click", () => { current = (current - 1 + entries.length) % entries.length; show(); });
  box.querySelector(".lightbox-next").addEventListener("click", () => { current = (current + 1) % entries.length; show(); });
  box.addEventListener("click", (event) => { if (event.target === box) close(); });
  document.addEventListener("keydown", (event) => {
    if (box.hidden) return;
    if (event.key === "Escape") close();
    else if (event.key === "ArrowLeft") { current = (current - 1 + entries.length) % entries.length; show(); }
    else if (event.key === "ArrowRight") { current = (current + 1) % entries.length; show(); }
    else if (event.key === "Tab") {
      // contain focus inside the dialog while it is open
      const focusable = Array.from(box.querySelectorAll("button")).filter((b) => b.style.visibility !== "hidden");
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      else if (!box.contains(document.activeElement)) { event.preventDefault(); first.focus(); }
    }
  });
})();

/* ------------------------------------------------------------ chart polish */

(function initCharts() {
  const charts = document.querySelectorAll(".viz-chart");
  if (!charts.length) return;
  const tip = document.createElement("div");
  tip.className = "viz-tip";
  tip.hidden = true;
  document.body.appendChild(tip);

  function moveTip(event) {
    const pad = 12;
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    const rect = tip.getBoundingClientRect();
    if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }

  charts.forEach((chart) => {
    chart.querySelectorAll("[data-viz-pt]").forEach((pt) => {
      pt.addEventListener("mouseenter", (event) => {
        tip.textContent = pt.dataset.vizLabel || "";
        tip.hidden = !tip.textContent;
        moveTip(event);
      });
      pt.addEventListener("mousemove", moveTip);
      pt.addEventListener("mouseleave", () => { tip.hidden = true; });
    });
    const keys = chart.querySelectorAll(".viz-key");
    const sel = (i) => `[data-viz-series="${i}"], [data-viz-series-line="${i}"], [data-viz-series-label="${i}"]`;
    // reconcile every series to its toggle state (off = dimmed + inert)
    function applyState() {
      keys.forEach((key) => {
        const off = key.classList.contains("off");
        key.setAttribute("aria-pressed", String(!off));
        chart.querySelectorAll(sel(key.dataset.vizToggle)).forEach((el) => {
          el.style.opacity = off ? "0.1" : "";
          if (el.hasAttribute("data-viz-pt")) el.style.pointerEvents = off ? "none" : "";
        });
      });
    }
    keys.forEach((key) => {
      const index = key.dataset.vizToggle;
      key.addEventListener("click", () => { key.classList.toggle("off"); applyState(); });
      // hover one series to spotlight it; the rest fade until mouseout
      key.addEventListener("mouseenter", () => {
        if (key.classList.contains("off")) return;
        keys.forEach((other) => {
          if (other === key) return;
          chart.querySelectorAll(sel(other.dataset.vizToggle)).forEach((el) => { el.style.opacity = "0.14"; });
        });
      });
      key.addEventListener("mouseleave", applyState);
    });
  });
})();

/* ------------------------------------------------------------ data preview */

(function initDataPreview() {
  const section = document.getElementById("data");
  if (!section) return;
  const viewer = section.querySelector(".data-viewer");
  const nameEl = section.querySelector(".data-viewer-name");
  const bodyEl = section.querySelector(".data-viewer-body");
  if (!viewer) return;

  function jsonNode(key, value, depth) {
    const wrap = document.createElement("div");
    if (value !== null && typeof value === "object") {
      const details = document.createElement("details");
      details.open = depth < 2;
      const summary = document.createElement("summary");
      const keySpan = document.createElement("span");
      keySpan.className = "json-key";
      keySpan.textContent = key;
      const meta = document.createElement("span");
      const size = Array.isArray(value) ? value.length : Object.keys(value).length;
      meta.textContent = Array.isArray(value) ? ` [${size}]` : ` {${size}}`;
      summary.append(keySpan, meta);
      details.appendChild(summary);
      const items = Array.isArray(value)
        ? value.map((item, i) => [String(i), item])
        : Object.entries(value);
      items.slice(0, 200).forEach(([k, v]) => details.appendChild(jsonNode(k, v, depth + 1)));
      if (items.length > 200) {
        const more = document.createElement("div");
        more.textContent = `… ${items.length - 200} more entries (open the raw file)`;
        details.appendChild(more);
      }
      wrap.appendChild(details);
    } else {
      const keySpan = document.createElement("span");
      keySpan.className = "json-key";
      keySpan.textContent = `${key}: `;
      const valSpan = document.createElement("span");
      valSpan.className = typeof value === "number" ? "json-num" : "json-str";
      valSpan.textContent = JSON.stringify(value);
      wrap.append(keySpan, valSpan);
    }
    return wrap;
  }

  function parseCsvLine(line) {
    const cells = [];
    let cell = "";
    let quoted = false;
    for (let i = 0; i < line.length; i += 1) {
      const ch = line[i];
      if (quoted) {
        if (ch === '"' && line[i + 1] === '"') { cell += '"'; i += 1; }
        else if (ch === '"') quoted = false;
        else cell += ch;
      } else if (ch === '"' && cell === "") {
        quoted = true;
      } else if (ch === ",") {
        cells.push(cell);
        cell = "";
      } else {
        cell += ch;
      }
    }
    cells.push(cell);
    return cells;
  }

  function csvTable(text) {
    const rows = text.split(/\r?\n/).filter((line) => line.length).slice(0, 60).map(parseCsvLine);
    const table = document.createElement("table");
    table.className = "md-table";
    rows.forEach((cells, i) => {
      const tr = document.createElement("tr");
      cells.forEach((cell) => {
        const td = document.createElement(i === 0 ? "th" : "td");
        td.textContent = cell;
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    return table;
  }

  section.querySelectorAll(".data-preview").forEach((button) => {
    button.addEventListener("click", async () => {
      const src = button.dataset.src;
      nameEl.textContent = src.replace(/^files\//, "");
      bodyEl.replaceChildren();
      viewer.hidden = false;
      try {
        const res = await fetch(src);
        const text = await res.text();
        if (src.endsWith(".json")) {
          bodyEl.appendChild(jsonNode("root", JSON.parse(text), 0));
        } else {
          bodyEl.appendChild(csvTable(text));
        }
      } catch (err) {
        const p = document.createElement("p");
        p.textContent = `Could not load ${src} — open it directly instead.`;
        bodyEl.appendChild(p);
      }
      viewer.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  });
  viewer.querySelector(".data-viewer-close").addEventListener("click", () => { viewer.hidden = true; });
})();

const DATA = JSON.parse(document.getElementById("site-data").textContent);

const COLORS = {
  teal: "#147d7e",
  green: "#2f7d4f",
  amber: "#b46b12",
  rose: "#b23a48",
  violet: "#6750a4",
  blue: "#2f5f98",
  ink: "#17212b",
  muted: "#5d6977",
  line: "#d7dde5",
  paper: "#f7f8fa"
};

const PALETTE = [COLORS.teal, COLORS.amber, COLORS.rose, COLORS.blue, COLORS.green, COLORS.violet];

const state = {
  search: "",
  program: "all",
  ready: "all",
  need: "all"
};

const els = {
  search: document.getElementById("globalSearch"),
  program: document.getElementById("programFilter"),
  ready: document.getElementById("readyFilter"),
  need: document.getElementById("needFilter"),
  reset: document.getElementById("resetFilters"),
  tooltip: document.getElementById("tooltip"),
  drawer: document.getElementById("detailDrawer"),
  drawerClose: document.getElementById("drawerClose"),
  drawerContent: document.getElementById("drawerContent")
};

const programById = new Map(DATA.programs.map((program) => [program.id, program]));
const experimentById = new Map(DATA.experiments.map((experiment) => [experiment.id, experiment]));

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB"];
  let scaled = value / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && scaled >= 1024; index += 1) {
    scaled /= 1024;
    unit = units[index];
  }
  return `${scaled.toFixed(scaled >= 10 ? 1 : 2)} ${unit}`;
}

function titleForProgram(programId) {
  return programById.get(programId)?.title || programId;
}

function repoLink(path) {
  if (!path) return "";
  return `${DATA.repo.github}/blob/main/${path}`;
}

function listText(items) {
  return (items || []).join(", ");
}

function textBlob(item) {
  return Object.values(item)
    .flatMap((value) => (Array.isArray(value) ? value : [value]))
    .join(" ")
    .toLowerCase();
}

function matchesSearch(item) {
  if (!state.search) return true;
  return textBlob(item).includes(state.search.toLowerCase());
}

function matchesProgram(item, field = "programs") {
  if (state.program === "all") return true;
  return (item[field] || []).includes(state.program);
}

function filteredExperiments() {
  return DATA.experiments.filter((experiment) => {
    if (!matchesSearch(experiment)) return false;
    if (!matchesProgram(experiment)) return false;
    if (state.ready !== "all" && experiment.anchor_ready !== state.ready) return false;
    if (state.need !== "all" && !(experiment.needs || []).includes(state.need)) return false;
    return true;
  });
}

function filteredQueue() {
  return DATA.queue.filter((item) => matchesSearch(item) && matchesProgram(item));
}

function clear(element) {
  element.replaceChildren();
}

function svg(tag, attrs = {}, children = []) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  children.forEach((child) => element.appendChild(child));
  return element;
}

function showTooltip(event, html) {
  els.tooltip.innerHTML = html;
  els.tooltip.style.left = `${event.clientX}px`;
  els.tooltip.style.top = `${event.clientY}px`;
  els.tooltip.classList.add("visible");
}

function hideTooltip() {
  els.tooltip.classList.remove("visible");
}

function attachTooltip(element, html) {
  element.addEventListener("mousemove", (event) => showTooltip(event, html));
  element.addEventListener("mouseleave", hideTooltip);
}

function setProgram(programId) {
  state.program = programId;
  els.program.value = programId;
  render();
}

function openDrawer(title, rows, body = "") {
  const detailRows = rows
    .filter((row) => row[1] !== undefined && row[1] !== "")
    .map(([label, value]) => `<dt>${escapeHtml(label)}</dt><dd>${value}</dd>`)
    .join("");
  els.drawerContent.innerHTML = `
    <h2>${escapeHtml(title)}</h2>
    ${body ? `<div class="drawer-body">${body}</div>` : ""}
    <dl>${detailRows}</dl>
  `;
  els.drawer.classList.add("open");
  els.drawer.setAttribute("aria-hidden", "false");
}

function closeDrawer() {
  els.drawer.classList.remove("open");
  els.drawer.setAttribute("aria-hidden", "true");
}

function renderFilters() {
  els.program.innerHTML = [
    '<option value="all">All programs</option>',
    ...DATA.programs.map((program) => `<option value="${escapeHtml(program.id)}">${escapeHtml(program.title)}</option>`)
  ].join("");

  const needs = Array.from(new Set(DATA.experiments.flatMap((experiment) => experiment.needs || []))).sort();
  els.need.innerHTML = [
    '<option value="all">All curation needs</option>',
    ...needs.map((need) => `<option value="${escapeHtml(need)}">${escapeHtml(need)}</option>`)
  ].join("");
}

function renderMetrics() {
  const metrics = [
    ["Experiments", DATA.summary.experiments, "Self-contained experiment folders"],
    ["Programs", DATA.summary.programs, "Durable lines of inquiry"],
    ["Future queue", DATA.summary.future_proposals, "Structured next probes"],
    ["Anchor-ready", DATA.summary.anchor_ready, "Reusable experiment anchors"],
    ["Claims", DATA.summary.claims, "Shared evidence statements"],
    ["Artifact manifests", DATA.summary.artifact_manifests, "Experiments with manifests"],
    ["Files indexed", DATA.summary.total_files, "Tracked corpus files"],
    ["Tracked size", formatBytes(DATA.summary.total_size_bytes), "Repository-local data footprint"]
  ];
  document.getElementById("metricGrid").innerHTML = metrics
    .map(
      ([label, value, help], index) => `
      <article class="metric-card">
        <p class="eyebrow">${escapeHtml(label)}</p>
        <strong style="color:${PALETTE[index % PALETTE.length]}">${escapeHtml(typeof value === "number" ? formatNumber(value) : value)}</strong>
        <span>${escapeHtml(help)}</span>
      </article>`
    )
    .join("");
}

function barChart(targetId, rows, options = {}) {
  const target = document.getElementById(targetId);
  clear(target);
  const width = options.width || 720;
  const rowHeight = options.rowHeight || 34;
  const margin = { top: 18, right: 58, bottom: 24, left: options.left || 210 };
  const height = Math.max(170, margin.top + margin.bottom + rows.length * rowHeight);
  const maxValue = Math.max(1, ...rows.map((row) => Number(row.value || 0)));
  const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": options.label || "bar chart" });
  rows.forEach((row, index) => {
    const y = margin.top + index * rowHeight;
    const barWidth = ((width - margin.left - margin.right) * Number(row.value || 0)) / maxValue;
    const color = options.colors?.[row.id] || PALETTE[index % PALETTE.length];
    const label = row.label || row.id;
    chart.appendChild(svg("text", { x: 8, y: y + 20, class: "chart-label" }, [document.createTextNode(label)]));
    const rect = svg("rect", {
      x: margin.left,
      y: y + 5,
      width: Math.max(2, barWidth),
      height: 20,
      rx: 5,
      fill: color
    });
    attachTooltip(rect, `<strong>${escapeHtml(label)}</strong><br>${formatNumber(row.value)}`);
    chart.appendChild(rect);
    chart.appendChild(svg("text", { x: margin.left + barWidth + 8, y: y + 20, class: "chart-value" }, [document.createTextNode(formatNumber(row.value))]));
  });
  target.appendChild(chart);
}

function donutChart() {
  const target = document.getElementById("readinessDonut");
  clear(target);
  const ready = DATA.summary.anchor_ready;
  const total = DATA.summary.experiments;
  const needs = Math.max(0, total - ready);
  const radius = 72;
  const circumference = 2 * Math.PI * radius;
  const readyArc = (ready / total) * circumference;
  const chart = svg("svg", { viewBox: "0 0 260 260", role: "img", "aria-label": "readiness donut" });
  chart.appendChild(svg("circle", { cx: 130, cy: 118, r: radius, fill: "none", stroke: "#e5e9ef", "stroke-width": 30 }));
  chart.appendChild(
    svg("circle", {
      cx: 130,
      cy: 118,
      r: radius,
      fill: "none",
      stroke: COLORS.green,
      "stroke-width": 30,
      "stroke-linecap": "round",
      "stroke-dasharray": `${readyArc} ${circumference - readyArc}`,
      transform: "rotate(-90 130 118)"
    })
  );
  chart.appendChild(svg("text", { x: 130, y: 112, "text-anchor": "middle", class: "chart-value", style: "font-size:34px" }, [document.createTextNode(String(ready))]));
  chart.appendChild(svg("text", { x: 130, y: 136, "text-anchor": "middle", class: "chart-label" }, [document.createTextNode("anchor-ready")]));
  chart.appendChild(svg("text", { x: 130, y: 222, "text-anchor": "middle", class: "chart-label" }, [document.createTextNode(`${needs} need curation`)]));
  target.appendChild(chart);
}

function programNetwork() {
  const target = document.getElementById("programNetwork");
  clear(target);
  const width = 920;
  const height = 520;
  const cx = width / 2;
  const cy = height / 2 + 10;
  const radius = 190;
  const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "program network" });
  chart.appendChild(svg("circle", { cx, cy, r: 64, fill: "#f0f6f7", stroke: COLORS.teal, "stroke-width": 2 }));
  chart.appendChild(svg("text", { x: cx, y: cy - 6, "text-anchor": "middle", class: "chart-value" }, [document.createTextNode("Research")]));
  chart.appendChild(svg("text", { x: cx, y: cy + 14, "text-anchor": "middle", class: "chart-value" }, [document.createTextNode("Programs")]));

  DATA.programs.forEach((program, index) => {
    const angle = -Math.PI / 2 + (index / DATA.programs.length) * Math.PI * 2;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    const nodeRadius = 18 + Math.min(25, program.experiment_count / 3.4);
    const color = PALETTE[index % PALETTE.length];
    chart.appendChild(svg("line", { x1: cx, y1: cy, x2: x, y2: y, stroke: "#cbd3dd", "stroke-width": 1.4 }));
    const node = svg("circle", {
      cx: x,
      cy: y,
      r: nodeRadius,
      fill: state.program === program.id ? color : "#ffffff",
      stroke: color,
      "stroke-width": state.program === program.id ? 4 : 2,
      tabindex: 0,
      role: "button"
    });
    node.addEventListener("click", () => setProgram(program.id));
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") setProgram(program.id);
    });
    attachTooltip(
      node,
      `<strong>${escapeHtml(program.title)}</strong><br>${formatNumber(program.experiment_count)} experiments<br>${formatNumber(program.queue_count)} queued proposals`
    );
    chart.appendChild(node);
    const labelX = x + Math.cos(angle) * (nodeRadius + 12);
    const labelY = y + Math.sin(angle) * (nodeRadius + 12);
    chart.appendChild(
      svg(
        "text",
        {
          x: labelX,
          y: labelY,
          "text-anchor": Math.cos(angle) > 0.22 ? "start" : Math.cos(angle) < -0.22 ? "end" : "middle",
          class: "node-label"
        },
        [document.createTextNode(program.title.replace(" And ", " & "))]
      )
    );
    const qx = cx + Math.cos(angle) * (radius + 74);
    const qy = cy + Math.sin(angle) * (radius + 74);
    chart.appendChild(svg("circle", { cx: qx, cy: qy, r: 6 + program.queue_count * 1.2, fill: color, opacity: 0.28 }));
  });
  target.appendChild(chart);
}

function scatterPlot() {
  const target = document.getElementById("experimentScatter");
  clear(target);
  const rows = filteredExperiments();
  const width = 520;
  const height = 340;
  const margin = { top: 24, right: 22, bottom: 44, left: 54 };
  const maxFiles = Math.max(1, ...DATA.experiments.map((experiment) => experiment.total_files));
  const maxSize = Math.max(1, ...DATA.experiments.map((experiment) => experiment.total_size_bytes));
  const log = (value) => Math.log10(Math.max(1, value));
  const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "experiment size scatter plot" });
  chart.appendChild(svg("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, stroke: COLORS.line }));
  chart.appendChild(svg("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, stroke: COLORS.line }));
  chart.appendChild(svg("text", { x: width / 2, y: height - 10, "text-anchor": "middle", class: "axis-label" }, [document.createTextNode("tracked size, log scale")]));
  chart.appendChild(svg("text", { x: 14, y: height / 2, transform: `rotate(-90 14 ${height / 2})`, "text-anchor": "middle", class: "axis-label" }, [document.createTextNode("files")]));
  rows.forEach((experiment) => {
    const x = margin.left + (log(experiment.total_size_bytes) / log(maxSize)) * (width - margin.left - margin.right);
    const y = height - margin.bottom - (experiment.total_files / maxFiles) * (height - margin.top - margin.bottom);
    const color = experiment.anchor_ready === "yes" ? COLORS.green : COLORS.rose;
    const point = svg("circle", { cx: x, cy: y, r: 4.5, fill: color, opacity: 0.76 });
    attachTooltip(
      point,
      `<strong>${escapeHtml(experiment.id)}</strong><br>${formatNumber(experiment.total_files)} files<br>${formatBytes(experiment.total_size_bytes)}`
    );
    point.addEventListener("click", () => openExperiment(experiment));
    chart.appendChild(point);
  });
  target.appendChild(chart);
}

function programHeatmap() {
  const target = document.getElementById("programHeatmap");
  clear(target);
  const metrics = ["Experiments", "Ready", "Queue", "Claims"];
  const width = 740;
  const rowHeight = 32;
  const height = 44 + DATA.programs.length * rowHeight;
  const labelWidth = 280;
  const cellWidth = 96;
  const maxValues = {
    Experiments: Math.max(...DATA.programs.map((program) => program.experiment_count), 1),
    Ready: Math.max(...DATA.programs.map((program) => program.anchor_ready_count), 1),
    Queue: Math.max(...DATA.programs.map((program) => program.queue_count), 1),
    Claims: Math.max(...DATA.programs.map((program) => program.claim_count), 1)
  };
  const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "program readiness heatmap" });
  metrics.forEach((metric, index) => {
    chart.appendChild(svg("text", { x: labelWidth + index * cellWidth + 8, y: 22, class: "chart-label" }, [document.createTextNode(metric)]));
  });
  DATA.programs.forEach((program, rowIndex) => {
    const y = 38 + rowIndex * rowHeight;
    chart.appendChild(svg("text", { x: 8, y: y + 20, class: "chart-label" }, [document.createTextNode(program.title)]));
    const values = [program.experiment_count, program.anchor_ready_count, program.queue_count, program.claim_count];
    values.forEach((value, colIndex) => {
      const metric = metrics[colIndex];
      const intensity = value / maxValues[metric];
      const color = [COLORS.blue, COLORS.green, COLORS.amber, COLORS.violet][colIndex];
      const rect = svg("rect", {
        x: labelWidth + colIndex * cellWidth,
        y,
        width: 78,
        height: 23,
        rx: 5,
        fill: color,
        opacity: 0.12 + intensity * 0.78
      });
      attachTooltip(rect, `<strong>${escapeHtml(program.title)}</strong><br>${metric}: ${formatNumber(value)}`);
      chart.appendChild(rect);
      chart.appendChild(svg("text", { x: labelWidth + colIndex * cellWidth + 39, y: y + 17, "text-anchor": "middle", class: "chart-value" }, [document.createTextNode(formatNumber(value))]));
    });
  });
  target.appendChild(chart);
}

function claimGraph() {
  const target = document.getElementById("claimGraph");
  clear(target);
  const width = 680;
  const height = 430;
  const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "claim program graph" });
  const claims = DATA.claims;
  const programs = DATA.programs;
  const claimY = (index) => 40 + index * ((height - 80) / Math.max(1, claims.length - 1));
  const programY = (index) => 32 + index * ((height - 64) / Math.max(1, programs.length - 1));
  const programPositions = new Map();
  programs.forEach((program, index) => {
    programPositions.set(program.id, { x: width - 170, y: programY(index) });
  });
  claims.forEach((claim, claimIndex) => {
    const y = claimY(claimIndex);
    splitList(claim.programs).forEach((programId) => {
      const targetPosition = programPositions.get(programId);
      if (!targetPosition) return;
      chart.appendChild(svg("line", { x1: 148, y1: y, x2: targetPosition.x, y2: targetPosition.y, stroke: "#cbd3dd", "stroke-width": 1.1 }));
    });
  });
  claims.forEach((claim, index) => {
    const y = claimY(index);
    const node = svg("circle", { cx: 118, cy: y, r: 14, fill: statusColor(claim.status), opacity: 0.9 });
    attachTooltip(node, `<strong>${escapeHtml(claim.id)}: ${escapeHtml(claim.title)}</strong><br>${escapeHtml(claim.status)}`);
    node.addEventListener("click", () => openClaim(claim));
    chart.appendChild(node);
    chart.appendChild(svg("text", { x: 8, y: y + 4, class: "node-label" }, [document.createTextNode(claim.id)]));
  });
  programs.forEach((program) => {
    const pos = programPositions.get(program.id);
    const node = svg("rect", { x: pos.x, y: pos.y - 10, width: 22, height: 22, rx: 5, fill: COLORS.blue, opacity: 0.75 });
    attachTooltip(node, `<strong>${escapeHtml(program.title)}</strong><br>${program.claim_count} claims`);
    node.addEventListener("click", () => setProgram(program.id));
    chart.appendChild(node);
    chart.appendChild(svg("text", { x: pos.x + 30, y: pos.y + 5, class: "node-label" }, [document.createTextNode(program.title)]));
  });
  target.appendChild(chart);
}

function splitList(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  return String(value).split(";").filter(Boolean);
}

function statusColor(status) {
  return {
    Confirmed: COLORS.green,
    Promising: COLORS.amber,
    Negative: COLORS.rose,
    Open: COLORS.blue,
    Retired: COLORS.muted
  }[status] || COLORS.violet;
}

function renderPrograms() {
  const search = state.search.toLowerCase();
  const rows = DATA.programs.filter((program) => {
    if (state.program !== "all" && program.id !== state.program) return false;
    if (!search) return true;
    return textBlob(program).includes(search);
  });
  document.getElementById("programCards").innerHTML = rows
    .map(
      (program) => `
      <article class="program-card" tabindex="0" data-program="${escapeHtml(program.id)}">
        <p class="eyebrow">${escapeHtml(program.id)}</p>
        <h3>${escapeHtml(program.title)}</h3>
        <p>${escapeHtml(program.focus)}</p>
        <div class="pill-row">
          <span class="pill teal">${formatNumber(program.experiment_count)} experiments</span>
          <span class="pill green">${formatNumber(program.anchor_ready_count)} ready</span>
          <span class="pill amber">${formatNumber(program.queue_count)} queued</span>
          <span class="pill rose">${formatNumber(program.claim_count)} claims</span>
        </div>
      </article>`
    )
    .join("");
  document.querySelectorAll(".program-card").forEach((card) => {
    const programId = card.getAttribute("data-program");
    card.addEventListener("click", () => setProgram(programId));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") setProgram(programId);
    });
  });
}

function renderExperiments() {
  const rows = filteredExperiments();
  document.getElementById("experimentCount").textContent = `${formatNumber(rows.length)} experiments shown`;
  const html = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Experiment</th>
            <th>Programs</th>
            <th>Ready</th>
            <th>Run surface</th>
            <th>Needs</th>
          </tr>
        </thead>
        <tbody>
          ${rows
            .slice(0, 180)
            .map(
              (experiment) => `
            <tr data-experiment="${escapeHtml(experiment.id)}">
              <td><strong>${escapeHtml(experiment.id)}</strong><br><span>${escapeHtml(experiment.title)}</span></td>
              <td>${escapeHtml(experiment.programs.map(titleForProgram).join(", "))}</td>
              <td><span class="pill ${experiment.anchor_ready === "yes" ? "green" : "rose"}">${experiment.anchor_ready}</span></td>
              <td>${escapeHtml(experiment.run_surface || "")}</td>
              <td>${escapeHtml((experiment.needs || []).join(", ") || "none")}</td>
            </tr>`
            )
            .join("")}
        </tbody>
      </table>
    </div>`;
  document.getElementById("experimentTable").innerHTML = rows.length ? html : '<div class="empty-state">No experiments match the active filters.</div>';
  document.querySelectorAll("[data-experiment]").forEach((row) => {
    row.addEventListener("click", () => openExperiment(experimentById.get(row.getAttribute("data-experiment"))));
  });
}

function openExperiment(experiment) {
  if (!experiment) return;
  const body = `<p>${escapeHtml(experiment.summary)}</p>`;
  openDrawer(experiment.id, [
    ["Title", escapeHtml(experiment.title)],
    ["Programs", escapeHtml(experiment.programs.map(titleForProgram).join(", "))],
    ["Tags", escapeHtml(listText(experiment.tags))],
    ["Anchor-ready", `<span class="pill ${experiment.anchor_ready === "yes" ? "green" : "rose"}">${escapeHtml(experiment.anchor_ready)}</span>`],
    ["Run surface", escapeHtml(experiment.run_surface)],
    ["Smoke command", escapeHtml(experiment.smoke_command)],
    ["Needs", escapeHtml((experiment.needs || []).join(", ") || "none")],
    ["Artifacts", escapeHtml(listText(experiment.recognized_artifacts))],
    ["Manifests", escapeHtml(listText(experiment.manifest_kinds))],
    ["Files", escapeHtml(formatNumber(experiment.total_files))],
    ["Size", escapeHtml(formatBytes(experiment.total_size_bytes))],
    ["README", experiment.primary_readme ? `<a href="${repoLink(experiment.primary_readme)}">Open in GitHub</a>` : ""],
    ["Report", experiment.primary_report ? `<a href="${repoLink(experiment.primary_report)}">Open in GitHub</a>` : ""]
  ], body);
}

function renderQueue() {
  const rows = filteredQueue();
  document.getElementById("queueCount").textContent = `${formatNumber(rows.length)} proposals shown`;
  const priorities = ["P0", "P1", "P2"];
  document.getElementById("queueBoard").innerHTML = priorities
    .map((priority) => {
      const items = rows.filter((item) => item.priority === priority);
      return `
        <div class="queue-column">
          <h3>${priority}</h3>
          ${items
            .map(
              (item) => `
            <article class="queue-card" data-queue="${escapeHtml(item.id)}">
              <p class="eyebrow">${escapeHtml(item.status)} / ${escapeHtml(item.effort)}</p>
              <h3>${escapeHtml(item.title)}</h3>
              <p class="question">${escapeHtml(item.question)}</p>
              <div class="pill-row">${item.programs
                .slice(0, 3)
                .map((programId) => `<span class="pill amber">${escapeHtml(titleForProgram(programId))}</span>`)
                .join("")}</div>
            </article>`
            )
            .join("")}
        </div>`;
    })
    .join("");
  document.querySelectorAll("[data-queue]").forEach((card) => {
    card.addEventListener("click", () => openQueue(DATA.queue.find((item) => item.id === card.getAttribute("data-queue"))));
  });
}

function openQueue(item) {
  if (!item) return;
  openDrawer(item.title, [
    ["Queue id", escapeHtml(item.id)],
    ["Priority", escapeHtml(item.priority)],
    ["Status", escapeHtml(item.status)],
    ["Effort", escapeHtml(item.effort)],
    ["Programs", escapeHtml(item.programs.map(titleForProgram).join(", "))],
    ["Question", escapeHtml(item.question)],
    ["Hypothesis", escapeHtml(item.hypothesis)],
    ["Minimal protocol", escapeHtml(item.minimal_protocol)],
    ["Success signal", escapeHtml(item.success_signal)],
    ["Failure signal", escapeHtml(item.failure_signal)],
    ["Expected artifacts", escapeHtml(listText(item.expected_artifacts))],
    ["Next step", escapeHtml(item.next_step)],
    ["Source", item.source ? `<a href="${repoLink(item.source)}">Open in GitHub</a>` : ""]
  ]);
}

function renderClaims() {
  const claims = DATA.claims.filter((claim) => {
    if (!matchesSearch(claim)) return false;
    if (state.program === "all") return true;
    return splitList(claim.programs).includes(state.program);
  });
  document.getElementById("claimCards").innerHTML = claims
    .map(
      (claim) => `
      <article class="claim-card" data-claim="${escapeHtml(claim.id)}">
        <p class="eyebrow">${escapeHtml(claim.status)}</p>
        <h3>${escapeHtml(claim.id)}: ${escapeHtml(claim.title)}</h3>
        <p>${escapeHtml(claim.summary)}</p>
        <div class="pill-row">${splitList(claim.programs)
          .map((programId) => `<span class="pill violet">${escapeHtml(titleForProgram(programId))}</span>`)
          .join("")}</div>
      </article>`
    )
    .join("");
  document.querySelectorAll("[data-claim]").forEach((card) => {
    card.addEventListener("click", () => openClaim(DATA.claims.find((claim) => claim.id === card.getAttribute("data-claim"))));
  });
}

function openClaim(claim) {
  if (!claim) return;
  openDrawer(`${claim.id}: ${claim.title}`, [
    ["Status", `<span class="pill" style="background:${statusColor(claim.status)}22;color:${statusColor(claim.status)}">${escapeHtml(claim.status)}</span>`],
    ["Programs", escapeHtml(splitList(claim.programs).map(titleForProgram).join(", "))],
    ["Evidence", escapeHtml(claim.evidence)],
    ["Summary", escapeHtml(claim.summary)],
    ["Implication", escapeHtml(claim.implication)],
    ["Index", `<a href="${repoLink("knowledge/claims/index.md")}">Open in GitHub</a>`]
  ]);
}

function renderNarrative() {
  const items = [
    ["Synthesis", DATA.narrative.synthesis, "knowledge/synthesis.md"],
    ["Roadmap", DATA.narrative.roadmap, "knowledge/research_roadmap.md"],
    ["Patterns", DATA.narrative.patterns, "knowledge/patterns.md"]
  ];
  document.getElementById("narrativeCards").innerHTML = items
    .map(
      ([title, text, path]) => `
      <article class="narrative-card">
        <h3>${escapeHtml(title)}</h3>
        <p>${escapeHtml(text || "Open the source document for current notes.")}</p>
        <a href="${repoLink(path)}">Open source</a>
      </article>`
    )
    .join("");
}

function renderCharts() {
  programNetwork();
  donutChart();
  scatterPlot();
  programHeatmap();
  claimGraph();
  barChart("needsChart", DATA.charts.needs, { label: "curation needs", left: 190 });
  barChart("queuePriorityChart", DATA.charts.queue_priority, {
    label: "queue priorities",
    left: 70,
    colors: { P0: COLORS.rose, P1: COLORS.amber, P2: COLORS.blue }
  });
  barChart("runSurfaceChart", DATA.charts.run_surfaces, { label: "run surfaces", left: 180, rowHeight: 30 });
  barChart("artifactKindChart", DATA.charts.artifact_kinds, { label: "artifact manifest kinds", left: 150 });
  barChart("extensionChart", DATA.charts.extensions, { label: "file extension counts", left: 90, rowHeight: 27 });
}

function render() {
  renderPrograms();
  renderExperiments();
  renderQueue();
  renderClaims();
  renderCharts();
}

function bindEvents() {
  els.search.addEventListener("input", (event) => {
    state.search = event.target.value.trim();
    render();
  });
  els.program.addEventListener("change", (event) => {
    state.program = event.target.value;
    render();
  });
  els.ready.addEventListener("change", (event) => {
    state.ready = event.target.value;
    render();
  });
  els.need.addEventListener("change", (event) => {
    state.need = event.target.value;
    render();
  });
  els.reset.addEventListener("click", () => {
    state.search = "";
    state.program = "all";
    state.ready = "all";
    state.need = "all";
    els.search.value = "";
    els.program.value = "all";
    els.ready.value = "all";
    els.need.value = "all";
    render();
  });
  els.drawerClose.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeDrawer();
  });
}

renderFilters();
renderMetrics();
renderNarrative();
bindEvents();
render();

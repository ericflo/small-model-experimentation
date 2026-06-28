#!/usr/bin/env python3
"""Aggregate slot-repair distillation runs and write reports."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path("experiments/qwen_slot_repair_distillation")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_slot_repair_distillation/checkpoints")

ITERATION_ORDER = [
    "pilot_slot_repair_distill_s96_b5",
    "pilot_slot_repair_distill_s96_b3_w8",
    "pilot_slot_repair_gated_s96",
    "pilot_slot_repair_gated_s96_oracle_base",
    "pilot_slot_repair_gated_s96_value_stabilized",
    "main_slot_repair_distill_s512",
]


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def pct(value: Any) -> str:
    x = as_float(value)
    if math.isnan(x):
        return "n/a"
    return f"{100.0 * x:.1f}%"


def scalar(value: Any) -> str:
    x = as_float(value)
    if math.isnan(x):
        return "n/a"
    return f"{x:.2f}"


def load_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/metrics.csv")):
        rows.extend(read_csv(path))
    return rows


def iteration_sort_key(row: Dict[str, Any]) -> tuple[int, str]:
    run = row.get("run", "")
    idx = ITERATION_ORDER.index(run) if run in ITERATION_ORDER else len(ITERATION_ORDER)
    return idx, run


def choose_primary(rows: Sequence[Dict[str, Any]]) -> str:
    preferred = "main_slot_repair_distill_s512"
    if any(row.get("run") == preferred for row in rows):
        return preferred
    runs = sorted({row.get("run", "") for row in rows if row.get("run")})
    non_smoke = [run for run in runs if "smoke" not in run]
    candidates = non_smoke or runs
    if not candidates:
        return ""
    return max(candidates, key=lambda run: (RUNS / run / "metrics.csv").stat().st_mtime if (RUNS / run / "metrics.csv").exists() else 0.0)


def primary_rows(rows: Sequence[Dict[str, Any]], primary_run: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("run") == primary_run]


def fresh_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("split", "").startswith("fresh_")]


def paired_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return next((row for row in rows if row.get("split") == "fresh_paired_len24"), {})


def load_train_rows(primary_run: str) -> List[Dict[str, Any]]:
    path = RUNS / primary_run / "editor_train_log.csv"
    return read_csv(path) if path.exists() else []


def load_metadata(primary_run: str) -> Dict[str, Any]:
    path = RUNS / primary_run / "results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("metadata", {})


def write_checkpoint_manifest() -> None:
    rows: List[Dict[str, Any]] = []
    for path in sorted(CHECKPOINT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(CHECKPOINT_ROOT)
        run = rel.parts[0] if rel.parts else ""
        rows.append({"run": run, "artifact_path": str(path), "bytes": path.stat().st_size})
    write_csv(ROOT / "checkpoint_manifest.csv", rows)


def write_figures(rows: Sequence[Dict[str, Any]], train_rows: Sequence[Dict[str, Any]], all_rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[figures] matplotlib unavailable: {exc}")
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    fres = fresh_rows(rows)
    if fres:
        labels = [row["split"].replace("fresh_", "").replace("_len24", "") for row in fres]
        series = [("base", "base_executor_accuracy"), ("editor", "editor_executor_accuracy"), ("oracle", "oracle_executor_accuracy")]
        x = list(range(len(labels)))
        width = 0.24
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        for offset, (label, key) in zip([-1.0, 0.0, 1.0], series):
            ax.bar([i + offset * width for i in x], [as_float(row.get(key)) for row in fres], width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("executor accuracy")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", ncols=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "executor_accuracy.png", dpi=160)
        plt.close(fig)

    paired = paired_row(rows)
    if paired:
        metrics = [
            ("executor", "executor_accuracy"),
            ("program", "program_exact"),
            ("prefix", "state_prefix_fraction"),
            ("both correct", "pair_both_correct"),
            ("state consistency", "pair_state_consistency"),
        ]
        labels = [m[0] for m in metrics]
        series = [("base", "base"), ("editor", "editor"), ("oracle", "oracle")]
        x = list(range(len(labels)))
        width = 0.24
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for offset, (label, prefix) in zip([-1.0, 0.0, 1.0], series):
            ys = [as_float(paired.get(f"{prefix}_{key}")) for _, key in metrics]
            ax.bar([i + offset * width for i in x], ys, width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("fraction")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", ncols=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "paired_details.png", dpi=160)
        plt.close(fig)

    if train_rows:
        xs = [int(float(row.get("epoch", 0))) for row in train_rows]
        base = [as_float(row.get("val_base_executor_accuracy")) for row in train_rows]
        editor = [as_float(row.get("val_editor_executor_accuracy")) for row in train_rows]
        oracle = [as_float(row.get("val_oracle_executor_accuracy")) for row in train_rows]
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.plot(xs, base, marker="o", label="base")
        ax.plot(xs, editor, marker="o", label="editor")
        ax.plot(xs, oracle, marker="o", label="oracle")
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("editor epoch")
        ax.set_ylabel("validation executor accuracy")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(FIGURES / "training_curve.png", dpi=160)
        plt.close(fig)

    iter_rows = sorted(
        [
            row
            for row in all_rows
            if row.get("split") == "fresh_paired_len24" and "smoke" not in row.get("run", "")
        ],
        key=iteration_sort_key,
    )
    if iter_rows:
        labels = [row["run"].replace("pilot_slot_repair_", "pilot_").replace("main_slot_repair_", "main_") for row in iter_rows]
        x = list(range(len(labels)))
        width = 0.24
        fig, ax = plt.subplots(figsize=(11.5, 5.4))
        for offset, (label, key) in zip(
            [-1.0, 0.0, 1.0],
            [("base", "base_executor_accuracy"), ("editor", "editor_executor_accuracy"), ("oracle", "oracle_executor_accuracy")],
        ):
            ax.bar([i + offset * width for i in x], [as_float(row.get(key)) for row in iter_rows], width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("fresh paired executor accuracy")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", ncols=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "iteration_summary.png", dpi=160)
        plt.close(fig)


def markdown_table(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "|" + "|".join(["---"] * len(rows[0])) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def report_markdown(
    rows: Sequence[Dict[str, Any]],
    train_rows: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    primary_run: str,
    all_rows: Sequence[Dict[str, Any]],
) -> str:
    paired = paired_row(rows)
    fresh = fresh_rows(rows)
    best_epoch = ""
    if train_rows:
        best = max(train_rows, key=lambda row: as_float(row.get("val_editor_executor_accuracy")))
        best_epoch = str(best.get("epoch", ""))
    base = as_float(paired.get("base_executor_accuracy"))
    editor = as_float(paired.get("editor_executor_accuracy"))
    oracle = as_float(paired.get("oracle_executor_accuracy"))
    gap = (editor - base) / (oracle - base) if not any(math.isnan(x) for x in [base, editor, oracle]) and abs(oracle - base) > 1e-9 else math.nan
    compiler_args = metadata.get("compiler_args", {})
    args = metadata.get("args", {})
    best_threshold = metadata.get("best_editor_threshold", "n/a")
    paired_gate_precision = paired.get("editor_gate_precision")
    paired_gate_recall = paired.get("editor_gate_recall")
    paired_avg_edits = paired.get("editor_avg_edits")
    lines = [
        "# Qwen Slot Repair Distillation",
        "",
        "## Abstract",
        "",
        "This experiment tests whether local repair headroom can be distilled into a gated slot editor for a frozen Qwen-attached numeric compiler. "
        "The compiler first emits an executable modular-arithmetic program. Offline candidate search identifies a corrected local program when one exists. "
        "A small transformer editor is then trained to decide which init/op/arg slots to edit and what replacement values to emit from the base compiler trace, without candidate enumeration at evaluation time.",
        "",
        "## Setup",
        "",
        f"- Primary run: `{primary_run}`",
        f"- Qwen substrate: `{compiler_args.get('model_id', 'Qwen/Qwen3-4B')}`",
        f"- Modulus: `{compiler_args.get('modulus', 97)}`",
        f"- Max program length: `{compiler_args.get('max_steps', 24)}`",
        f"- Best editor epoch: `{best_epoch or 'n/a'}`",
        f"- Best validation threshold: `{best_threshold}`",
        f"- Editor target mode: `{args.get('editor_target_mode', 'oracle_or_gold')}`",
        f"- Unchanged-slot value loss weight: `{args.get('unchanged_value_loss_weight', 'n/a')}`",
        "",
        "Training labels come from exact offline trajectories. At evaluation time, the editor receives only the base compiler trace and predicts a single program. "
        "The oracle column is the local candidate-search ceiling, not an inference method.",
        "",
        "## Results",
        "",
        "### Fresh Splits",
        "",
    ]
    table_rows = [["Split", "Base", "Editor", "Oracle", "Gap recovered", "Editor in candidates"]]
    for row in fresh:
        table_rows.append(
            [
                row.get("split", ""),
                pct(row.get("base_executor_accuracy")),
                pct(row.get("editor_executor_accuracy")),
                pct(row.get("oracle_executor_accuracy")),
                pct(row.get("editor_oracle_gap_recovered")),
                pct(row.get("editor_in_candidate_set_fraction")),
            ]
        )
    lines.append(markdown_table(table_rows))
    lines.extend(["", "![Executor accuracy](../analysis/figures/executor_accuracy.png)", "", "### Fresh Paired Details", ""])
    detail_rows = [["Metric", "Base", "Editor", "Oracle"]]
    for label, key in [
        ("Executor accuracy", "executor_accuracy"),
        ("Program exact", "program_exact"),
        ("State prefix fraction", "state_prefix_fraction"),
        ("Pair both-correct", "pair_both_correct"),
        ("Pair state consistency", "pair_state_consistency"),
    ]:
        detail_rows.append([label, pct(paired.get(f"base_{key}")), pct(paired.get(f"editor_{key}")), pct(paired.get(f"oracle_{key}"))])
    lines.append(markdown_table(detail_rows))
    lines.extend(
        [
            "",
            f"The final paired editor uses {scalar(paired_avg_edits)} edits per program on average. "
            f"Against its training target definition, gate precision is {pct(paired_gate_precision)} and gate recall is {pct(paired_gate_recall)}.",
        ]
    )
    lines.extend(
        [
            "",
            "![Paired details](../analysis/figures/paired_details.png)",
            "",
            "### Training Dynamics",
            "",
            "The training curve tracks validation accuracy after each editor epoch.",
            "",
            "![Training curve](../analysis/figures/training_curve.png)",
            "",
            "### Iteration Summary",
            "",
            "The direct full-slot editor either copied the base program or damaged too many slots. "
            "The gated editor fixed that interface problem, and a small unchanged-slot value loss made one pilot improve fresh paired accuracy. "
            "The larger primary run did not preserve that fresh-split gain.",
            "",
        ]
    )
    iter_table = [["Run", "Validation", "Fresh paired base", "Fresh paired editor", "Fresh paired oracle", "Avg edits"]]
    for row in sorted(
        [
            item
            for item in all_rows
            if item.get("split") == "fresh_paired_len24" and "smoke" not in item.get("run", "")
        ],
        key=iteration_sort_key,
    ):
        val = next((candidate for candidate in all_rows if candidate.get("run") == row.get("run") and candidate.get("split") == "val_len24"), {})
        iter_table.append(
            [
                row.get("run", ""),
                f"{pct(val.get('base_executor_accuracy'))}->{pct(val.get('editor_executor_accuracy'))}",
                pct(row.get("base_executor_accuracy")),
                pct(row.get("editor_executor_accuracy")),
                pct(row.get("oracle_executor_accuracy")),
                scalar(row.get("editor_avg_edits")),
            ]
        )
    lines.append(markdown_table(iter_table))
    lines.extend(
        [
            "",
            "![Iteration summary](../analysis/figures/iteration_summary.png)",
            "",
            "## Interpretation",
            "",
            f"On the fresh paired split, the editor moves exact execution from {pct(base)} to {pct(editor)}. "
            f"The local oracle ceiling is {pct(oracle)}, so the editor recovers {pct(gap)} of the measured base-to-oracle gap.",
            "",
            "The main result is therefore not a successful distillation of the local oracle. "
            "The editor learns a real validation signal, but its selected edits do not transfer robustly to fresh prompt distributions. "
            "The high oracle ceiling shows that nearby corrected programs usually exist; the failure is the one-shot policy's ability to choose the right sparse edits and values from the base trace alone.",
            "",
            "## Limitations",
            "",
            "- The task is synthetic modular arithmetic.",
            "- Candidate labels use exact trajectories during training.",
            "- The frozen compiler and deterministic runtime are specialized.",
            "- The editor consumes engineered base-trace features, not only raw Qwen hidden states.",
            "- The main result is one primary seed.",
            "- The deployable editor emits one program and does not get to evaluate candidate executions at inference time.",
            "",
            "## Artifacts",
            "",
            "Small experiment files live in:",
            "",
            "```text",
            "experiments/qwen_slot_repair_distillation/",
            "```",
            "",
            "Large artifacts live in:",
            "",
            "```text",
            "large_artifacts/qwen_slot_repair_distillation/checkpoints/",
            "```",
            "",
            "Primary files:",
            "",
            "- `analysis/summary.md`",
            "- `analysis/final_metrics.csv`",
            "- `analysis/all_final_metrics.csv`",
            "- `analysis/figures/executor_accuracy.png`",
            "- `analysis/figures/paired_details.png`",
            "- `analysis/figures/training_curve.png`",
            "- `analysis/figures/iteration_summary.png`",
            f"- `runs/{primary_run}/metrics.csv`",
            f"- `runs/{primary_run}/editor_train_log.csv`",
            "- `reports/qwen_slot_repair_distillation_paper.md`",
            "- `reports/qwen_slot_repair_distillation_paper.html`",
            "- `checkpoint_manifest.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown: str, title: str) -> str:
    lines = markdown.splitlines()
    out: List[str] = []
    in_ul = False
    in_code = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            if in_code:
                out.append("</code></pre>")
                in_code = False
            else:
                out.append("<pre><code>")
                in_code = True
            i += 1
            continue
        if in_code:
            out.append(html.escape(line))
            i += 1
            continue
        if line.startswith("|") and line.endswith("|"):
            table_lines: List[str] = []
            while i < len(lines) and lines[i].startswith("|") and lines[i].endswith("|"):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2:
                header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
                body = table_lines[2:]
                out.append("<table><thead><tr>")
                out.extend(f"<th>{html.escape(cell)}</th>" for cell in header)
                out.append("</tr></thead><tbody>")
                for row in body:
                    cells = [cell.strip() for cell in row.strip("|").split("|")]
                    out.append("<tr>")
                    out.extend(f"<td>{html.escape(cell)}</td>" for cell in cells)
                    out.append("</tr>")
                out.append("</tbody></table>")
            continue
        if not line.strip():
            if in_ul:
                out.append("</ul>")
                in_ul = False
            i += 1
            continue
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("![") and "](" in line:
            alt = line[2:].split("]", 1)[0]
            src = line.split("](", 1)[1].rstrip(")")
            out.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{html.escape(line[2:])}</li>")
        else:
            out.append(f"<p>{html.escape(line)}</p>")
        i += 1
    if in_ul:
        out.append("</ul>")
    css = """
body { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #17201a; background: #fafafa; }
main { max-width: 980px; margin: 0 auto; padding: 48px 24px 72px; background: #fff; }
h1 { font-size: 38px; margin: 0 0 24px; }
h2 { margin-top: 34px; border-top: 1px solid #d8dfd9; padding-top: 22px; }
h3 { margin-top: 26px; }
p, li { line-height: 1.55; font-size: 16px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0 24px; font-size: 14px; }
th, td { border-bottom: 1px solid #d8dfd9; padding: 8px 10px; text-align: left; }
th { background: #f1f4f1; font-weight: 650; }
pre { background: #f1f4f1; padding: 14px; overflow-x: auto; border-radius: 6px; }
img { max-width: 100%; border: 1px solid #d8dfd9; border-radius: 6px; background: #fff; }
figcaption { font-size: 13px; color: #526052; margin-top: 6px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
"""
    return f"<!doctype html><html><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title><style>{css}</style></head><body><main>{''.join(out)}</main></body></html>\n"


def write_summary(rows: Sequence[Dict[str, Any]], primary_run: str) -> None:
    primary = primary_rows(rows, primary_run)
    lines = ["# Slot Repair Distillation Analysis Summary", "", f"Primary run: `{primary_run}`", "", "## Metrics", ""]
    table_rows = [["Split", "Base", "Editor", "Oracle", "Gap recovered", "Editor changed", "Editor in candidates"]]
    for row in primary:
        table_rows.append(
            [
                row.get("split", ""),
                pct(row.get("base_executor_accuracy")),
                pct(row.get("editor_executor_accuracy")),
                pct(row.get("oracle_executor_accuracy")),
                pct(row.get("editor_oracle_gap_recovered")),
                pct(row.get("editor_changed_fraction")),
                pct(row.get("editor_in_candidate_set_fraction")),
            ]
        )
    lines.append(markdown_table(table_rows))
    lines.append("")
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    primary_run = choose_primary(rows)
    primary = primary_rows(rows, primary_run)
    train_rows = load_train_rows(primary_run)
    metadata = load_metadata(primary_run)
    write_csv(ANALYSIS / "all_final_metrics.csv", rows)
    write_csv(ANALYSIS / "final_metrics.csv", primary)
    write_figures(primary, train_rows, rows)
    write_summary(rows, primary_run)
    report = report_markdown(primary, train_rows, metadata, primary_run, rows)
    (REPORTS / "qwen_slot_repair_distillation_paper.md").write_text(report)
    (REPORTS / "qwen_slot_repair_distillation_paper.html").write_text(markdown_to_html(report, "Qwen Slot Repair Distillation"))
    write_checkpoint_manifest()
    print(ANALYSIS / "summary.md")


if __name__ == "__main__":
    main()

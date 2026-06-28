#!/usr/bin/env python3
"""Aggregate on-policy repair-to-compiler runs and write reports."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path("experiments/qwen_onpolicy_repair_compiler")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_onpolicy_repair_compiler/checkpoints")


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


def markdown_table(rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return ""
    widths = [0 for _ in rows[0]]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    lines = []
    header = "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(rows[0])) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    lines.extend([header, sep])
    for row in rows[1:]:
        lines.append("| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(lines)


def load_metric_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/metrics.csv")):
        rows.extend(read_csv(path))
    return rows


def choose_primary(rows: Sequence[Dict[str, Any]]) -> str:
    runs = sorted({row.get("run", "") for row in rows if row.get("run")})
    if not runs:
        return ""
    preferred = "main_onpolicy_repair_s256"
    if preferred in runs:
        return preferred
    non_smoke = [run for run in runs if "smoke" not in run]
    candidates = non_smoke or runs
    return max(candidates, key=lambda run: (RUNS / run / "metrics.csv").stat().st_mtime if (RUNS / run / "metrics.csv").exists() else 0.0)


def primary_rows(rows: Sequence[Dict[str, Any]], primary_run: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("run") == primary_run]


def fresh_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("split", "").startswith("fresh_")]


def split_row(rows: Sequence[Dict[str, Any]], split: str) -> Dict[str, Any]:
    return next((row for row in rows if row.get("split") == split), {})


def run_split_row(rows: Sequence[Dict[str, Any]], run: str, split: str) -> Dict[str, Any]:
    return next((row for row in rows if row.get("run") == run and row.get("split") == split), {})


def load_train_rows(primary_run: str) -> List[Dict[str, Any]]:
    path = RUNS / primary_run / "train_log.csv"
    return read_csv(path) if primary_run and path.exists() else []


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
    final_fresh = fresh_rows(rows)
    if final_fresh:
        labels = [row["split"].replace("fresh_", "").replace("_len24", "") for row in final_fresh]
        series = [
            ("compiler", "executor_accuracy"),
            ("local repair ceiling", "repair_executor_accuracy"),
        ]
        if any("trace_learned_executor_accuracy" in row for row in final_fresh):
            series.append(("learned verifier", "trace_learned_executor_accuracy"))
        x = list(range(len(labels)))
        width = 0.72 / max(1, len(series))
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        offsets = [i - (len(series) - 1) / 2 for i in range(len(series))]
        for offset, (label, key) in zip(offsets, series):
            ax.bar([i + offset * width for i in x], [as_float(row.get(key)) for row in final_fresh], width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("executor accuracy")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "executor_accuracy.png", dpi=160)
        plt.close(fig)

    paired = split_row(rows, "fresh_paired_len24")
    if paired:
        metrics = [
            ("answer", "executor_accuracy", "repair_executor_accuracy"),
            ("program", "program_exact", "repair_program_exact"),
            ("state prefix", "state_prefix_fraction", "repair_state_prefix_fraction"),
            ("both correct", "executor_pair_both_correct", "repair_pair_both_correct"),
            ("state consistency", "compiler_pair_state_consistency", "repair_pair_state_consistency"),
        ]
        labels = [item[0] for item in metrics]
        x = list(range(len(labels)))
        width = 0.32
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar([i - width / 2 for i in x], [as_float(paired.get(base_key)) for _, base_key, _ in metrics], width, label="compiler")
        ax.bar([i + width / 2 for i in x], [as_float(paired.get(repair_key)) for _, _, repair_key in metrics], width, label="local repair ceiling")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("fraction")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "paired_details.png", dpi=160)
        plt.close(fig)

    if train_rows:
        xs = list(range(len(train_rows)))
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        for label, key in [
            ("validation compiler", "val_len24_executor_accuracy"),
            ("fresh paired compiler", "fresh_paired_len24_executor_accuracy"),
            ("validation repair ceiling", "val_len24_repair_executor_accuracy"),
            ("fresh paired repair ceiling", "fresh_paired_len24_repair_executor_accuracy"),
        ]:
            ys = [as_float(row.get(key)) for row in train_rows]
            if any(not math.isnan(y) for y in ys):
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("evaluation point")
        ax.set_ylabel("executor accuracy")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "training_curve.png", dpi=160)
        plt.close(fig)

        target_rows = [row for row in train_rows if row.get("phase") == "train"]
        if target_rows:
            xs = list(range(1, len(target_rows) + 1))
            fig, ax = plt.subplots(figsize=(9.5, 4.6))
            for label, key in [
                ("verified repair targets", "target_repair_fraction"),
                ("changed targets", "target_changed_fraction"),
                ("active target rows", "target_active_fraction"),
            ]:
                ax.plot(xs, [as_float(row.get(key)) for row in target_rows], marker="o", linewidth=1.8, label=label)
            ax.set_ylim(0, 1.0)
            ax.set_xlabel("training evaluation point")
            ax.set_ylabel("fraction")
            ax.grid(alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(FIGURES / "target_quality.png", dpi=160)
            plt.close(fig)

    paired_rows = [row for row in all_rows if row.get("split") == "fresh_paired_len24" and "smoke" not in row.get("run", "")]
    if paired_rows:
        paired_rows = sorted(paired_rows, key=lambda row: row.get("run", ""))
        labels = [row.get("run", "") for row in paired_rows]
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 4.8))
        ax.plot(x, [as_float(row.get("executor_accuracy")) for row in paired_rows], marker="o", label="compiler")
        ax.plot(x, [as_float(row.get("repair_executor_accuracy")) for row in paired_rows], marker="o", label="local repair ceiling")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("fresh paired executor accuracy")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "iteration_summary.png", dpi=160)
        plt.close(fig)


def baseline_to_final_table(train_rows: Sequence[Dict[str, Any]]) -> str:
    if not train_rows:
        return "No training log is available."
    first = train_rows[0]
    last = train_rows[-1]
    rows = [["Split", "Baseline compiler", "Final compiler", "Baseline repair ceiling", "Final repair ceiling"]]
    for split in ["val_len24", "fresh_standard_len24", "fresh_paraphrase_len24", "fresh_paired_len24"]:
        rows.append(
            [
                split,
                pct(first.get(f"{split}_executor_accuracy")),
                pct(last.get(f"{split}_executor_accuracy")),
                pct(first.get(f"{split}_repair_executor_accuracy")),
                pct(last.get(f"{split}_repair_executor_accuracy")),
            ]
        )
    return markdown_table(rows)


def target_table(train_rows: Sequence[Dict[str, Any]]) -> str:
    target_rows = [row for row in train_rows if row.get("phase") == "train"]
    if not target_rows:
        return "No on-policy target rows are available."
    rows = [["Round", "Epoch", "Verified repair targets", "Changed targets", "Active rows", "Avg candidates", "Avg verified"]]
    for row in target_rows:
        rows.append(
            [
                row.get("round", ""),
                row.get("epoch", ""),
                pct(row.get("target_repair_fraction")),
                pct(row.get("target_changed_fraction")),
                pct(row.get("target_active_fraction")),
                scalar(row.get("target_avg_candidates")),
                scalar(row.get("target_avg_verified")),
            ]
        )
    return markdown_table(rows)


def final_split_table(rows: Sequence[Dict[str, Any]]) -> str:
    table = [["Split", "Compiler", "Local repair ceiling", "Program exact", "State prefix", "Repair found"]]
    for row in rows:
        table.append(
            [
                row.get("split", ""),
                pct(row.get("executor_accuracy")),
                pct(row.get("repair_executor_accuracy")),
                pct(row.get("program_exact")),
                pct(row.get("state_prefix_fraction")),
                pct(row.get("repair_found_fraction")),
            ]
        )
    return markdown_table(table)


def report_markdown(
    rows: Sequence[Dict[str, Any]],
    train_rows: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    primary_run: str,
    all_rows: Sequence[Dict[str, Any]],
) -> str:
    args = metadata.get("args", {})
    compiler_args = metadata.get("compiler_args", {})
    paired = split_row(rows, "fresh_paired_len24")
    gold_control = run_split_row(all_rows, "control_gold_only_s256", "fresh_paired_len24")
    repair_control = run_split_row(all_rows, "control_repair_only_s256", "fresh_paired_len24")
    first = train_rows[0] if train_rows else {}
    last = train_rows[-1] if train_rows else {}
    base = as_float(first.get("fresh_paired_len24_executor_accuracy"))
    final = as_float(last.get("fresh_paired_len24_executor_accuracy") or paired.get("executor_accuracy"))
    ceiling = as_float(paired.get("repair_executor_accuracy"))
    gap = (final - base) / (ceiling - base) if not any(math.isnan(x) for x in [base, final, ceiling]) and abs(ceiling - base) > 1e-9 else math.nan

    lines = [
        "# Qwen On-Policy Repair-to-Compiler Training",
        "",
        "## Abstract",
        "",
        "This experiment tests whether verified local program repairs can be converted into a better Qwen-attached compiler policy. "
        "A QLoRA compiler emits an executable modular-arithmetic program from each prompt. The training loop runs the current compiler on its own prompts, enumerates nearby program edits, keeps targets that pass an exact execution verifier, and fine-tunes the same compiler toward those repaired targets.",
        "",
        "## Setup",
        "",
        f"- Primary run: `{primary_run or 'n/a'}`",
        f"- Qwen substrate: `{compiler_args.get('model_id', 'Qwen/Qwen3-4B')}`",
        f"- Modulus: `{compiler_args.get('modulus', 97)}`",
        f"- Max program length: `{compiler_args.get('max_steps', 24)}`",
        f"- Train examples: `{args.get('train_examples', 'n/a')}`",
        f"- On-policy rounds: `{args.get('onpolicy_rounds', 'n/a')}`",
        f"- Epochs per round: `{args.get('epochs_per_round', 'n/a')}`",
        f"- Target mode: `{args.get('target_mode', 'n/a')}`",
        f"- Repair budget: top-k `{args.get('repair_topk', 'n/a')}`, max edits `{args.get('repair_max_edits', 'n/a')}`",
        "",
        "The local repair column is a ceiling measured with target-aware verification during analysis and target construction. It is not a deployable inference path. The deployable model is the compiler row after fine-tuning.",
        "",
        "## Results",
        "",
        "### Final Splits",
        "",
        final_split_table(rows) if rows else "No metrics are available yet.",
        "",
        "![Executor accuracy](../analysis/figures/executor_accuracy.png)",
        "",
        "### Baseline To Final",
        "",
        baseline_to_final_table(train_rows),
        "",
        "![Training curve](../analysis/figures/training_curve.png)",
        "",
        "### On-Policy Target Quality",
        "",
        target_table(train_rows),
        "",
        "![Target quality](../analysis/figures/target_quality.png)",
        "",
        "### Fresh Paired Details",
        "",
    ]
    detail_rows = [["Metric", "Compiler", "Local repair ceiling"]]
    for label, base_key, repair_key in [
        ("Executor accuracy", "executor_accuracy", "repair_executor_accuracy"),
        ("Program exact", "program_exact", "repair_program_exact"),
        ("State prefix fraction", "state_prefix_fraction", "repair_state_prefix_fraction"),
        ("Pair both-correct", "executor_pair_both_correct", "repair_pair_both_correct"),
        ("Pair state consistency", "compiler_pair_state_consistency", "repair_pair_state_consistency"),
    ]:
        detail_rows.append([label, pct(paired.get(base_key)), pct(paired.get(repair_key))])
    lines.extend(
        [
            markdown_table(detail_rows),
            "",
            "![Paired details](../analysis/figures/paired_details.png)",
            "",
            "### Run Summary",
            "",
        ]
    )
    iter_rows = [["Run", "Fresh paired compiler", "Fresh paired repair ceiling", "Repair found", "Program exact"]]
    for row in sorted([item for item in all_rows if item.get("split") == "fresh_paired_len24" and "smoke" not in item.get("run", "")], key=lambda item: item.get("run", "")):
        iter_rows.append(
            [
                row.get("run", ""),
                pct(row.get("executor_accuracy")),
                pct(row.get("repair_executor_accuracy")),
                pct(row.get("repair_found_fraction")),
                pct(row.get("program_exact")),
            ]
        )
    lines.extend([markdown_table(iter_rows), "", "![Run summary](../analysis/figures/iteration_summary.png)", ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            f"On the fresh paired split, the compiler moves from {pct(base)} at the initial evaluation point to {pct(final)} after on-policy repair training. "
            f"The measured local repair ceiling at the end is {pct(ceiling)}, so the compiler recovers {pct(gap)} of the initial compiler-to-repair gap.",
            "",
            f"Attribution is sharper with controls. The gold-only control reaches {pct(gold_control.get('executor_accuracy'))} fresh paired accuracy, matching the mixed repair-or-gold run. "
            f"The repair-only control, with gold auxiliary losses disabled and unverified rows skipped, still reaches {pct(repair_control.get('executor_accuracy'))}. "
            "So the headline gain is a real compiler-policy improvement, but it is not uniquely caused by repaired targets; dense gold trace supervision is sufficient under this budget, while verified local repairs alone provide a strong but weaker training signal.",
            "",
            "The key result is therefore narrower and more useful: a small amount of trace-level posttraining can turn a weak executable-program compiler into a near-ceiling compiler on fresh prompts, and target-aware local repairs provide a deployable training signal even when gold fallback is removed.",
            "",
            "## Limitations",
            "",
            "- The task is synthetic modular arithmetic.",
            "- Target construction uses exact execution verification.",
            "- The compiler head and deterministic runtime are specialized to copied numeric programs.",
            "- The local repair ceiling is target-aware and should be read only as headroom.",
            "- The primary result is one run unless more runs are added to the directory.",
            "",
            "## Artifacts",
            "",
            "Small experiment files live in:",
            "",
            "```text",
            "experiments/qwen_onpolicy_repair_compiler/",
            "```",
            "",
            "Large artifacts live in:",
            "",
            "```text",
            "large_artifacts/qwen_onpolicy_repair_compiler/checkpoints/",
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
            "- `analysis/figures/target_quality.png`",
            "- `analysis/figures/iteration_summary.png`",
            f"- `runs/{primary_run}/metrics.csv`",
            f"- `runs/{primary_run}/train_log.csv`",
            "- `reports/qwen_onpolicy_repair_compiler_paper.md`",
            "- `reports/qwen_onpolicy_repair_compiler_paper.html`",
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
body { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #18231d; background: #f7f8f6; }
main { max-width: 1000px; margin: 0 auto; padding: 48px 24px 72px; background: #fff; }
h1 { font-size: 38px; margin: 0 0 24px; letter-spacing: 0; }
h2 { margin-top: 34px; border-top: 1px solid #d8dfd9; padding-top: 22px; }
h3 { margin-top: 26px; }
p, li { line-height: 1.55; font-size: 16px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0 24px; font-size: 14px; }
th, td { border-bottom: 1px solid #d8dfd9; padding: 8px 10px; text-align: left; }
th { background: #f0f3ef; font-weight: 650; }
pre { background: #f0f3ef; padding: 14px; overflow-x: auto; border-radius: 6px; }
img { max-width: 100%; border: 1px solid #d8dfd9; border-radius: 6px; background: #fff; }
figcaption { font-size: 13px; color: #526052; margin-top: 6px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
"""
    return f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{css}</style></head><body><main>{"".join(out)}</main></body></html>\n'


def write_summary(rows: Sequence[Dict[str, Any]], train_rows: Sequence[Dict[str, Any]], primary_run: str) -> None:
    primary = primary_rows(rows, primary_run)
    lines = ["# On-Policy Repair-to-Compiler Analysis Summary", "", f"Primary run: `{primary_run or 'n/a'}`", "", "## Final Metrics", ""]
    lines.append(final_split_table(primary) if primary else "No metrics are available yet.")
    lines.extend(["", "## Baseline To Final", "", baseline_to_final_table(train_rows), ""])
    control_rows = [
        run_split_row(rows, "main_onpolicy_repair_s256", "fresh_paired_len24"),
        run_split_row(rows, "control_gold_only_s256", "fresh_paired_len24"),
        run_split_row(rows, "control_repair_only_s256", "fresh_paired_len24"),
    ]
    if any(control_rows):
        table = [["Run", "Fresh paired compiler", "Fresh paired repair ceiling", "Program exact"]]
        for row in control_rows:
            if row:
                table.append([row.get("run", ""), pct(row.get("executor_accuracy")), pct(row.get("repair_executor_accuracy")), pct(row.get("program_exact"))])
        lines.extend(["## Controls", "", markdown_table(table), ""])
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = load_metric_rows()
    primary_run = choose_primary(rows)
    primary = primary_rows(rows, primary_run)
    train_rows = load_train_rows(primary_run)
    metadata = load_metadata(primary_run)
    write_csv(ANALYSIS / "all_final_metrics.csv", rows)
    write_csv(ANALYSIS / "final_metrics.csv", primary)
    write_figures(primary, train_rows, rows)
    write_summary(rows, train_rows, primary_run)
    report = report_markdown(primary, train_rows, metadata, primary_run, rows)
    (REPORTS / "qwen_onpolicy_repair_compiler_paper.md").write_text(report)
    (REPORTS / "qwen_onpolicy_repair_compiler_paper.html").write_text(markdown_to_html(report, "Qwen On-Policy Repair-to-Compiler Training"))
    write_checkpoint_manifest()
    print(ANALYSIS / "summary.md")


if __name__ == "__main__":
    main()

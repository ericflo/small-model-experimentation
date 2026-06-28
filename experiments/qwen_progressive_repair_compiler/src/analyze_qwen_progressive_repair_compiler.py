#!/usr/bin/env python3
"""Aggregate progressive repair compiler runs and write reports."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path("experiments/qwen_progressive_repair_compiler")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_progressive_repair_compiler/checkpoints")


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


def choose_primary(rows: Sequence[Dict[str, Any]]) -> str:
    preferred = "main_progressive_repair_s512"
    if any(row.get("run") == preferred for row in rows):
        return preferred
    runs = sorted({row.get("run", "") for row in rows if row.get("run")})
    non_smoke = [run for run in runs if "smoke" not in run]
    candidates = non_smoke or runs
    if not candidates:
        return ""
    return max(candidates, key=lambda run: (RUNS / run / "metrics.csv").stat().st_mtime if (RUNS / run / "metrics.csv").exists() else 0.0)


def load_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/metrics.csv")):
        rows.extend(read_csv(path))
    return rows


def load_train_rows(primary_run: str) -> List[Dict[str, Any]]:
    path = RUNS / primary_run / "verifier_train_log.csv"
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


def primary_rows(rows: Sequence[Dict[str, Any]], primary_run: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("run") == primary_run]


def fresh_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("split", "").startswith("fresh_")]


def paired_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return next((row for row in rows if row.get("split") == "fresh_paired_len24"), {})


def write_figures(rows: Sequence[Dict[str, Any]], train_rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[figures] matplotlib unavailable: {exc}")
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    fres = fresh_rows(rows)
    if fres:
        labels = [row["split"].replace("fresh_", "").replace("_len24", "") for row in fres]
        series = [
            ("base", "base_executor_accuracy"),
            ("learned", "learned_executor_accuracy"),
            ("pair rerank", "pair_rerank_executor_accuracy"),
            ("oracle", "oracle_executor_accuracy"),
        ]
        x = list(range(len(labels)))
        width = 0.18
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        for offset, (label, key) in zip([-1.5, -0.5, 0.5, 1.5], series):
            ax.bar([i + offset * width for i in x], [as_float(row.get(key)) for row in fres], width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("executor accuracy")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", ncols=4, fontsize=8)
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
        series = [
            ("base", "base"),
            ("learned", "learned"),
            ("pair rerank", "pair_rerank"),
            ("oracle", "oracle"),
        ]
        x = list(range(len(labels)))
        width = 0.18
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for offset, (label, prefix) in zip([-1.5, -0.5, 0.5, 1.5], series):
            ys = [as_float(paired.get(f"{prefix}_{key}")) for _, key in metrics]
            ax.bar([i + offset * width for i in x], ys, width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("fraction")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", ncols=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "paired_details.png", dpi=160)
        plt.close(fig)

    if train_rows:
        xs = [int(float(row.get("epoch", 0))) for row in train_rows]
        learned = [as_float(row.get("val_learned_executor_accuracy")) for row in train_rows]
        oracle = [as_float(row.get("val_oracle_executor_accuracy")) for row in train_rows]
        base = [as_float(row.get("val_base_executor_accuracy")) for row in train_rows]
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.plot(xs, base, marker="o", label="base")
        ax.plot(xs, learned, marker="o", label="learned")
        ax.plot(xs, oracle, marker="o", label="oracle")
        last_stage = None
        for row in train_rows:
            stage = row.get("stage", "")
            epoch = int(float(row.get("epoch", 0)))
            if stage != last_stage:
                ax.axvline(epoch, color="black", alpha=0.12, linewidth=1)
                ax.text(epoch, 0.03, stage, rotation=90, va="bottom", fontsize=8, alpha=0.7)
                last_stage = stage
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("verifier epoch")
        ax.set_ylabel("validation executor accuracy")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(FIGURES / "training_curve.png", dpi=160)
        plt.close(fig)


def markdown_table(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "|" + "|".join(["---"] * len(rows[0])) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def report_markdown(rows: Sequence[Dict[str, Any]], train_rows: Sequence[Dict[str, Any]], metadata: Dict[str, Any], primary_run: str) -> str:
    paired = paired_row(rows)
    fresh = fresh_rows(rows)
    best_epoch = ""
    if train_rows:
        best = max(train_rows, key=lambda row: as_float(row.get("val_learned_executor_accuracy")))
        best_epoch = str(best.get("epoch", ""))
    base = as_float(paired.get("base_executor_accuracy"))
    learned = as_float(paired.get("learned_executor_accuracy"))
    pair = as_float(paired.get("pair_rerank_executor_accuracy"))
    oracle = as_float(paired.get("oracle_executor_accuracy"))
    gap = (learned - base) / (oracle - base) if not any(math.isnan(x) for x in [base, learned, oracle]) and abs(oracle - base) > 1e-9 else math.nan
    lines = [
        "# Qwen Progressive Repair Compiler",
        "",
        "## Abstract",
        "",
        "This experiment tests a deployable repair-selection layer for a frozen Qwen-attached numeric compiler. "
        "The compiler converts a prompt into an executable modular-arithmetic program. Around that program, the "
        "system enumerates local edits and trains a small transformer verifier to choose among candidate execution "
        "traces without seeing the true answer or true state trajectory at test time.",
        "",
        "The new variable is a progressive candidate-space curriculum. The verifier starts on small one-edit "
        "neighborhoods and then trains on the full two-edit repair space. Final evaluation uses the full candidate "
        "space for all methods.",
        "",
        "## Setup",
        "",
        f"- Primary run: `{primary_run}`",
        f"- Qwen substrate: `{metadata.get('compiler_args', {}).get('model_id', 'Qwen/Qwen3-4B')}`",
        f"- Modulus: `{metadata.get('compiler_args', {}).get('modulus', 97)}`",
        f"- Max program length: `{metadata.get('compiler_args', {}).get('max_steps', 24)}`",
        f"- Best verifier epoch: `{best_epoch or 'n/a'}`",
        "",
        "Candidate labels are computed offline from exact trajectories. At evaluation time, the learned verifier "
        "receives only candidate-local information: copied slots, edit metadata, compiler probabilities, predicted "
        "states, and soft-executor support.",
        "",
        "## Results",
        "",
        "### Fresh Splits",
        "",
    ]
    table_rows = [["Split", "Base", "Learned", "Pair rerank", "Oracle", "Gap recovered"]]
    for row in fresh:
        table_rows.append(
            [
                row.get("split", ""),
                pct(row.get("base_executor_accuracy")),
                pct(row.get("learned_executor_accuracy")),
                pct(row.get("pair_rerank_executor_accuracy")),
                pct(row.get("oracle_executor_accuracy")),
                pct(row.get("learned_oracle_gap_recovered")),
            ]
        )
    lines.append(markdown_table(table_rows))
    lines.extend(
        [
            "",
            "![Executor accuracy](../analysis/figures/executor_accuracy.png)",
            "",
            "### Fresh Paired Details",
            "",
        ]
    )
    detail_rows = [["Metric", "Base", "Learned", "Pair rerank", "Oracle"]]
    for label, key in [
        ("Executor accuracy", "executor_accuracy"),
        ("Program exact", "program_exact"),
        ("State prefix fraction", "state_prefix_fraction"),
        ("Pair both-correct", "pair_both_correct"),
        ("Pair state consistency", "pair_state_consistency"),
    ]:
        detail_rows.append(
            [
                label,
                pct(paired.get(f"base_{key}")),
                pct(paired.get(f"learned_{key}")),
                pct(paired.get(f"pair_rerank_{key}")),
                pct(paired.get(f"oracle_{key}")),
            ]
        )
    lines.append(markdown_table(detail_rows))
    lines.extend(
        [
            "",
            "![Paired details](../analysis/figures/paired_details.png)",
            "",
            "### Training Dynamics",
            "",
            "The training curve tracks validation accuracy on the full candidate set after every verifier epoch. "
            "Stage labels mark the candidate budget used for that part of training.",
            "",
            "![Training curve](../analysis/figures/training_curve.png)",
            "",
            "## Interpretation",
            "",
            f"On the fresh paired split, the learned verifier moves exact execution from {pct(base)} to {pct(learned)}. "
            f"The oracle ceiling is {pct(oracle)}, so the learned selector recovers {pct(gap)} of the measured "
            "base-to-oracle gap. Pair reranking is a separate consistency control for paired prompt renderings; it "
            f"reaches {pct(pair)} executor accuracy on the paired split.",
            "",
            "The key bottleneck is still selection quality. The candidate space often contains a correct executable "
            "program, but the learned verifier does not always identify it. The experiment therefore supports "
            "continuing toward verifier-to-compiler distillation or joint training, because the current external "
            "selector converts only part of the available local-repair headroom.",
            "",
            "The staged curriculum is not, by itself, the breakthrough mechanism in this run. Validation accuracy "
            "stayed near the base compiler through the small and medium stages, then improved only after the verifier "
            "trained on the full candidate neighborhood. That points to a practical next step: use the full repair "
            "space from the start or distill full-space successful choices into the compiler rather than relying on a long "
            "warm-up over easier neighborhoods.",
            "",
            "## Limitations",
            "",
            "- The task is synthetic modular arithmetic.",
            "- The compiler and runtime are specialized.",
            "- Candidate labels use exact trajectories during training.",
            "- The verifier consumes engineered trace features rather than only raw hidden states.",
            "- Pair reranking requires paired prompt renderings and is not a single-prompt method.",
            "",
            "## Artifacts",
            "",
            "Small experiment files live in:",
            "",
            "```text",
            "experiments/qwen_progressive_repair_compiler/",
            "```",
            "",
            "Large artifacts live in:",
            "",
            "```text",
            "large_artifacts/qwen_progressive_repair_compiler/checkpoints/",
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
            f"- `runs/{primary_run}/metrics.csv`",
            f"- `runs/{primary_run}/verifier_train_log.csv`",
            "- `reports/qwen_progressive_repair_compiler_paper.md`",
            "- `reports/qwen_progressive_repair_compiler_paper.html`",
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
                body = table_lines[2:] if set(table_lines[1].replace("|", "").replace("-", "").replace(":", "").strip()) == set() else table_lines[1:]
                out.append("<table><thead><tr>")
                out.extend(f"<th>{html.escape(cell)}</th>" for cell in header)
                out.append("</tr></thead><tbody>")
                for row in body:
                    cells = [cell.strip() for cell in row.strip("|").split("|")]
                    out.append("<tr>")
                    out.extend(f"<td>{html.escape(cell)}</td>" for cell in cells)
                    out.append("</tr>")
                out.append("</tbody></table>")
            else:
                out.extend(f"<p>{html.escape(row)}</p>" for row in table_lines)
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
    lines = [
        "# Progressive Repair Compiler Analysis Summary",
        "",
        f"Primary run: `{primary_run}`",
        "",
        "## Fresh Metrics",
        "",
    ]
    table_rows = [["Split", "Base", "Learned", "Pair rerank", "Oracle", "Gap recovered", "Candidates"]]
    for row in primary:
        table_rows.append(
            [
                row.get("split", ""),
                pct(row.get("base_executor_accuracy")),
                pct(row.get("learned_executor_accuracy")),
                pct(row.get("pair_rerank_executor_accuracy")),
                pct(row.get("oracle_executor_accuracy")),
                pct(row.get("learned_oracle_gap_recovered")),
                scalar(row.get("avg_candidates")),
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
    write_figures(primary, train_rows)
    write_summary(rows, primary_run)
    report = report_markdown(primary, train_rows, metadata, primary_run)
    (REPORTS / "qwen_progressive_repair_compiler_paper.md").write_text(report)
    (REPORTS / "qwen_progressive_repair_compiler_paper.html").write_text(
        markdown_to_html(report, "Qwen Progressive Repair Compiler")
    )
    write_checkpoint_manifest()
    print(ANALYSIS / "summary.md")


if __name__ == "__main__":
    main()

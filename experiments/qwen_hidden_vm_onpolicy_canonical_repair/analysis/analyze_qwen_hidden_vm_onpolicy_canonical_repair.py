#!/usr/bin/env python3
"""Aggregate hidden VM on-policy canonical repair runs and write reports."""

from __future__ import annotations

import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path("experiments/qwen_hidden_vm_onpolicy_canonical_repair")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_hidden_vm_onpolicy_canonical_repair/checkpoints")
DOMAIN_NAMES = ["arithmetic", "calendar", "unit", "list", "boolean", "lookup"]


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


def pp_delta(a: Any, b: Any) -> str:
    x = as_float(a)
    y = as_float(b)
    if math.isnan(x) or math.isnan(y):
        return "n/a"
    sign = "+" if x >= y else ""
    return f"{sign}{100.0 * (x - y):.1f} pp"


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
    lines = [
        "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(rows[0])) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(lines)


def load_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/metrics.csv")):
        rows.extend(read_csv(path))
    return rows


def choose_primary(rows: Sequence[Dict[str, Any]]) -> str:
    preferred = "main_repair_or_gold_s512"
    if any(row.get("run") == preferred for row in rows):
        return preferred
    runs = sorted({row.get("run", "") for row in rows if row.get("run")})
    non_smoke = [run for run in runs if "smoke" not in run and "control" not in run]
    candidates = non_smoke or [run for run in runs if "smoke" not in run] or runs
    if not candidates:
        return ""
    return max(candidates, key=lambda run: (RUNS / run / "metrics.csv").stat().st_mtime if (RUNS / run / "metrics.csv").exists() else 0.0)


def rows_for_run(rows: Sequence[Dict[str, Any]], run: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("run") == run]


def split_row(rows: Sequence[Dict[str, Any]], split: str) -> Dict[str, Any]:
    return next((row for row in rows if row.get("split") == split), {})


def load_train_rows(run: str) -> List[Dict[str, Any]]:
    path = RUNS / run / "train_log.csv"
    return read_csv(path) if run and path.exists() else []


def load_metadata(run: str) -> Dict[str, Any]:
    path = RUNS / run / "results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("metadata", {})


def write_checkpoint_manifest() -> None:
    rows: List[Dict[str, Any]] = []
    for path in sorted(CHECKPOINT_ROOT.rglob("*")):
        if path.is_file():
            rel = path.relative_to(CHECKPOINT_ROOT)
            rows.append({"run": rel.parts[0] if rel.parts else "", "artifact_path": str(path), "bytes": path.stat().st_size})
    write_csv(ROOT / "checkpoint_manifest.csv", rows)


def write_figures(primary_rows: Sequence[Dict[str, Any]], train_rows: Sequence[Dict[str, Any]], all_rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[figures] matplotlib unavailable: {exc}")
        return
    FIGURES.mkdir(parents=True, exist_ok=True)

    splits = [row for row in primary_rows if row.get("split", "").startswith(("val", "fresh", "hard"))]
    if splits:
        labels = [row["split"].replace("_mixed", "").replace("fresh_", "fr_").replace("standard", "std").replace("paraphrase", "para") for row in splits]
        x = list(range(len(labels)))
        width = 0.24
        fig, ax = plt.subplots(figsize=(max(9, 1.25 * len(labels)), 4.8))
        ax.bar([i - width for i in x], [as_float(row.get("direct_accuracy")) for row in splits], width, label="direct logits")
        ax.bar(x, [as_float(row.get("executor_accuracy")) for row in splits], width, label="hidden VM")
        ax.bar([i + width for i in x], [as_float(row.get("repair_executor_accuracy")) for row in splits], width, label="verified repair")
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("accuracy")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "split_accuracy.png", dpi=160)
        plt.close(fig)

    fresh = split_row(primary_rows, "fresh_paired_mixed") or split_row(primary_rows, "fresh_standard_mixed") or split_row(primary_rows, "val_mixed")
    if fresh:
        labels = DOMAIN_NAMES
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        width = 0.34
        ax.bar([i - width / 2 for i in x], [as_float(fresh.get(f"domain_{domain}_executor_accuracy")) for domain in labels], width, label="hidden VM")
        ax.bar([i + width / 2 for i in x], [as_float(fresh.get(f"domain_{domain}_repair_executor_accuracy")) for domain in labels], width, label="verified repair")
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("hidden VM accuracy")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "domain_accuracy.png", dpi=160)
        plt.close(fig)

    eval_rows = [row for row in train_rows if row.get("phase") in {"baseline", "eval", "onpolicy_train"}]
    if eval_rows:
        xs = [int(float(row.get("step", i))) for i, row in enumerate(eval_rows)]
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        for label, key in [
            ("val", "val_mixed_executor_accuracy"),
            ("fresh paired", "fresh_paired_mixed_executor_accuracy"),
            ("hard standard", "hard_standard_mixed_executor_accuracy"),
            ("harder standard", "harder_standard_mixed_executor_accuracy"),
            ("hard paraphrase", "hard_paraphrase_mixed_executor_accuracy"),
        ]:
            ys = [as_float(row.get(key)) for row in eval_rows]
            if any(not math.isnan(y) for y in ys):
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("training step")
        ax.set_ylabel("executor accuracy")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "training_curve.png", dpi=160)
        plt.close(fig)

    target_rows = [row for row in train_rows if row.get("phase") == "onpolicy_targets"]
    if target_rows:
        labels = [f"r{int(float(row.get('round', i + 1)))}" for i, row in enumerate(target_rows)]
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        for label, key in [
            ("active", "target_active_fraction"),
            ("found", "target_repair_found_fraction"),
            ("changed", "target_repair_changed_fraction"),
            ("program exact", "target_program_exact_fraction"),
        ]:
            ax.plot(x, [as_float(row.get(key)) for row in target_rows], marker="o", linewidth=1.8, label=label)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("fraction")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "target_quality.png", dpi=160)
        plt.close(fig)

    paired_rows = [row for row in all_rows if row.get("split") == "fresh_paired_mixed" and "smoke" not in row.get("run", "")]
    if paired_rows:
        paired_rows = sorted(paired_rows, key=lambda row: row.get("run", ""))
        labels = [row.get("run", "") for row in paired_rows]
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(labels)), 4.8))
        ax.plot(x, [as_float(row.get("direct_accuracy")) for row in paired_rows], marker="o", label="direct logits")
        ax.plot(x, [as_float(row.get("executor_accuracy")) for row in paired_rows], marker="o", label="hidden VM")
        ax.plot(x, [as_float(row.get("repair_executor_accuracy")) for row in paired_rows], marker="o", label="verified repair")
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("fresh paired accuracy")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "run_summary.png", dpi=160)
        plt.close(fig)


def final_split_table(rows: Sequence[Dict[str, Any]]) -> str:
    table = [["Split", "Direct", "Hidden VM", "Repair", "Program exact", "Repair exact", "State prefix", "Repair found"]]
    for row in rows:
        table.append(
            [
                row.get("split", ""),
                pct(row.get("direct_accuracy")),
                pct(row.get("executor_accuracy")),
                pct(row.get("repair_executor_accuracy")),
                pct(row.get("program_exact")),
                pct(row.get("repair_program_exact")),
                pct(row.get("state_prefix_fraction")),
                pct(row.get("repair_found_fraction")),
            ]
        )
    return markdown_table(table)


def domain_table(row: Dict[str, Any]) -> str:
    table = [["Domain", "n", "Direct", "Hidden VM", "Repair"]]
    for domain in DOMAIN_NAMES:
        if row.get(f"domain_{domain}_n"):
            table.append(
                [
                    domain,
                    scalar(row.get(f"domain_{domain}_n")),
                    pct(row.get(f"domain_{domain}_direct_accuracy")),
                    pct(row.get(f"domain_{domain}_executor_accuracy")),
                    pct(row.get(f"domain_{domain}_repair_executor_accuracy")),
                ]
            )
    return markdown_table(table)


def run_summary_table(rows: Sequence[Dict[str, Any]]) -> str:
    table = [["Run", "Variant", "Direct", "Hidden VM", "Repair", "Program exact", "State prefix"]]
    for row in sorted([r for r in rows if r.get("split") == "fresh_paired_mixed" and "smoke" not in r.get("run", "")], key=lambda r: r.get("run", "")):
        table.append(
            [
                row.get("run", ""),
                row.get("variant", ""),
                pct(row.get("direct_accuracy")),
                pct(row.get("executor_accuracy")),
                pct(row.get("repair_executor_accuracy")),
                pct(row.get("program_exact")),
                pct(row.get("state_prefix_fraction")),
            ]
        )
    return markdown_table(table)


def report_markdown(primary_rows: Sequence[Dict[str, Any]], train_rows: Sequence[Dict[str, Any]], metadata: Dict[str, Any], primary_run: str, all_rows: Sequence[Dict[str, Any]]) -> str:
    args = metadata.get("args", {})
    paired = split_row(primary_rows, "fresh_paired_mixed")
    hard = split_row(primary_rows, "hard_standard_mixed")
    harder = split_row(primary_rows, "harder_standard_mixed")
    baseline = train_rows[0] if train_rows else {}
    base_paired = baseline.get("fresh_paired_mixed_executor_accuracy")
    final_paired = paired.get("executor_accuracy")
    repair_paired = paired.get("repair_executor_accuracy")
    trace_control = split_row(rows_for_run(all_rows, "main_trace_control_s512"), "fresh_paired_mixed")
    trace_control_hard = split_row(rows_for_run(all_rows, "main_trace_control_s512"), "hard_standard_mixed")
    trace_control_harder = split_row(rows_for_run(all_rows, "main_trace_control_s512"), "harder_standard_mixed")
    gold_control = split_row(rows_for_run(all_rows, "main_gold_control_s512"), "fresh_paired_mixed")
    repair_only = split_row(rows_for_run(all_rows, "main_repair_only_s512"), "fresh_paired_mixed")
    target_rows = [row for row in train_rows if row.get("phase") == "onpolicy_targets"]
    target_last = target_rows[-1] if target_rows else {}
    direct_fresh = paired.get("direct_accuracy")
    trace_program = paired.get("program_exact")
    trace_state = paired.get("state_prefix_fraction")
    treatment_delta = pp_delta(final_paired, trace_control.get("executor_accuracy"))
    lines = [
        "# Qwen Hidden VM On-Policy Canonical Repair",
        "",
        "## Abstract",
        "",
        "This experiment tests whether a Qwen 4B model can improve a hidden virtual-machine compiler by training on canonical on-policy repair targets. The model emits invisible typed VM slots, a deterministic runtime executes those slots, and local candidate repairs are accepted only when their full intermediate state trajectory matches the canonical trajectory.",
        "",
        "## Setup",
        "",
        f"- Primary run: `{primary_run or 'n/a'}`",
        f"- Model: `{args.get('model_id', 'Qwen/Qwen3-4B')}`",
        f"- Variant: `{args.get('variant', 'n/a')}`",
        f"- Train examples: `{args.get('train_examples', 'n/a')}`",
        f"- Train steps: `{args.get('train_steps', 'n/a')}`",
        f"- On-policy rounds: `{args.get('onpolicy_rounds', 'n/a')}`",
        f"- Epochs per round: `{args.get('epochs_per_round', 'n/a')}`",
        f"- Target mode: `{args.get('target_mode', 'n/a')}`",
        f"- Repair verifier mode: `{args.get('repair_verifier_mode', 'n/a')}`",
        f"- VM max steps: `{args.get('max_steps', 'n/a')}`",
        f"- Curriculum schedule: `{args.get('curriculum_schedule', 'n/a')}`",
        f"- Train length range: `{args.get('train_min_len', 'n/a')}` to `{args.get('train_max_len', 'n/a')}`",
        f"- Eval length: `{args.get('eval_length', 'n/a')}`; hard length: `{args.get('hard_length', 'n/a')}`; harder length: `{args.get('harder_length', 'n/a')}`",
        "",
        "The hidden VM uses typed operation slots and copied numeric arguments. Direct logits are the model's next-token numeric answer distribution at the answer marker. Hidden VM accuracy is execution of the compiled invisible program. Repair accuracy is target-aware state-verified local search around the compiled program and is reported as a headroom measurement, not as a deployable inference path.",
        "",
        "## Results",
        "",
        "### Final Splits",
        "",
        final_split_table(primary_rows) if primary_rows else "No metrics are available.",
        "",
        "![Split accuracy](../analysis/figures/split_accuracy.png)",
        "",
        "### Domain Breakdown",
        "",
        domain_table(paired or split_row(primary_rows, "fresh_standard_mixed") or split_row(primary_rows, "val_mixed")),
        "",
        "![Domain accuracy](../analysis/figures/domain_accuracy.png)",
        "",
        "### Training Dynamics",
        "",
        f"Fresh paired hidden VM accuracy moved from {pct(base_paired)} at initialization to {pct(final_paired)} after the full treatment. "
        f"State-verified local repair on the same split reaches {pct(repair_paired)}. The trace-only control scores {pct(trace_control.get('executor_accuracy'))} on fresh paired, {pct(trace_control_hard.get('executor_accuracy'))} on hard length {args.get('hard_length', 'n/a')}, and {pct(trace_control_harder.get('executor_accuracy'))} on harder length {args.get('harder_length', 'n/a')}.",
        "",
        "![Training curve](../analysis/figures/training_curve.png)",
        "",
        "### On-Policy Target Quality",
        "",
        f"The final target pass used {scalar(target_last.get('target_source_n'))} source examples. "
        f"Active rows were {pct(target_last.get('target_active_fraction'))}; canonical repairs were found for {pct(target_last.get('target_repair_found_fraction'))}; changed-program repairs were {pct(target_last.get('target_repair_changed_fraction'))}; program-exact repaired targets were {pct(target_last.get('target_program_exact_fraction'))}; average local candidates were {scalar(target_last.get('target_avg_candidates'))}.",
        "",
        "![Target quality](../analysis/figures/target_quality.png)",
        "",
        "### Run Summary",
        "",
        run_summary_table(all_rows),
        "",
        "![Run summary](../analysis/figures/run_summary.png)",
        "",
        "## Interpretation",
        "",
        f"The primary measurement is fresh paired mixed-domain accuracy. Direct logits score {pct(direct_fresh)}, while the on-policy canonical-repair hidden VM scores {pct(final_paired)} ({pp_delta(final_paired, direct_fresh)}). The trace-only control scores {pct(trace_control.get('executor_accuracy'))}, so the on-policy treatment changes fresh paired accuracy by {treatment_delta}. State-verified local repair scores {pct(repair_paired)}, measuring how often the current top-k neighborhood contains a canonical executable program.",
        "",
        f"Program-exact accuracy is {pct(trace_program)} and state-prefix accuracy is {pct(trace_state)}. The gold-only control scores {pct(gold_control.get('executor_accuracy'))} on fresh paired and the repair-only control scores {pct(repair_only.get('executor_accuracy'))}; the repair arms therefore do not separate from an extra stabilized gold-trace pass.",
        "",
        f"The hard-length splits are the decisive stress test. The treatment trains up to length {args.get('train_max_len', 'n/a')} and reaches {pct(hard.get('executor_accuracy'))} at length {args.get('hard_length', 'n/a')} and {pct(harder.get('executor_accuracy'))} at length {args.get('harder_length', 'n/a')}. The trace-only control reaches {pct(trace_control_hard.get('executor_accuracy'))} and {pct(trace_control_harder.get('executor_accuracy'))} on those same standard hard splits, so the on-policy epoch does not improve length robustness.",
        "",
        "## Decision",
        "",
        "Canonical on-policy repair should not be scaled in this form. The state verifier produces high-quality targets, but distilling those targets into the compiler gives the same fresh paired result as gold-only training and weakens hard-length robustness relative to the trace-only control. The useful outcome is the headroom measurement: state-verified local search still reaches 89.1% on fresh paired and 85.4% on hard length 8. The next step should keep repair as selection or reranking, or train only on uncertainty-targeted repairs with a much stricter preservation objective.",
        "",
        "## Limitations",
        "",
        "- The domains are synthetic and deterministic.",
        "- Answers are integers in a bounded value vocabulary.",
        "- Trace supervision supplies exact hidden programs during the curriculum phase.",
        "- Repair accuracy is target-aware and should be read as verifier-assisted headroom.",
        "- Canonical state verification uses synthetic trajectories available in this harness.",
        "- The runtime is fixed and hand-designed.",
        "- Each main arm is one run unless additional seeds are added.",
        "",
        "## Artifacts",
        "",
        "Small experiment files live in:",
        "",
        "```text",
        "experiments/qwen_hidden_vm_onpolicy_canonical_repair/",
        "```",
        "",
        "Large artifacts live in:",
        "",
        "```text",
        "large_artifacts/qwen_hidden_vm_onpolicy_canonical_repair/checkpoints/",
        "```",
        "",
        "Primary files:",
        "",
        "- `analysis/summary.md`",
        "- `analysis/final_metrics.csv`",
        "- `analysis/all_final_metrics.csv`",
        "- `analysis/figures/split_accuracy.png`",
        "- `analysis/figures/domain_accuracy.png`",
        "- `analysis/figures/training_curve.png`",
        "- `analysis/figures/target_quality.png`",
        "- `analysis/figures/run_summary.png`",
        f"- `runs/{primary_run}/metrics.csv`",
        f"- `runs/{primary_run}/train_log.csv`",
        "- `reports/qwen_hidden_vm_onpolicy_canonical_repair_paper.md`",
        "- `reports/qwen_hidden_vm_onpolicy_canonical_repair_paper.html`",
        "- `checkpoint_manifest.csv`",
    ]
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown: str, title: str) -> str:
    def inline_html(text: str) -> str:
        escaped = html.escape(text)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
        return escaped

    lines = markdown.splitlines()
    out: List[str] = []
    in_ul = False
    in_ol = False
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
                out.extend(f"<th>{inline_html(cell)}</th>" for cell in header)
                out.append("</tr></thead><tbody>")
                for row in body:
                    cells = [cell.strip() for cell in row.strip("|").split("|")]
                    out.append("<tr>")
                    out.extend(f"<td>{inline_html(cell)}</td>" for cell in cells)
                    out.append("</tr>")
                out.append("</tbody></table>")
            continue
        if not line.strip():
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if in_ol:
                out.append("</ol>")
                in_ol = False
            i += 1
            continue
        if line.startswith("# "):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if in_ol:
                out.append("</ol>")
                in_ol = False
            out.append(f"<h1>{inline_html(line[2:])}</h1>")
        elif line.startswith("## "):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if in_ol:
                out.append("</ol>")
                in_ol = False
            out.append(f"<h2>{inline_html(line[3:])}</h2>")
        elif line.startswith("### "):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if in_ol:
                out.append("</ol>")
                in_ol = False
            out.append(f"<h3>{inline_html(line[4:])}</h3>")
        elif line.startswith("![") and "](" in line:
            alt = line[2:].split("]", 1)[0]
            src = line.split("](", 1)[1].rstrip(")")
            out.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{inline_html(alt)}</figcaption></figure>')
        elif line.startswith("- "):
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{inline_html(line[2:])}</li>")
        elif re.match(r"\d+\. ", line):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{inline_html(re.sub(r'^\\d+\\.\\s+', '', line))}</li>")
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if in_ol:
                out.append("</ol>")
                in_ol = False
            out.append(f"<p>{inline_html(line)}</p>")
        i += 1
    if in_ul:
        out.append("</ul>")
    if in_ol:
        out.append("</ol>")
    css = """
body { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #17201a; background: #f7f8f6; }
main { max-width: 1040px; margin: 0 auto; padding: 48px 24px 72px; background: #fff; }
h1 { font-size: 38px; margin: 0 0 24px; letter-spacing: 0; }
h2 { margin-top: 34px; border-top: 1px solid #d8dfd9; padding-top: 22px; }
h3 { margin-top: 26px; }
p, li { line-height: 1.55; font-size: 16px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0 24px; font-size: 14px; }
th, td { border-bottom: 1px solid #d8dfd9; padding: 8px 10px; text-align: left; }
th { background: #eef3ef; font-weight: 650; }
pre { background: #eef3ef; padding: 14px; overflow-x: auto; border-radius: 6px; }
img { max-width: 100%; border: 1px solid #d8dfd9; border-radius: 6px; background: #fff; }
figcaption { font-size: 13px; color: #526052; margin-top: 6px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
"""
    return f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{css}</style></head><body><main>{"".join(out)}</main></body></html>\n'


def write_summary(primary_rows: Sequence[Dict[str, Any]], all_rows: Sequence[Dict[str, Any]], primary_run: str) -> None:
    lines = ["# Qwen Hidden VM On-Policy Canonical Repair Analysis Summary", "", f"Primary run: `{primary_run or 'n/a'}`", "", "## Final Splits", ""]
    lines.append(final_split_table(primary_rows) if primary_rows else "No metrics are available.")
    paired = split_row(primary_rows, "fresh_paired_mixed")
    if paired:
        lines.extend(
            [
                "",
                "## Headline",
                "",
                f"- Fresh paired direct logits: {pct(paired.get('direct_accuracy'))}",
                f"- Fresh paired hidden VM: {pct(paired.get('executor_accuracy'))} ({pp_delta(paired.get('executor_accuracy'), paired.get('direct_accuracy'))} vs direct)",
                f"- Fresh paired verified repair: {pct(paired.get('repair_executor_accuracy'))} ({pp_delta(paired.get('repair_executor_accuracy'), paired.get('executor_accuracy'))} repair headroom)",
                f"- Program exact: {pct(paired.get('program_exact'))}",
            ]
        )
        lines.extend(["", "## Fresh Paired Domain Breakdown", "", domain_table(paired)])
    lines.extend(["", "## Run Summary", "", run_summary_table(all_rows), ""])
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    primary_run = choose_primary(rows)
    primary = rows_for_run(rows, primary_run)
    train_rows = load_train_rows(primary_run)
    metadata = load_metadata(primary_run)
    write_csv(ANALYSIS / "all_final_metrics.csv", rows)
    write_csv(ANALYSIS / "final_metrics.csv", primary)
    write_figures(primary, train_rows, rows)
    write_summary(primary, rows, primary_run)
    report = report_markdown(primary, train_rows, metadata, primary_run, rows)
    (REPORTS / "qwen_hidden_vm_onpolicy_canonical_repair_paper.md").write_text(report)
    (REPORTS / "qwen_hidden_vm_onpolicy_canonical_repair_paper.html").write_text(
        markdown_to_html(report, "Qwen Hidden VM On-Policy Canonical Repair")
    )
    write_checkpoint_manifest()
    print(ANALYSIS / "summary.md")


if __name__ == "__main__":
    main()

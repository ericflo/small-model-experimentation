#!/usr/bin/env python3
"""Build standalone reports for the structural latent compiler experiment."""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/workspace/experiments/qwen_structural_latent_compiler_expansion")
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def split_length(split: str) -> int:
    match = re.search(r"_L(\d+)$", str(split))
    return int(match.group(1)) if match else -1


def split_family(split: str) -> str:
    return str(split).split("_L", 1)[0]


def load_csvs(name: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob(f"*/{name}")):
        if path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if "run" not in df.columns:
            df["run"] = path.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_results() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/results.json")):
        try:
            out.append(json.loads(path.read_text()))
        except Exception:
            continue
    return out


def save_metric_chart(metrics: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    if metrics.empty or metric not in metrics.columns:
        return
    df = metrics.copy()
    df["length"] = df["split"].map(split_length)
    df["family"] = df["split"].map(split_family)
    df = df[df["length"] > 0]
    if df.empty:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for (run, family), sub in df.groupby(["run", "family"]):
        sub = sub.sort_values(["global_step", "length"])
        label = f"{run} / {family}"
        plt.plot(sub["length"], sub[metric], marker="o", linewidth=1.8, label=label)
    plt.title(title)
    plt.xlabel("Program length")
    plt.ylabel(metric)
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180)
    plt.close()


def save_stage_chart(metrics: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    if metrics.empty or metric not in metrics.columns:
        return
    df = metrics.copy()
    df["length"] = df["split"].map(split_length)
    df["family"] = df["split"].map(split_family)
    df = df[(df["length"] > 0) & df[metric].notna()]
    if df.empty:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for split, sub in df.groupby("split"):
        sub = sub.sort_values("global_step")
        plt.plot(sub["global_step"], sub[metric], marker="o", linewidth=1.8, label=split)
    plt.title(title)
    plt.xlabel("Global training step")
    plt.ylabel(metric)
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180)
    plt.close()


def save_train_chart(train: pd.DataFrame) -> None:
    if train.empty:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for run, sub in train.groupby("run"):
        sub = sub.sort_values("global_step")
        if "loss" in sub.columns:
            plt.plot(sub["global_step"], sub["loss"], linewidth=1.6, label=f"{run} loss")
    plt.title("Training Loss")
    plt.xlabel("Global training step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(FIGURES / "training_loss.png", dpi=180)
    plt.close()

    if "state_train_accuracy" in train.columns:
        plt.figure(figsize=(10, 5))
        for run, sub in train.groupby("run"):
            sub = sub.sort_values("global_step")
            plt.plot(sub["global_step"], sub["state_train_accuracy"], linewidth=1.6, label=f"{run}")
        plt.title("Training State Accuracy")
        plt.xlabel("Global training step")
        plt.ylabel("State accuracy")
        plt.ylim(-0.03, 1.03)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8, loc="best")
        plt.tight_layout()
        plt.savefig(FIGURES / "training_state_accuracy.png", dpi=180)
        plt.close()


def fmt_pct(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if math.isnan(val):
        return ""
    return f"{100.0 * val:.1f}%"


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    cols = [
        "run",
        "stage",
        "split",
        "global_step",
        "max_steps",
        "executor_accuracy",
        "program_exact",
        "state_prefix_fraction",
        "state_all_exact",
        "executor_pair_both_correct",
        "compiler_pair_state_consistency",
    ]
    present = [c for c in cols if c in metrics.columns]
    df = metrics[present].copy()
    if "split" in df.columns:
        df["length"] = df["split"].map(split_length)
    return df.sort_values([c for c in ["run", "global_step", "length", "split"] if c in df.columns])


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    use = df.head(max_rows).copy()
    for col in use.columns:
        if use[col].dtype.kind in "fc":
            if col.endswith("accuracy") or "fraction" in col or "correct" in col or "consistency" in col or col == "program_exact":
                use[col] = use[col].map(fmt_pct)
            else:
                use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    return use.to_markdown(index=False)


def build_report(metrics: pd.DataFrame, train: pd.DataFrame, results: List[Dict[str, Any]]) -> str:
    summary = build_summary(metrics)
    best_row = None
    final_l24 = pd.DataFrame()
    if not metrics.empty and "executor_accuracy" in metrics.columns:
        scored = metrics[metrics["executor_accuracy"].notna()].copy()
        if not scored.empty:
            best_row = scored.sort_values("executor_accuracy", ascending=False).iloc[0].to_dict()
        tmp = metrics.copy()
        tmp["length"] = tmp["split"].map(split_length)
        if not tmp.empty:
            max_step = tmp["global_step"].max()
            final_l24 = tmp[(tmp["global_step"] == max_step) & (tmp["length"] == 24)].copy()

    lines: List[str] = []
    lines.append("# Structural Latent Compiler Expansion")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(
        "Can a Qwen-attached executable latent compiler be expanded from short modular programs to longer modular programs while preserving direct executable accuracy, without beam search, candidate reranking, or tokenized program output?"
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- A Qwen causal LM reads the arithmetic prompt and fixed latent register markers.")
    lines.append("- A structural compiler head predicts one initial value plus typed operation and argument slots.")
    lines.append("- A differentiable modular executor supervises final answer probability and intermediate state traces.")
    lines.append("- The compiler is expanded by copying learned short-slot parameters into longer slot structures, then continuing training.")
    lines.append("- The run reports argmax executable accuracy, exact program recovery, state prefix recovery, and paraphrase-pair consistency.")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    if results:
        run_rows = []
        for result in results:
            meta = result.get("metadata", {})
            args = result.get("args", {})
            run_rows.append(
                {
                    "run": result.get("run"),
                    "elapsed_sec": result.get("elapsed_sec"),
                    "model": args.get("model_id"),
                    "stage_max_steps": args.get("stage_max_steps"),
                    "stage_steps": args.get("stage_steps"),
                    "train_examples": args.get("train_examples"),
                    "gpu": meta.get("gpu_name"),
                }
            )
        lines.append(markdown_table(pd.DataFrame(run_rows), max_rows=20))
    else:
        lines.append("_No completed runs found._")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    if not final_l24.empty:
        lines.append("Final expanded 24-slot compiler, length-24 splits:")
        lines.append("")
        keep = [
            "split",
            "executor_accuracy",
            "program_exact",
            "state_prefix_fraction",
            "executor_pair_both_correct",
            "compiler_pair_state_consistency",
        ]
        lines.append(markdown_table(final_l24[[c for c in keep if c in final_l24.columns]], max_rows=10))
        lines.append("")
    if best_row:
        lines.append(f"Best single split executable accuracy was {fmt_pct(best_row.get('executor_accuracy'))}; the length-24 table above is the main result.")
        lines.append("")
    lines.append(markdown_table(summary, max_rows=80))
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fig in [
        "executor_accuracy_by_length.png",
        "executor_accuracy_by_step.png",
        "program_exact_by_step.png",
        "state_prefix_by_step.png",
        "paired_state_consistency_by_step.png",
        "training_loss.png",
        "training_state_accuracy.png",
    ]:
        if (FIGURES / fig).exists():
            lines.append(f"![{fig}](figures/{fig})")
            lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This report is intentionally standalone. The key readout is whether expansion improves or preserves executable accuracy at longer lengths, and whether paraphrase-paired programs compile to the same latent execution trace."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Run outputs: `experiments/qwen_structural_latent_compiler_expansion/runs/`")
    lines.append("- Reports and figures: `experiments/qwen_structural_latent_compiler_expansion/reports/`")
    lines.append("- Large checkpoints: `large_artifacts/qwen_structural_latent_compiler_expansion/checkpoints/`")
    lines.append("")
    return "\n".join(lines)


def markdown_to_html(md: str) -> str:
    # A small Markdown subset renderer is enough for this report and avoids
    # depending on optional markdown packages.
    lines = md.splitlines()
    body: List[str] = []
    in_table = False
    table_lines: List[str] = []

    def flush_table() -> None:
        nonlocal in_table, table_lines
        if not table_lines:
            return
        rows = [line.strip().strip("|").split("|") for line in table_lines if line.strip()]
        if len(rows) >= 2:
            header = [cell.strip() for cell in rows[0]]
            data = rows[2:] if set(rows[1][0].strip()) <= {"-", ":"} else rows[1:]
            body.append("<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in header) + "</tr></thead><tbody>")
            for row in data:
                body.append("<tr>" + "".join(f"<td>{html.escape(cell.strip())}</td>" for cell in row) + "</tr>")
            body.append("</tbody></table>")
        table_lines = []
        in_table = False

    for line in lines:
        if line.startswith("|") and line.endswith("|"):
            in_table = True
            table_lines.append(line)
            continue
        if in_table:
            flush_table()
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body.append(f"<p class=\"bullet\">{html.escape(line[2:])}</p>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt = line[2:].split("](", 1)[0]
            src = line.split("](", 1)[1][:-1]
            body.append(f"<figure><img src=\"{html.escape(src)}\" alt=\"{html.escape(alt)}\"><figcaption>{html.escape(alt)}</figcaption></figure>")
        elif line.strip():
            body.append(f"<p>{html.escape(line)}</p>")
        else:
            body.append("")
    if in_table:
        flush_table()
    css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px auto; max-width: 1160px; line-height: 1.45; color: #17202a; }
h1, h2 { letter-spacing: 0; }
table { border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 13px; }
th, td { border: 1px solid #d7dde5; padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #eef3f8; }
figure { margin: 24px 0; }
img { max-width: 100%; border: 1px solid #d7dde5; }
.bullet::before { content: "- "; }
p { margin: 8px 0; }
"""
    return "<!doctype html><html><head><meta charset=\"utf-8\"><title>Structural Latent Compiler Expansion</title><style>" + css + "</style></head><body>" + "\n".join(body) + "</body></html>"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = load_csvs("metrics.csv")
    train = load_csvs("train_log.csv")
    results = load_results()
    if not metrics.empty:
        metrics["length"] = metrics["split"].map(split_length)
        metrics.to_csv(REPORTS / "aggregate_metrics.csv", index=False)
    if not train.empty:
        train.to_csv(REPORTS / "aggregate_train_log.csv", index=False)
    save_metric_chart(metrics, "executor_accuracy", "executor_accuracy_by_length.png", "Executable Accuracy by Length")
    save_stage_chart(metrics, "executor_accuracy", "executor_accuracy_by_step.png", "Executable Accuracy Across Expansion Stages")
    save_stage_chart(metrics, "program_exact", "program_exact_by_step.png", "Exact Program Recovery Across Expansion Stages")
    save_stage_chart(metrics, "state_prefix_fraction", "state_prefix_by_step.png", "State Prefix Recovery Across Expansion Stages")
    save_stage_chart(metrics, "compiler_pair_state_consistency", "paired_state_consistency_by_step.png", "Paraphrase Pair State Consistency")
    save_train_chart(train)
    md = build_report(metrics, train, results)
    (REPORTS / "structural_latent_compiler_expansion_report.md").write_text(md)
    (REPORTS / "structural_latent_compiler_expansion_report.html").write_text(markdown_to_html(md))
    print(REPORTS / "structural_latent_compiler_expansion_report.html")


if __name__ == "__main__":
    main()

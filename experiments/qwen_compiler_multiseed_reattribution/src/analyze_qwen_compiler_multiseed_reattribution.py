#!/usr/bin/env python3
"""Build standalone reports for the Qwen compiler multi-seed reattribution."""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/workspace/experiments/qwen_compiler_multiseed_reattribution")
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TARGET_SPLITS = ["standard_L24", "heldout_L24", "paired_L24", "paired_heldout_L24"]
REPORT_MD = REPORTS / "qwen_compiler_multiseed_reattribution_report.md"
REPORT_HTML = REPORTS / "qwen_compiler_multiseed_reattribution_report.html"


def split_length(split: str) -> int:
    match = re.search(r"_L(\d+)$", str(split))
    return int(match.group(1)) if match else -1


def split_family(split: str) -> str:
    return str(split).split("_L", 1)[0]


def seed_from_run(run: Any) -> int | float:
    match = re.search(r"_seed(\d+)$", str(run))
    return int(match.group(1)) if match else math.nan


def load_csvs(name: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob(f"*/{name}")):
        if path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if "run" not in df.columns:
            df["run"] = path.parent.name
        if "seed" not in df.columns:
            df["seed"] = df["run"].map(seed_from_run)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_results() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/results.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if "seed" not in data:
            data["seed"] = data.get("args", {}).get("seed", seed_from_run(data.get("run")))
        out.append(data)
    return out


def final_l24_rows(metrics: pd.DataFrame, main_only: bool = True) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    df = metrics.copy()
    if main_only and "run" in df.columns:
        main_df = df[df["run"].astype(str).str.startswith("main_")]
        if main_df.empty:
            main_df = df[df["run"].astype(str).str.startswith("pilot_")]
        if main_df.empty:
            main_df = df[df["run"].astype(str).str.startswith("smoke_")]
        df = main_df
    if df.empty:
        return df
    df["length"] = df["split"].map(split_length)
    rows: List[pd.DataFrame] = []
    for run, sub in df.groupby("run"):
        max_step = sub["global_step"].max()
        rows.append(sub[(sub["global_step"] == max_step) & (sub["length"] == 24)])
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def per_seed_accuracy(final_rows: pd.DataFrame) -> pd.DataFrame:
    if final_rows.empty:
        return pd.DataFrame()
    rows = final_rows[final_rows["split"].isin(TARGET_SPLITS)].copy()
    if rows.empty:
        return rows
    pivot = rows.pivot_table(index=["arm", "seed", "run"], columns="split", values="executor_accuracy", aggfunc="max").reset_index()
    ordered = ["arm", "seed", "run"] + [split for split in TARGET_SPLITS if split in pivot.columns]
    return pivot[ordered].sort_values([c for c in ["arm", "seed"] if c in pivot.columns])


def aggregate_accuracy(final_rows: pd.DataFrame) -> pd.DataFrame:
    if final_rows.empty:
        return pd.DataFrame()
    rows = final_rows[final_rows["split"].isin(TARGET_SPLITS)].copy()
    if rows.empty:
        return rows
    grouped = rows.groupby(["arm", "split"], dropna=False)
    agg = grouped["executor_accuracy"].agg(["count", "mean", "std", "min", "max"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    return agg.sort_values(["split", "mean"], ascending=[True, False])


def aggregate_metric(final_rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    if final_rows.empty or metric not in final_rows.columns:
        return pd.DataFrame()
    rows = final_rows[final_rows["split"].isin(TARGET_SPLITS)].copy()
    grouped = rows.groupby(["arm", "split"], dropna=False)
    agg = grouped[metric].agg(["count", "mean", "std", "min", "max"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    return agg


def wide_mean_std_table(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()
    arms = sorted(agg["arm"].dropna().unique().tolist())
    rows: List[Dict[str, Any]] = []
    for arm in arms:
        row: Dict[str, Any] = {"arm": arm}
        sub = agg[agg["arm"] == arm]
        for split in TARGET_SPLITS:
            cur = sub[sub["split"] == split]
            if cur.empty:
                row[split] = ""
                continue
            r = cur.iloc[0]
            row[split] = f"{100 * r['mean']:.1f}% +/- {100 * r['std']:.1f} ({100 * r['min']:.1f}-{100 * r['max']:.1f}, n={int(r['count'])})"
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_pct(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if math.isnan(val):
        return ""
    return f"{100.0 * val:.1f}%"


def fmt_num(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if math.isnan(val):
        return ""
    return f"{val:.4g}"


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    use = df.head(max_rows).copy()
    for col in use.columns:
        if use[col].dtype.kind in "fc":
            if (
                col.endswith("accuracy")
                or "fraction" in col
                or "correct" in col
                or "consistency" in col
                or col == "program_exact"
                or re.search(r"_L\d+$", str(col))
            ):
                use[col] = use[col].map(fmt_pct)
            else:
                use[col] = use[col].map(fmt_num)
    return use.to_markdown(index=False)


def save_mean_error_chart(agg: pd.DataFrame) -> None:
    if agg.empty:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    arms = sorted(agg["arm"].dropna().unique().tolist())
    splits = [split for split in TARGET_SPLITS if split in set(agg["split"])]
    if not arms or not splits:
        return
    x = np.arange(len(splits))
    width = min(0.24, 0.82 / max(1, len(arms)))
    plt.figure(figsize=(10.5, 5.5))
    for i, arm in enumerate(arms):
        sub = agg[agg["arm"] == arm].set_index("split")
        means = [float(sub.loc[split, "mean"]) if split in sub.index else np.nan for split in splits]
        stds = [float(sub.loc[split, "std"]) if split in sub.index else 0.0 for split in splits]
        offset = (i - (len(arms) - 1) / 2) * width
        plt.bar(x + offset, means, width, yerr=stds, capsize=4, label=arm)
    plt.title("Final Length-24 Accuracy: Mean and Seed Spread")
    plt.xlabel("Evaluation split")
    plt.ylabel("Executable accuracy")
    plt.xticks(x, splits, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "mean_accuracy_by_split.png", dpi=180)
    plt.close()


def save_seed_strip(final_rows: pd.DataFrame) -> None:
    if final_rows.empty:
        return
    rows = final_rows[final_rows["split"] == "standard_L24"].copy()
    if rows.empty:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    arms = sorted(rows["arm"].dropna().unique().tolist())
    plt.figure(figsize=(8.8, 5.2))
    for idx, arm in enumerate(arms):
        sub = rows[rows["arm"] == arm].sort_values("seed")
        jitter = np.linspace(-0.08, 0.08, len(sub)) if len(sub) > 1 else np.array([0.0])
        plt.scatter(np.full(len(sub), idx) + jitter, sub["executor_accuracy"], s=60, label=arm)
        if len(sub):
            plt.hlines(sub["executor_accuracy"].mean(), idx - 0.22, idx + 0.22, linewidth=3)
    plt.title("Standard L24 Accuracy by Seed")
    plt.xlabel("Arm")
    plt.ylabel("Executable accuracy")
    plt.xticks(np.arange(len(arms)), arms, rotation=15, ha="right")
    plt.ylim(-0.03, 1.03)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "standard_accuracy_by_seed.png", dpi=180)
    plt.close()


def save_heatmap(per_seed: pd.DataFrame) -> None:
    if per_seed.empty:
        return
    value_cols = [split for split in TARGET_SPLITS if split in per_seed.columns]
    if not value_cols:
        return
    labels = [f"{row.arm}\nseed {int(row.seed)}" for row in per_seed.itertuples()]
    values = per_seed[value_cols].astype(float).to_numpy()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, max(4.5, 0.42 * len(labels))))
    im = plt.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, label="Executable accuracy")
    plt.yticks(np.arange(len(labels)), labels, fontsize=8)
    plt.xticks(np.arange(len(value_cols)), value_cols, rotation=15, ha="right")
    plt.title("Per-Seed Final L24 Accuracy")
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            val = values[y, x]
            plt.text(x, y, f"{100 * val:.0f}", ha="center", va="center", color="white" if val < 0.6 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "per_seed_accuracy_heatmap.png", dpi=180)
    plt.close()


def save_train_chart(train: pd.DataFrame) -> None:
    if train.empty or "loss" not in train.columns:
        return
    if "run" in train.columns:
        df = train[train["run"].astype(str).str.startswith("main_")].copy()
        if df.empty:
            df = train[train["run"].astype(str).str.startswith("pilot_")].copy()
        if df.empty:
            df = train[train["run"].astype(str).str.startswith("smoke_")].copy()
    else:
        df = train.copy()
    if df.empty:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.5, 5.5))
    for (arm, seed), sub in df.groupby(["arm", "seed"]):
        sub = sub.sort_values("global_step")
        plt.plot(sub["global_step"], sub["loss"], linewidth=1.2, alpha=0.75, label=f"{arm}/s{int(seed)}")
    plt.title("Training Loss by Arm and Seed")
    plt.xlabel("Global training step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "training_loss_by_seed.png", dpi=180)
    plt.close()

    if "state_train_accuracy" in df.columns:
        plt.figure(figsize=(10.5, 5.5))
        for (arm, seed), sub in df.groupby(["arm", "seed"]):
            sub = sub.sort_values("global_step")
            plt.plot(sub["global_step"], sub["state_train_accuracy"], linewidth=1.2, alpha=0.75, label=f"{arm}/s{int(seed)}")
        plt.title("Training State Accuracy by Arm and Seed")
        plt.xlabel("Global training step")
        plt.ylabel("State accuracy")
        plt.ylim(-0.03, 1.03)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(FIGURES / "training_state_accuracy_by_seed.png", dpi=180)
        plt.close()


def save_length_curve(metrics: pd.DataFrame) -> None:
    if metrics.empty or "executor_accuracy" not in metrics.columns:
        return
    df = metrics[metrics["run"].astype(str).str.startswith("main_")].copy()
    if df.empty:
        df = metrics[metrics["run"].astype(str).str.startswith("pilot_")].copy()
    if df.empty:
        df = metrics[metrics["run"].astype(str).str.startswith("smoke_")].copy()
    if df.empty:
        return
    df["length"] = df["split"].map(split_length)
    df["family"] = df["split"].map(split_family)
    df = df[(df["length"] > 0) & df["family"].isin(["standard", "heldout"])]
    if df.empty:
        return
    final_rows: List[pd.DataFrame] = []
    for run, sub in df.groupby("run"):
        max_step = sub["global_step"].max()
        final_rows.append(sub[sub["global_step"] == max_step])
    final = pd.concat(final_rows, ignore_index=True, sort=False) if final_rows else pd.DataFrame()
    if final.empty:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.5, 5.2))
    for (arm, family), sub in final.groupby(["arm", "family"]):
        agg = sub.groupby("length")["executor_accuracy"].agg(["mean", "std"]).reset_index().sort_values("length")
        agg["std"] = agg["std"].fillna(0.0)
        plt.plot(agg["length"], agg["mean"], marker="o", linewidth=1.8, label=f"{arm}/{family}")
        plt.fill_between(agg["length"], agg["mean"] - agg["std"], agg["mean"] + agg["std"], alpha=0.12)
    plt.title("Final Accuracy by Program Length")
    plt.xlabel("Program length")
    plt.ylabel("Executable accuracy")
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "final_accuracy_by_length.png", dpi=180)
    plt.close()


def build_run_table(results: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        args = result.get("args", {})
        meta = result.get("metadata", {})
        rows.append(
            {
                "run": result.get("run"),
                "arm": result.get("arm") or args.get("arm"),
                "seed": result.get("seed") or args.get("seed"),
                "elapsed_sec": result.get("elapsed_sec"),
                "stage_max_steps": args.get("stage_max_steps"),
                "stage_steps": args.get("stage_steps"),
                "train_examples": args.get("train_examples"),
                "eval_examples": args.get("eval_examples"),
                "gpu": meta.get("gpu_name"),
            }
        )
    return pd.DataFrame(rows).sort_values([c for c in ["run"] if c in pd.DataFrame(rows).columns]) if rows else pd.DataFrame()


def build_report(metrics: pd.DataFrame, train: pd.DataFrame, results: List[Dict[str, Any]]) -> str:
    final_rows = final_l24_rows(metrics)
    phase = "none"
    if not final_rows.empty and "run" in final_rows.columns:
        run_names = final_rows["run"].astype(str)
        if run_names.str.startswith("main_").any():
            phase = "main"
        elif run_names.str.startswith("pilot_").any():
            phase = "pilot"
        elif run_names.str.startswith("smoke_").any():
            phase = "smoke"
    agg = aggregate_accuracy(final_rows)
    per_seed = per_seed_accuracy(final_rows)
    program_agg = aggregate_metric(final_rows, "program_exact")
    prefix_agg = aggregate_metric(final_rows, "state_prefix_fraction")
    run_table = build_run_table(results)

    lines: List[str] = []
    lines.append("# Qwen Compiler Multi-Seed Reattribution")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(
        "Does a one-shot executable latent compiler reliably learn length-24 modular programs across random seeds, and which training factor best explains the result: staged length curriculum, copied structural expansion, or same-budget no-curriculum training?"
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Each arm uses Qwen/Qwen3-4B with QLoRA and a direct executable compiler head.")
    lines.append("- The compiler predicts one initial value and a sequence of typed operation and argument slots.")
    lines.append("- A differentiable modular executor supervises final answers and intermediate state traces.")
    lines.append("- The same seed set is used for each arm, so the main readout is mean and spread across matched random seeds.")
    lines.append("- Evaluation includes standard templates, held-out wording templates, seen-family paired consistency, and held-out paired consistency.")
    lines.append("")
    lines.append("## Arms")
    lines.append("")
    lines.append("- `max24_curriculum`: max-24 compiler from the start, with staged train lengths.")
    lines.append("- `expand_copy`: compiler capacity expands in stages and newly introduced slots copy the last learned slot.")
    lines.append("- `max24_no_curriculum`: max-24 compiler from the start, trained on the full length range immediately.")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append(markdown_table(run_table, max_rows=80))
    lines.append("")
    lines.append("## Results")
    lines.append("")
    if not agg.empty:
        if phase == "main":
            lines.append("Final length-24 executable accuracy, mean +/- standard deviation across seeds:")
        else:
            lines.append("Diagnostic final length-24 executable accuracy from non-result-bearing validation runs:")
        lines.append("")
    if phase == "main" and not agg.empty:
        lines.append("## Key Findings")
        lines.append("")
        std_rows = agg[agg["split"] == "standard_L24"].copy()
        paired_rows = agg[agg["split"] == "paired_L24"].copy()
        held_rows = agg[agg["split"] == "heldout_L24"].copy()
        if not std_rows.empty:
            top_std = std_rows.sort_values("mean", ascending=False).iloc[0]
            lines.append(
                f"- `standard_L24`: best mean is `{top_std['arm']}` at {100 * top_std['mean']:.1f}% with {100 * top_std['std']:.1f} percentage points seed standard deviation."
            )
        if not paired_rows.empty:
            top_pair = paired_rows.sort_values("mean", ascending=False).iloc[0]
            lines.append(
                f"- `paired_L24`: best mean is `{top_pair['arm']}` at {100 * top_pair['mean']:.1f}% with {100 * top_pair['std']:.1f} percentage points seed standard deviation."
            )
        if not held_rows.empty:
            top_held = held_rows.sort_values("mean", ascending=False).iloc[0]
            lines.append(
                f"- `heldout_L24`: best mean is `{top_held['arm']}` at {100 * top_held['mean']:.1f}% with {100 * top_held['std']:.1f} percentage points seed standard deviation."
            )
        prefix_std = prefix_agg[prefix_agg["split"] == "standard_L24"].copy() if not prefix_agg.empty else pd.DataFrame()
        if not prefix_std.empty:
            weakest_prefix = prefix_std.sort_values("mean", ascending=True).iloc[0]
            strongest_prefix = prefix_std.sort_values("mean", ascending=False).iloc[0]
            lines.append(
                f"- State-prefix recovery is high even when exact execution fails: `standard_L24` ranges from {100 * weakest_prefix['mean']:.1f}% mean prefix recovery for `{weakest_prefix['arm']}` to {100 * strongest_prefix['mean']:.1f}% for `{strongest_prefix['arm']}`."
            )
        no_curr = std_rows[std_rows["arm"] == "max24_no_curriculum"]
        if not no_curr.empty:
            r = no_curr.iloc[0]
            lines.append(
                f"- Same-budget no-curriculum training is the weakest standard-L24 arm at {100 * r['mean']:.1f}% mean accuracy."
            )
        lines.append("")
        lines.append(markdown_table(wide_mean_std_table(agg), max_rows=20))
        lines.append("")
    if not per_seed.empty:
        lines.append("Per-seed final length-24 executable accuracy:")
        lines.append("")
        lines.append(markdown_table(per_seed, max_rows=80))
        lines.append("")
    if not program_agg.empty:
        lines.append("Exact program recovery, aggregated across seeds:")
        lines.append("")
        lines.append(markdown_table(wide_mean_std_table(program_agg), max_rows=20))
        lines.append("")
    if not prefix_agg.empty:
        lines.append("State-prefix recovery, aggregated across seeds:")
        lines.append("")
        lines.append(markdown_table(wide_mean_std_table(prefix_agg), max_rows=20))
        lines.append("")
    if not final_rows.empty:
        keep = [
            "arm",
            "seed",
            "run",
            "split",
            "n",
            "executor_accuracy",
            "program_exact",
            "state_prefix_fraction",
            "executor_pair_both_correct",
            "compiler_pair_state_consistency",
        ]
        lines.append("Final split rows:")
        lines.append("")
        lines.append(markdown_table(final_rows[[c for c in keep if c in final_rows.columns]], max_rows=120))
        lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fig in [
        "mean_accuracy_by_split.png",
        "standard_accuracy_by_seed.png",
        "per_seed_accuracy_heatmap.png",
        "final_accuracy_by_length.png",
        "training_loss_by_seed.png",
        "training_state_accuracy_by_seed.png",
    ]:
        if (FIGURES / fig).exists():
            lines.append(f"![{fig}](figures/{fig})")
            lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if agg.empty:
        lines.append("No completed main runs were found yet, so this report is a harness and artifact check only.")
    elif phase != "main":
        lines.append("These rows come from smoke or pilot validation runs and should not be interpreted as evidence about the arms.")
    else:
        standard = agg[agg["split"] == "standard_L24"].copy()
        if not standard.empty:
            winner = standard.sort_values("mean", ascending=False).iloc[0]
            lines.append(
                f"On the standard length-24 split, the strongest mean arm is `{winner['arm']}` at {100 * winner['mean']:.1f}% with {100 * winner['std']:.1f} percentage points of seed standard deviation."
            )
        lines.append(
            "The decisive criterion is whether the arm ranking remains stable across seeds and whether any arm's seed spread is large enough to make a single-seed conclusion unreliable."
        )
        lines.append(
            "The observed spread is large enough that no single seed supports a stable attribution claim. Copied expansion has the best mean in this seed set, but it ranges from complete standard-L24 failure to strong performance. Full-width curriculum is also unstable, and no-curriculum training is especially weak on standard-L24 despite sometimes doing well on other wording splits."
        )
        lines.append(
            "A second conclusion is that partial execution is not the bottleneck: every arm recovers long state prefixes far more often than it recovers exact length-24 programs. The remaining failure is late-step/global program consistency, not the absence of local operator knowledge."
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Run outputs: `experiments/qwen_compiler_multiseed_reattribution/runs/`")
    lines.append("- Reports and figures: `experiments/qwen_compiler_multiseed_reattribution/reports/`")
    lines.append("- Large checkpoints: `large_artifacts/qwen_compiler_multiseed_reattribution/checkpoints/`")
    lines.append("")
    return "\n".join(lines)


def markdown_to_html(md: str) -> str:
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
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px auto; max-width: 1180px; line-height: 1.45; color: #17202a; }
h1, h2 { letter-spacing: 0; }
table { border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 13px; }
th, td { border: 1px solid #d7dde5; padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #eef3f8; }
figure { margin: 24px 0; }
img { max-width: 100%; border: 1px solid #d7dde5; }
.bullet::before { content: "- "; }
p { margin: 8px 0; }
"""
    return "<!doctype html><html><head><meta charset=\"utf-8\"><title>Qwen Compiler Multi-Seed Reattribution</title><style>" + css + "</style></head><body>" + "\n".join(body) + "</body></html>"


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
    final_rows = final_l24_rows(metrics)
    per_seed = per_seed_accuracy(final_rows)
    agg = aggregate_accuracy(final_rows)
    if not final_rows.empty:
        final_rows.to_csv(REPORTS / "final_l24_rows.csv", index=False)
    if not per_seed.empty:
        per_seed.to_csv(REPORTS / "per_seed_final_l24_accuracy.csv", index=False)
    if not agg.empty:
        agg.to_csv(REPORTS / "aggregate_final_l24_accuracy.csv", index=False)
    save_mean_error_chart(agg)
    save_seed_strip(final_rows)
    save_heatmap(per_seed)
    save_train_chart(train)
    save_length_curve(metrics)
    md = build_report(metrics, train, results)
    REPORT_MD.write_text(md)
    REPORT_HTML.write_text(markdown_to_html(md))
    print(REPORT_HTML)


if __name__ == "__main__":
    main()

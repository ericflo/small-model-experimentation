#!/usr/bin/env python3
"""Aggregate support-contrastive meta-ICL runs."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXP_NAME = "qwen_support_contrastive_meta_icl"
ROOT = Path("/workspace/experiments") / EXP_NAME
LARGE_ROOT = Path("/workspace/large_artifacts") / EXP_NAME
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGS = ANALYSIS / "figures"
REPORTS = ROOT / "reports"


def pct(x: object) -> str:
    try:
        y = float(x)
    except Exception:
        return "" if x is None else str(x)
    if math.isnan(y):
        return ""
    return f"{100 * y:.1f}%"


def arm_name(method: str, cfg: dict[str, object]) -> str:
    if method == "base":
        return "base"
    objective = str(cfg.get("objective", "unknown"))
    train_mode = str(cfg.get("train_mode", "unknown"))
    if objective == "support_contrastive" and train_mode == "counterfactual":
        return "contrastive_cf"
    if objective == "ce" and train_mode == "counterfactual":
        return "ce_cf"
    if objective == "ce" and train_mode == "ordinary":
        return "ce_ordinary"
    if objective == "ce" and train_mode == "shuffled_labels":
        return "ce_shuffled_labels"
    return f"{objective}_{train_mode}"


def md_table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int | None = None) -> str:
    if cols:
        df = df[cols]
    if max_rows is not None:
        df = df.head(max_rows)
    tmp = df.copy()
    for c in tmp.columns:
        if c in {"normal", "shuffled", "none", "contrast", "delta"} or "exact" in c or c.endswith("_mean") or c.endswith("_std") or c.endswith("_gap"):
            tmp[c] = tmp[c].map(lambda x: pct(x) if pd.notna(x) else "")
    return tmp.to_markdown(index=False) if len(tmp) else "_No rows._"


def load_runs(include_smoke: bool = False, include_pilot: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    tasks: list[pd.DataFrame] = []
    rows: list[pd.DataFrame] = []
    logs: list[pd.DataFrame] = []
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists() or not (run_dir / "summary.csv").exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not include_smoke and (bool(cfg.get("smoke")) or run_dir.name.startswith("smoke")):
            continue
        if not include_pilot and run_dir.name.startswith("pilot"):
            continue
        for filename, acc in [("summary.csv", summaries), ("task_metrics.csv", tasks), ("row_predictions.csv", rows)]:
            df = pd.read_csv(run_dir / filename)
            df["run_name"] = run_dir.name
            df["seed"] = int(cfg.get("seed", -1))
            df["objective"] = str(cfg.get("objective", "unknown"))
            df["train_mode"] = str(cfg.get("train_mode", "unknown"))
            df["arm"] = [arm_name(m, cfg) for m in df["method"]]
            acc.append(df)
        train_path = run_dir / "training_log.csv"
        if train_path.exists():
            log = pd.read_csv(train_path)
            if len(log):
                log["run_name"] = run_dir.name
                log["seed"] = int(cfg.get("seed", -1))
                log["arm"] = arm_name("adapter", cfg)
                logs.append(log)
    if not summaries:
        raise SystemExit(f"No completed runs found in {RUNS}")
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(tasks, ignore_index=True),
        pd.concat(rows, ignore_index=True),
        pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(),
    )


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    out = (
        summary.groupby(["arm", "split", "support_mode"], as_index=False)
        .agg(
            runs=("run_name", "nunique"),
            tasks=("tasks", "mean"),
            rows=("rows", "mean"),
            row_exact_mean=("row_exact", "mean"),
            row_exact_std=("row_exact", "std"),
            full_task_exact_mean=("full_task_exact", "mean"),
            full_task_exact_std=("full_task_exact", "std"),
        )
        .sort_values(["split", "arm", "support_mode"])
        .reset_index(drop=True)
    )
    for c in ["row_exact_std", "full_task_exact_std"]:
        out[c] = out[c].fillna(0.0)
    return out


def value(agg: pd.DataFrame, arm: str, split: str, support: str, col: str = "full_task_exact_mean") -> float | None:
    m = agg[(agg.arm == arm) & (agg.split == split) & (agg.support_mode == support)]
    return float(m.iloc[0][col]) if len(m) else None


def add_support_gaps(agg: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (arm, split), g in agg.groupby(["arm", "split"]):
        normal = value(agg, arm, split, "normal") or 0.0
        shuffled = value(agg, arm, split, "shuffled") or 0.0
        none = value(agg, arm, split, "none") or 0.0
        contrast = value(agg, arm, split, "contrast")
        rows.append(
            {
                "arm": arm,
                "split": split,
                "normal": normal,
                "shuffled": shuffled,
                "none": none,
                "contrast": contrast,
                "normal_minus_shuffled_gap": normal - shuffled,
                "normal_minus_none_gap": normal - none,
                "normal_minus_contrast_gap": normal - contrast if contrast is not None else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "arm"]).reset_index(drop=True)


def task_flips(task_df: pd.DataFrame, arm: str, split: str = "public_prose") -> pd.DataFrame:
    base = task_df[(task_df.arm == "base") & (task_df.split == split) & (task_df.support_mode == "normal")]
    comp = task_df[(task_df.arm == arm) & (task_df.split == split) & (task_df.support_mode == "normal")]
    if base.empty or comp.empty:
        return pd.DataFrame()
    merged = comp.merge(base[["task_id", "full_task_exact"]].rename(columns={"full_task_exact": "base_full_task_exact"}), on="task_id")
    merged["delta"] = merged["full_task_exact"].astype(float) - merged["base_full_task_exact"].astype(float)
    return (
        merged.groupby(["run_name", "seed"], as_index=False)
        .agg(helped=("delta", lambda s: int((s > 0).sum())), hurt=("delta", lambda s: int((s < 0).sum())), tied=("delta", lambda s: int((s == 0).sum())), tasks=("delta", "size"))
        .sort_values("run_name")
    )


def family_table(task_df: pd.DataFrame, arm: str = "contrastive_cf", split: str = "public_prose") -> pd.DataFrame:
    df = task_df[(task_df.arm.isin(["base", arm])) & (task_df.split == split) & (task_df.support_mode == "normal")]
    if df.empty:
        return pd.DataFrame()
    tab = (
        df.groupby(["arm", "family"], as_index=False)
        .agg(tasks=("task_id", "nunique"), row_exact=("row_exact", "mean"), full_task_exact=("full_task_exact", "mean"))
        .pivot(index="family", columns="arm", values=["tasks", "row_exact", "full_task_exact"])
    )
    tab.columns = [f"{a}_{b}" for a, b in tab.columns]
    tab = tab.reset_index().fillna(0)
    if f"full_task_exact_{arm}" in tab and "full_task_exact_base" in tab:
        tab["delta"] = tab[f"full_task_exact_{arm}"] - tab["full_task_exact_base"]
        tab = tab.sort_values("delta", ascending=False)
    return tab


def plot_all(agg: pd.DataFrame, summary: pd.DataFrame, task_df: pd.DataFrame, train_log: pd.DataFrame) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    main = agg[(agg.support_mode == "normal") & (agg.split.isin(["public_prose", "synthetic_counterfactual"]))]
    main = main.sort_values(["split", "full_task_exact_mean"], ascending=[True, False])
    labels = [f"{r.arm}\n{r.split.replace('_counterfactual','').replace('public_prose','public')}" for r in main.itertuples()]
    colors = ["#455a64" if r.arm == "base" else "#2a9d8f" if r.arm == "contrastive_cf" else "#d9822b" for r in main.itertuples()]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(main)), main["full_task_exact_mean"], yerr=main["full_task_exact_std"], color=colors, capsize=5)
    plt.xticks(range(len(main)), labels, rotation=35, ha="right", fontsize=8)
    plt.ylabel("Full-task exact")
    plt.ylim(0, 1)
    plt.title("Normal-Support Full-Task Exact")
    plt.tight_layout()
    plt.savefig(FIGS / "aggregate_normal_full_task_exact.png", dpi=180)
    plt.close()

    gaps = add_support_gaps(agg)
    public_gaps = gaps[gaps.split == "public_prose"]
    if len(public_gaps):
        plt.figure(figsize=(9, 5))
        x = range(len(public_gaps))
        plt.bar(x, public_gaps["normal_minus_shuffled_gap"], color="#6c5ce7", label="normal - shuffled")
        plt.bar(x, public_gaps["normal_minus_none_gap"], fill=False, edgecolor="#111827", linewidth=2, label="normal - none")
        plt.xticks(list(x), public_gaps["arm"], rotation=25, ha="right")
        plt.ylabel("Full-task exact gap")
        plt.ylim(0, 1)
        plt.title("Public Support-Dependence Gaps")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGS / "public_support_gaps.png", dpi=180)
        plt.close()

    if len(train_log):
        plt.figure(figsize=(10, 5))
        for run_name, g in train_log.groupby("run_name"):
            plt.plot(g["step"], g["loss"], label=run_name)
        plt.xlabel("Optimizer step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGS / "aggregate_training_loss.png", dpi=180)
        plt.close()

    flips = task_flips(task_df, "contrastive_cf")
    if len(flips):
        plt.figure(figsize=(8, 5))
        bottom = [0] * len(flips)
        for col, color in [("helped", "#2a9d8f"), ("hurt", "#e76f51"), ("tied", "#9aa0a6")]:
            plt.bar(flips["run_name"], flips[col], bottom=bottom, color=color, label=col)
            bottom = [b + v for b, v in zip(bottom, flips[col])]
        plt.ylabel("Public tasks")
        plt.title("Contrastive Public Task Flips Versus Base")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGS / "contrastive_task_flips.png", dpi=180)
        plt.close()


def write_report(agg: pd.DataFrame, summary: pd.DataFrame, task_df: pd.DataFrame, row_df: pd.DataFrame, train_log: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    report_md = REPORTS / f"{EXP_NAME}_report.md"
    report_html = REPORTS / f"{EXP_NAME}_report.html"
    gaps = add_support_gaps(agg)
    flips = task_flips(task_df, "contrastive_cf")
    fam = family_table(task_df)
    per_run = summary[(summary.support_mode == "normal") & (summary.split.isin(["public_prose", "synthetic_counterfactual"]))].sort_values(["arm", "run_name", "split"])

    base_pub = value(agg, "base", "public_prose", "normal")
    con_pub = value(agg, "contrastive_cf", "public_prose", "normal")
    con_pub_std = value(agg, "contrastive_cf", "public_prose", "normal", "full_task_exact_std")
    ce_cf_pub = value(agg, "ce_cf", "public_prose", "normal")
    ce_ord_pub = value(agg, "ce_ordinary", "public_prose", "normal")
    ce_shuf_pub = value(agg, "ce_shuffled_labels", "public_prose", "normal")
    con_pub_shuf = value(agg, "contrastive_cf", "public_prose", "shuffled")
    con_pub_none = value(agg, "contrastive_cf", "public_prose", "none")

    lines: list[str] = []
    lines.append("# Support-Contrastive Meta-ICL")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("Can a support-contrastive LoRA objective make Qwen3-4B answer public text-transformation tasks better while making the answer depend more strongly on intact support examples?")
    lines.append("")
    lines.append("The contrastive arm trains on synthetic few-shot transformations. For each target answer, the positive prompt contains intact support examples, while negative prompts contain shuffled support labels, no support examples, or a counterfactual support set from an incompatible rule. Public benchmark labels are used only for evaluation.")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    if base_pub is not None and con_pub is not None:
        lines.append(f"- Public full-task exact: base `{pct(base_pub)}`; contrastive arm `{pct(con_pub)}` with seed spread `{pct(con_pub_std)}`.")
    if con_pub is not None:
        lines.append(f"- Contrastive public support controls: normal `{pct(con_pub)}`, shuffled `{pct(con_pub_shuf)}`, no support `{pct(con_pub_none)}`.")
    if ce_cf_pub is not None:
        lines.append(f"- CE-only counterfactual control: `{pct(ce_cf_pub)}`.")
    if ce_ord_pub is not None:
        lines.append(f"- CE-only ordinary control: `{pct(ce_ord_pub)}`.")
    if ce_shuf_pub is not None:
        lines.append(f"- CE-only shuffled-label control: `{pct(ce_shuf_pub)}`.")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append(md_table(agg, ["arm", "split", "support_mode", "runs", "tasks", "rows", "row_exact_mean", "row_exact_std", "full_task_exact_mean", "full_task_exact_std"]))
    lines.append("")
    lines.append("## Support-Dependence Gaps")
    lines.append("")
    lines.append(md_table(gaps))
    lines.append("")
    lines.append("## Seed-Level Normal-Support Metrics")
    lines.append("")
    lines.append(md_table(per_run, ["run_name", "seed", "arm", "split", "tasks", "rows", "row_exact", "full_task_exact"]))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if con_pub is not None and base_pub is not None:
        if con_pub > base_pub + 0.05:
            lines.append("The support-contrastive arm improves public strict task consistency over the frozen base model.")
        else:
            lines.append("The support-contrastive arm does not clearly improve public strict task consistency over the frozen base model.")
    if con_pub is not None and con_pub_shuf is not None and con_pub_none is not None:
        if con_pub > con_pub_shuf + 0.10 and con_pub > con_pub_none + 0.20:
            lines.append("The contrastive arm is strongly support-sensitive at evaluation time.")
        else:
            lines.append("The contrastive arm is not strongly enough separated from corrupted-support controls to prove causal support use.")
    controls = [x for x in [ce_cf_pub, ce_ord_pub, ce_shuf_pub] if x is not None]
    if con_pub is not None and controls:
        best_control = max(controls)
        if con_pub > best_control + 0.05:
            lines.append("The margin objective beats the CE-only control family on public transfer.")
        elif abs(con_pub - best_control) <= 0.025:
            lines.append("The margin objective ties the best CE-only control, so the contrastive ingredient is not isolated at this scale.")
        else:
            lines.append("A CE-only control beats the support-contrastive arm, so this objective is not the current best recipe.")
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    for name, caption in [
        ("aggregate_normal_full_task_exact.png", "Normal-support full-task exact"),
        ("public_support_gaps.png", "Public support-dependence gaps"),
        ("contrastive_task_flips.png", "Contrastive task flips versus base"),
        ("aggregate_training_loss.png", "Training loss"),
    ]:
        if (FIGS / name).exists():
            lines.append(f"![{caption}](../analysis/figures/{name})")
            lines.append("")
    lines.append("## Contrastive Public Task Flips Versus Base")
    lines.append("")
    lines.append(md_table(flips))
    lines.append("")
    lines.append("## Public Family Breakdown")
    lines.append("")
    lines.append(md_table(fam, max_rows=80))
    lines.append("")
    misses = row_df[(row_df.arm == "contrastive_cf") & (row_df.split == "public_prose") & (row_df.support_mode == "normal") & (~row_df["exact"].astype(bool))]
    lines.append("## Public Error Sample")
    lines.append("")
    lines.append(md_table(misses.sort_values(["run_name", "task_id", "query_index"])[["run_name", "task_id", "family", "input", "target", "prediction"]], max_rows=60))
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Experiment root: `{ROOT}`")
    lines.append(f"- Large artifacts root: `{LARGE_ROOT}`")
    lines.append(f"- Runs: `{RUNS}`")
    lines.append(f"- Checkpoints: `{LARGE_ROOT / 'checkpoints'}`")
    lines.append(f"- Aggregate summary CSV: `{ANALYSIS / 'aggregate_summary.csv'}`")
    lines.append(f"- Combined task metrics: `{ANALYSIS / 'aggregate_task_metrics.csv'}`")
    lines.append(f"- Combined row predictions: `{ANALYSIS / 'aggregate_row_predictions.csv'}`")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("The public evaluation is capped for runtime. Exact-match scoring is strict. The main contrastive arm is multiseed; CE controls may be single-seed depending on the completed run matrix. The synthetic task generator covers a limited set of text transformations.")
    report_md.write_text("\n".join(lines), encoding="utf-8")

    try:
        import markdown  # type: ignore

        body = markdown.markdown("\n".join(lines), extensions=["tables"])
    except Exception:
        body = "<pre>" + html.escape("\n".join(lines)) + "</pre>"
    report_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Support-Contrastive Meta-ICL</title>"
        "<style>body{font-family:Inter,Arial,sans-serif;max-width:1200px;margin:32px auto;line-height:1.48;color:#1f2937}"
        "table{border-collapse:collapse;font-size:13px;margin:14px 0}td,th{border:1px solid #d1d5db;padding:4px 7px}th{background:#f3f4f6}"
        "img{max-width:100%;margin:14px 0;border:1px solid #e5e7eb}code{background:#f3f4f6;padding:1px 4px;border-radius:4px}</style></head><body>"
        + body
        + "</body></html>",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_smoke", action="store_true")
    ap.add_argument("--include_pilot", action="store_true")
    args = ap.parse_args()
    summary, task_df, row_df, train_log = load_runs(args.include_smoke, args.include_pilot)
    agg = aggregate(summary)
    gaps = add_support_gaps(agg)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    agg.to_csv(ANALYSIS / "aggregate_summary.csv", index=False)
    gaps.to_csv(ANALYSIS / "support_gaps.csv", index=False)
    summary.to_csv(ANALYSIS / "aggregate_run_summaries.csv", index=False)
    task_df.to_csv(ANALYSIS / "aggregate_task_metrics.csv", index=False)
    row_df.to_csv(ANALYSIS / "aggregate_row_predictions.csv", index=False)
    if len(train_log):
        train_log.to_csv(ANALYSIS / "aggregate_training_log.csv", index=False)
    plot_all(agg, summary, task_df, train_log)
    write_report(agg, summary, task_df, row_df, train_log)
    print(agg.to_string(index=False))
    print(f"Report: {REPORTS / (EXP_NAME + '_report.md')}")


if __name__ == "__main__":
    main()

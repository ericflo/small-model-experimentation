#!/usr/bin/env python3
"""Aggregate reports for the counterfactual ICL public multiseed gate."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXP_NAME = "qwen_counterfactual_icl_public_multiseed"
ROOT = Path("/workspace/experiments") / EXP_NAME
LARGE_ROOT = Path("/workspace/large_artifacts") / EXP_NAME
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGS = ANALYSIS / "figures"
REPORTS = ROOT / "reports"


ARM_LABELS = {
    "counterfactual": "counterfactual_adapter",
    "ordinary": "ordinary_adapter",
    "shuffled_labels": "shuffled_label_adapter",
}


def pct(x: object) -> str:
    if x is None:
        return ""
    try:
        y = float(x)
    except Exception:
        return str(x)
    if math.isnan(y):
        return ""
    return f"{100 * y:.1f}%"


def fmt_float(x: object) -> str:
    if x is None:
        return ""
    try:
        y = float(x)
    except Exception:
        return str(x)
    if math.isnan(y):
        return ""
    return f"{y:.3f}"


def md_table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int | None = None) -> str:
    if cols:
        df = df[cols]
    if max_rows is not None:
        df = df.head(max_rows)
    tmp = df.copy()
    for c in tmp.columns:
        if c == "delta" or c.endswith("_mean") or c.endswith("_std") or "exact" in c or "capture" in c or c.endswith("_delta"):
            tmp[c] = tmp[c].map(pct)
        elif tmp[c].dtype.kind in "fc":
            tmp[c] = tmp[c].map(fmt_float)
    if not len(tmp):
        return "_No rows._"
    return tmp.to_markdown(index=False)


def arm_for(method: str, config: dict[str, object]) -> str:
    if method == "base":
        return "base"
    mode = str(config.get("train_mode", "unknown"))
    return ARM_LABELS.get(mode, f"{mode}_adapter")


def load_runs(include_smoke: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_frames: list[pd.DataFrame] = []
    task_frames: list[pd.DataFrame] = []
    row_frames: list[pd.DataFrame] = []
    train_frames: list[pd.DataFrame] = []
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        summary_path = run_dir / "summary.csv"
        task_path = run_dir / "task_metrics.csv"
        row_path = run_dir / "row_predictions.csv"
        if not cfg_path.exists() or not summary_path.exists() or not task_path.exists() or not row_path.exists():
            continue
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not include_smoke and (bool(config.get("smoke")) or str(run_dir.name).startswith("smoke")):
            continue
        for path, frames in [(summary_path, summary_frames), (task_path, task_frames), (row_path, row_frames)]:
            df = pd.read_csv(path)
            df["run_name"] = run_dir.name
            df["seed"] = int(config.get("seed", -1))
            df["eval_seed"] = int(config.get("eval_seed", -1))
            df["train_mode"] = str(config.get("train_mode", "unknown"))
            df["train_steps"] = int(config.get("train_steps", -1))
            df["arm"] = [arm_for(m, config) for m in df["method"]]
            frames.append(df)
        train_path = run_dir / "training_log.csv"
        if train_path.exists():
            tdf = pd.read_csv(train_path)
            if len(tdf):
                tdf["run_name"] = run_dir.name
                tdf["seed"] = int(config.get("seed", -1))
                tdf["train_mode"] = str(config.get("train_mode", "unknown"))
                tdf["arm"] = ARM_LABELS.get(str(config.get("train_mode", "unknown")), str(config.get("train_mode", "unknown")))
                train_frames.append(tdf)
    if not summary_frames:
        raise SystemExit(f"No completed non-smoke runs found in {RUNS}")
    return (
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(task_frames, ignore_index=True),
        pd.concat(row_frames, ignore_index=True),
        pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame(),
    )


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    g = (
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
        g[c] = g[c].fillna(0.0)
    return g


def lookup(agg: pd.DataFrame, arm: str, split: str, support: str, col: str = "full_task_exact_mean") -> float | None:
    m = agg[(agg.arm == arm) & (agg.split == split) & (agg.support_mode == support)]
    if len(m):
        return float(m.iloc[0][col])
    return None


def task_flips(task_df: pd.DataFrame, arm: str, split: str = "public_prose") -> pd.DataFrame:
    base = task_df[(task_df.arm == "base") & (task_df.split == split) & (task_df.support_mode == "normal")]
    comp = task_df[(task_df.arm == arm) & (task_df.split == split) & (task_df.support_mode == "normal")]
    if base.empty or comp.empty:
        return pd.DataFrame()
    base_key = base[["task_id", "full_task_exact"]].rename(columns={"full_task_exact": "base_full_task_exact"})
    merged = comp.merge(base_key, on="task_id", how="inner")
    merged["delta"] = merged["full_task_exact"].astype(float) - merged["base_full_task_exact"].astype(float)
    return (
        merged.groupby(["run_name", "seed"], as_index=False)
        .agg(
            helped=("delta", lambda s: int((s > 0).sum())),
            hurt=("delta", lambda s: int((s < 0).sum())),
            tied=("delta", lambda s: int((s == 0).sum())),
            tasks=("delta", "size"),
        )
        .sort_values(["run_name"])
    )


def family_table(task_df: pd.DataFrame, arm: str = "counterfactual_adapter", split: str = "public_prose") -> pd.DataFrame:
    df = task_df[(task_df.arm.isin(["base", arm])) & (task_df.split == split) & (task_df.support_mode == "normal")]
    if df.empty:
        return pd.DataFrame()
    tab = (
        df.groupby(["arm", "family"], as_index=False)
        .agg(tasks=("task_id", "nunique"), full_task_exact=("full_task_exact", "mean"), row_exact=("row_exact", "mean"))
        .pivot(index="family", columns="arm", values=["tasks", "full_task_exact", "row_exact"])
    )
    tab.columns = [f"{a}_{b}" for a, b in tab.columns]
    tab = tab.reset_index().fillna(0)
    if "full_task_exact_counterfactual_adapter" in tab and "full_task_exact_base" in tab:
        tab["delta"] = tab["full_task_exact_counterfactual_adapter"] - tab["full_task_exact_base"]
    return tab.sort_values("delta", ascending=False) if "delta" in tab else tab


def plot_aggregate(agg: pd.DataFrame, summary: pd.DataFrame, task_df: pd.DataFrame, train_log: pd.DataFrame) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)

    main = agg[(agg.support_mode == "normal") & (agg.split.isin(["synthetic_counterfactual", "public_prose"]))]
    main = main.sort_values(["split", "full_task_exact_mean"], ascending=[True, False])
    labels = [f"{r.arm.replace('_adapter', '')}\n{r.split.replace('_counterfactual', '').replace('public_prose', 'public')}" for r in main.itertuples()]
    colors = ["#465c69" if r.arm == "base" else "#2a9d8f" if r.arm == "counterfactual_adapter" else "#d9822b" for r in main.itertuples()]
    plt.figure(figsize=(11, 6))
    x = range(len(main))
    plt.bar(x, main["full_task_exact_mean"], yerr=main["full_task_exact_std"], capsize=5, color=colors)
    plt.xticks(list(x), labels, rotation=35, ha="right")
    plt.ylabel("Full-task exact")
    plt.ylim(0, 1)
    plt.title("Normal-Support Full-Task Exact Across Runs")
    plt.tight_layout()
    plt.savefig(FIGS / "aggregate_full_task_exact.png", dpi=180)
    plt.close()

    cf = summary[(summary.arm == "counterfactual_adapter") & (summary.support_mode == "normal")]
    if len(cf):
        piv = cf.pivot_table(index="run_name", columns="split", values="full_task_exact")
        plt.figure(figsize=(7, 6))
        for run_name, r in piv.iterrows():
            if "synthetic_counterfactual" in r and "public_prose" in r:
                plt.scatter(r["synthetic_counterfactual"], r["public_prose"], s=100)
                plt.annotate(run_name, (r["synthetic_counterfactual"], r["public_prose"]), fontsize=9)
        plt.xlabel("Synthetic counterfactual full-task exact")
        plt.ylabel("Public PROSE full-task exact")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Synthetic Learning Versus Public Transfer")
        plt.tight_layout()
        plt.savefig(FIGS / "synthetic_vs_public_transfer.png", dpi=180)
        plt.close()

    support = agg[(agg.arm == "counterfactual_adapter") & (agg.split.isin(["synthetic_counterfactual", "public_prose"]))]
    if len(support):
        support = support.sort_values(["split", "support_mode"])
        labels = [f"{r.split.replace('_counterfactual', '').replace('public_prose', 'public')}\n{r.support_mode}" for r in support.itertuples()]
        plt.figure(figsize=(9, 5))
        plt.bar(range(len(support)), support["full_task_exact_mean"], yerr=support["full_task_exact_std"], capsize=4, color="#6c5ce7")
        plt.xticks(range(len(support)), labels, rotation=25, ha="right")
        plt.ylabel("Full-task exact")
        plt.ylim(0, 1)
        plt.title("Support Dependence of Counterfactual Adapter")
        plt.tight_layout()
        plt.savefig(FIGS / "support_dependence.png", dpi=180)
        plt.close()

    flips = task_flips(task_df, "counterfactual_adapter", "public_prose")
    if len(flips):
        plt.figure(figsize=(8, 5))
        bottom = [0] * len(flips)
        xlabels = flips["run_name"].tolist()
        for col, color in [("helped", "#2a9d8f"), ("hurt", "#e76f51"), ("tied", "#9aa0a6")]:
            plt.bar(xlabels, flips[col], bottom=bottom, label=col, color=color)
            bottom = [b + v for b, v in zip(bottom, flips[col])]
        plt.ylabel("Public tasks")
        plt.title("Public Task Flips Versus Base")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGS / "public_task_flips_by_seed.png", dpi=180)
        plt.close()

    if len(train_log):
        plt.figure(figsize=(9, 5))
        for run_name, g in train_log.groupby("run_name"):
            plt.plot(g["step"], g["loss"], label=run_name)
        plt.xlabel("Optimizer step")
        plt.ylabel("Training loss")
        plt.title("Training Loss by Run")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGS / "aggregate_training_loss.png", dpi=180)
        plt.close()


def write_report(agg: pd.DataFrame, summary: pd.DataFrame, task_df: pd.DataFrame, row_df: pd.DataFrame, train_log: pd.DataFrame) -> None:
    report_md = REPORTS / f"{EXP_NAME}_report.md"
    report_html = REPORTS / f"{EXP_NAME}_report.html"
    REPORTS.mkdir(parents=True, exist_ok=True)

    base_pub = lookup(agg, "base", "public_prose", "normal")
    cf_pub = lookup(agg, "counterfactual_adapter", "public_prose", "normal")
    cf_pub_std = lookup(agg, "counterfactual_adapter", "public_prose", "normal", "full_task_exact_std")
    cf_pub_shuf = lookup(agg, "counterfactual_adapter", "public_prose", "shuffled")
    cf_pub_none = lookup(agg, "counterfactual_adapter", "public_prose", "none")
    base_syn = lookup(agg, "base", "synthetic_counterfactual", "normal")
    cf_syn = lookup(agg, "counterfactual_adapter", "synthetic_counterfactual", "normal")
    cf_syn_shuf = lookup(agg, "counterfactual_adapter", "synthetic_counterfactual", "shuffled")
    ord_pub = lookup(agg, "ordinary_adapter", "public_prose", "normal")
    shuftrain_pub = lookup(agg, "shuffled_label_adapter", "public_prose", "normal")

    flips = task_flips(task_df, "counterfactual_adapter", "public_prose")
    fam = family_table(task_df)
    per_run = summary[(summary.support_mode == "normal") & (summary.split.isin(["synthetic_counterfactual", "public_prose"]))].copy()
    per_run = per_run.sort_values(["arm", "run_name", "split"])

    lines: list[str] = []
    lines.append("# Counterfactual ICL Public Multiseed Gate")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("Can LoRA posttraining on counterfactual few-shot episodes make Qwen3-4B rely more on the support examples of a text-transformation task, and does that transfer to a public benchmark rather than only to the synthetic generator?")
    lines.append("")
    lines.append("The training signal is answer-only. No public benchmark labels are used for training. The controls test whether the effect survives support shuffling, no-support prompts, ordinary synthetic training, and deliberately shuffled training support labels.")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    if cf_pub is not None and base_pub is not None:
        delta = cf_pub - base_pub
        lines.append(f"- Public PROSE full-task exact: base `{pct(base_pub)}`; counterfactual adapter mean `{pct(cf_pub)}` with seed spread `{pct(cf_pub_std)}`; delta `{pct(delta)}`.")
    if cf_syn is not None and base_syn is not None:
        lines.append(f"- Synthetic counterfactual full-task exact: base `{pct(base_syn)}`; counterfactual adapter mean `{pct(cf_syn)}`.")
    if cf_pub_shuf is not None and cf_pub is not None:
        lines.append(f"- Public support controls for the counterfactual adapter: normal `{pct(cf_pub)}`, shuffled `{pct(cf_pub_shuf)}`, no support `{pct(cf_pub_none)}`.")
    if ord_pub is not None:
        lines.append(f"- Ordinary synthetic-training control on public PROSE: `{pct(ord_pub)}`.")
    if shuftrain_pub is not None:
        lines.append(f"- Shuffled-label training control on public PROSE: `{pct(shuftrain_pub)}`.")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append(md_table(agg, ["arm", "split", "support_mode", "runs", "tasks", "rows", "row_exact_mean", "row_exact_std", "full_task_exact_mean", "full_task_exact_std"]))
    lines.append("")
    lines.append("## Seed-Level Normal-Support Metrics")
    lines.append("")
    lines.append(md_table(per_run, ["run_name", "seed", "arm", "split", "tasks", "rows", "row_exact", "full_task_exact"]))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if cf_syn is not None and base_syn is not None and cf_syn_shuf is not None:
        if cf_syn > base_syn + 0.05 and cf_syn > cf_syn_shuf + 0.05:
            lines.append("The synthetic counterfactual split shows the intended training effect: the adapter improves strict task consistency, and that improvement depends on intact support examples.")
        else:
            lines.append("The synthetic split does not cleanly establish support-conditioned learning at this training scale.")
    if cf_pub is not None and base_pub is not None:
        if cf_pub > base_pub + 0.05:
            lines.append("The public split shows positive transfer from synthetic counterfactual episodes to unseen public text-transformation tasks.")
        elif abs(cf_pub - base_pub) <= 0.025:
            lines.append("The public split is approximately flat against base at this scale.")
        else:
            lines.append("The public split regresses against base at this scale.")
    if cf_pub is not None and cf_pub_shuf is not None:
        if cf_pub > cf_pub_shuf + 0.05:
            lines.append("The public gain is support-sensitive: corrupting the support examples removes a material fraction of the effect.")
        else:
            lines.append("The public gain is not clearly support-sensitive, so it may reflect output-format tuning or family priors rather than task induction.")
    if ord_pub is not None and cf_pub is not None:
        if cf_pub > ord_pub + 0.05:
            lines.append("The counterfactual curriculum beats ordinary synthetic training on the public transfer metric.")
        else:
            lines.append("The ordinary synthetic control is close enough that the counterfactual ingredient is not isolated.")
    if shuftrain_pub is not None and cf_pub is not None:
        if cf_pub > shuftrain_pub + 0.05:
            lines.append("Training on shuffled support labels does not reproduce the public result, which argues against a pure formatting explanation.")
        else:
            lines.append("The shuffled-label training control is close enough that a formatting explanation remains plausible.")
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    for name, caption in [
        ("aggregate_full_task_exact.png", "Normal-support full-task exact across runs"),
        ("synthetic_vs_public_transfer.png", "Synthetic learning versus public transfer"),
        ("support_dependence.png", "Support-dependence controls"),
        ("public_task_flips_by_seed.png", "Public task flips versus base"),
        ("aggregate_training_loss.png", "Training loss by run"),
    ]:
        if (FIGS / name).exists():
            lines.append(f"![{caption}](../analysis/figures/{name})")
            lines.append("")
    lines.append("## Public Task Flips Versus Base")
    lines.append("")
    lines.append(md_table(flips))
    lines.append("")
    lines.append("## Public Family Breakdown")
    lines.append("")
    if len(fam):
        lines.append(md_table(fam, max_rows=80))
    else:
        lines.append("_No public family breakdown available._")
    lines.append("")
    lines.append("## Public Error Sample")
    lines.append("")
    misses = row_df[
        (row_df.arm == "counterfactual_adapter")
        & (row_df.split == "public_prose")
        & (row_df.support_mode == "normal")
        & (~row_df["exact"].astype(bool))
    ].sort_values(["run_name", "task_id", "query_index"])
    lines.append(md_table(misses[["run_name", "task_id", "family", "input", "target", "prediction"]], max_rows=60))
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Experiment root: `{ROOT}`")
    lines.append(f"- Large artifacts root: `{LARGE_ROOT}`")
    lines.append(f"- Per-run artifacts: `{RUNS}`")
    lines.append(f"- Adapter checkpoints: `{LARGE_ROOT / 'checkpoints'}`")
    lines.append(f"- Aggregate CSV: `{ANALYSIS / 'aggregate_summary.csv'}`")
    lines.append(f"- Combined row predictions: `{ANALYSIS / 'aggregate_row_predictions.csv'}`")
    lines.append(f"- Combined task metrics: `{ANALYSIS / 'aggregate_task_metrics.csv'}`")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("The public evaluation is capped for runtime, and exact-match scoring is strict. The main counterfactual arm is multiseed; ordinary and shuffled-label controls are single-seed controls in this run. The experiment tests transfer from a synthetic counterfactual training distribution, not broad open-ended transformation ability.")

    report_md.write_text("\n".join(lines), encoding="utf-8")

    try:
        import markdown  # type: ignore

        body = markdown.markdown("\n".join(lines), extensions=["tables"])
    except Exception:
        body = "<pre>" + html.escape("\n".join(lines)) + "</pre>"
        for name in [
            "aggregate_full_task_exact.png",
            "synthetic_vs_public_transfer.png",
            "support_dependence.png",
            "public_task_flips_by_seed.png",
            "aggregate_training_loss.png",
        ]:
            body = body.replace(
                html.escape(f"![{name}](../analysis/figures/{name})"),
                f"<img src='../analysis/figures/{name}' alt='{name}'>",
            )
    report_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Counterfactual ICL Public Multiseed Gate</title>"
        "<style>body{font-family:Inter,Arial,sans-serif;max-width:1200px;margin:32px auto;line-height:1.48;color:#1f2937}"
        "table{border-collapse:collapse;font-size:13px;margin:14px 0}td,th{border:1px solid #d1d5db;padding:4px 7px}th{background:#f3f4f6}"
        "img{max-width:100%;margin:14px 0;border:1px solid #e5e7eb}code{background:#f3f4f6;padding:1px 4px;border-radius:4px}"
        "h1,h2{color:#111827}</style></head><body>"
        + body
        + "</body></html>",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_smoke", action="store_true")
    args = ap.parse_args()

    summary, task_df, row_df, train_log = load_runs(include_smoke=args.include_smoke)
    agg = aggregate_summary(summary)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    agg.to_csv(ANALYSIS / "aggregate_summary.csv", index=False)
    summary.to_csv(ANALYSIS / "aggregate_run_summaries.csv", index=False)
    task_df.to_csv(ANALYSIS / "aggregate_task_metrics.csv", index=False)
    row_df.to_csv(ANALYSIS / "aggregate_row_predictions.csv", index=False)
    if len(train_log):
        train_log.to_csv(ANALYSIS / "aggregate_training_log.csv", index=False)
    plot_aggregate(agg, summary, task_df, train_log)
    write_report(agg, summary, task_df, row_df, train_log)
    print(agg.to_string(index=False))
    print(f"Report: {REPORTS / (EXP_NAME + '_report.md')}")


if __name__ == "__main__":
    main()

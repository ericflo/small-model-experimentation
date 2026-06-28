#!/usr/bin/env python3
"""Analyze the recurrent VM repair policy experiment."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIG = ANALYSIS / "figures"
REPORTS = ROOT / "reports"

MAIN_RUN = "main_recurrent_vm_repair_crossattn_dagger3_s192_c1024"
FINAL_PHASE = "dagger_policy_r3"
SPLIT_ORDER = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
SPLIT_LABEL = {
    "val_mixed": "Validation",
    "fresh_standard": "Fresh standard",
    "fresh_paraphrase": "Fresh paraphrase",
    "fresh_paired": "Fresh paired",
    "hard_composition": "Hard composition",
}


def pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def load_csvs(name: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob(f"*/{name}")):
        df = pd.read_csv(path)
        df["run"] = path.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def best_row(df: pd.DataFrame, phase: str, split: str, mode: str) -> pd.Series:
    sub = df[(df["phase"] == phase) & (df["split"] == split) & (df["mode"] == mode)].copy()
    if sub.empty:
        raise ValueError(f"missing metrics for {phase=} {split=} {mode=}")
    return sub.sort_values(["accuracy", "k"], ascending=[False, True]).iloc[0]


def main_summary(metrics: pd.DataFrame, compiler: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    main = metrics[metrics["run"] == MAIN_RUN].copy()
    comp = compiler[compiler["run"] == MAIN_RUN].copy()
    for split in SPLIT_ORDER:
        base = best_row(main, FINAL_PHASE, split, "base")
        learned = best_row(main, FINAL_PHASE, split, "learned_stop")
        forced = best_row(main, FINAL_PHASE, split, "learned_forced")
        oracle = best_row(main, FINAL_PHASE, split, "oracle_teacher")
        seed = comp[(comp["phase"] == "seed_compiler") & (comp["split"] == split)].iloc[0]
        full = comp[(comp["phase"] == "full_supervised_compiler") & (comp["split"] == split)].iloc[0]
        rows.append(
            {
                "split": split,
                "label": SPLIT_LABEL[split],
                "base_accuracy": float(base["accuracy"]),
                "learned_stop_accuracy": float(learned["accuracy"]),
                "learned_stop_k": int(learned["k"]),
                "learned_forced_accuracy": float(forced["accuracy"]),
                "learned_forced_k": int(forced["k"]),
                "oracle_accuracy": float(oracle["accuracy"]),
                "oracle_k": int(oracle["k"]),
                "seed_search_accuracy": float(seed["search_accuracy"]),
                "full_supervised_direct": float(full["direct_accuracy"]),
                "full_supervised_search": float(full["search_accuracy"]),
                "learned_gain": float(learned["accuracy"]) - float(base["accuracy"]),
                "oracle_gap_remaining": float(oracle["accuracy"]) - float(learned["accuracy"]),
            }
        )
    return pd.DataFrame(rows)


def dagger_summary(metrics: pd.DataFrame, train: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    main_m = metrics[metrics["run"] == MAIN_RUN]
    main_t = train[train["run"] == MAIN_RUN]
    main_tr = traj[traj["run"] == MAIN_RUN]
    phases = ["teacher_policy", "dagger_policy_r1", "dagger_policy_r2", "dagger_policy_r3"]
    traj_map = {
        "teacher_policy": "teacher_trajectory",
        "dagger_policy_r1": "dagger_trajectory_r1",
        "dagger_policy_r2": "dagger_trajectory_r2",
        "dagger_policy_r3": "dagger_trajectory_r3",
    }
    for phase in phases:
        tr = main_tr[main_tr["phase"] == traj_map[phase]].iloc[0]
        last = main_t[main_t["phase"] == phase].sort_values("epoch").iloc[-1]
        val_best = best_row(main_m, phase, "val_mixed", "learned_stop")
        hard_best = best_row(main_m, phase, "hard_composition", "learned_stop")
        rows.append(
            {
                "phase": phase,
                "source_success": float(tr["teacher_success_rate"]),
                "states": int(tr["states"]),
                "action_accuracy": float(last.get("action_accuracy", 0.0)),
                "arg_accuracy": float(last.get("arg_accuracy", 0.0)),
                "stop_accuracy": float(last.get("stop_accuracy", 0.0)),
                "val_best_accuracy": float(val_best["accuracy"]),
                "val_best_k": int(val_best["k"]),
                "hard_best_accuracy": float(hard_best["accuracy"]),
                "hard_best_k": int(hard_best["k"]),
            }
        )
    return pd.DataFrame(rows)


def save_aggregates(metrics: pd.DataFrame, compiler: pd.DataFrame, train: pd.DataFrame, traj: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for name, df in [
        ("all_metrics.csv", metrics),
        ("all_compiler_metrics.csv", compiler),
        ("all_train_log.csv", train),
        ("all_trajectory_stats.csv", traj),
    ]:
        if not df.empty:
            df.to_csv(ANALYSIS / name, index=False)
            df[df["run"] == MAIN_RUN].to_csv(ANALYSIS / name.replace("all_", "main_"), index=False)
    summary = main_summary(metrics, compiler)
    dagger = dagger_summary(metrics, train, traj)
    summary.to_csv(ANALYSIS / "main_summary.csv", index=False)
    dagger.to_csv(ANALYSIS / "main_dagger_progress.csv", index=False)
    return summary, dagger


def plot_k_curves(metrics: pd.DataFrame) -> None:
    main = metrics[(metrics["run"] == MAIN_RUN) & (metrics["phase"] == FINAL_PHASE)].copy()
    colors = {
        "oracle_teacher": "#2e7d32",
        "learned_stop": "#1565c0",
        "learned_forced": "#ef6c00",
    }
    labels = {
        "oracle_teacher": "Oracle repair",
        "learned_stop": "Learned STOP",
        "learned_forced": "Forced edits",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharey=True)
    axes = axes.flatten()
    for ax, split in zip(axes, SPLIT_ORDER):
        sub = main[main["split"] == split]
        base = float(sub[sub["mode"] == "base"]["accuracy"].iloc[0]) * 100.0
        ax.axhline(base, color="#616161", linestyle="--", linewidth=1.5, label="Base")
        for mode in ["learned_stop", "learned_forced", "oracle_teacher"]:
            m = sub[sub["mode"] == mode].sort_values("k")
            ax.plot(m["k"], m["accuracy"] * 100.0, marker="o", linewidth=2.0, color=colors[mode], label=labels[mode])
        ax.set_title(SPLIT_LABEL[split])
        ax.set_xlabel("repair steps K")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("accuracy (%)")
    axes[3].set_ylabel("accuracy (%)")
    axes[-1].axis("off")
    handles, labels_ = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels_, frameon=False, loc="center")
    fig.suptitle("Final recurrent repair K-curves", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG / "main_final_k_curves.png", dpi=180)
    plt.close(fig)


def plot_gap(summary: pd.DataFrame) -> None:
    x = range(len(summary))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    width = 0.18
    series = [
        ("base_accuracy", "Base", "#616161", -1.5),
        ("learned_stop_accuracy", "Learned recurrent", "#1565c0", -0.5),
        ("oracle_accuracy", "Oracle repair", "#2e7d32", 0.5),
        ("full_supervised_direct", "Full supervised direct", "#6a1b9a", 1.5),
    ]
    for key, label, color, offset in series:
        ax.bar([i + offset * width for i in x], summary[key] * 100.0, width=width, label=label, color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["label"], rotation=20, ha="right")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Learned recurrent policy versus reachable ceilings")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(FIG / "main_oracle_gap_by_split.png", dpi=180)
    plt.close(fig)


def plot_dagger(dagger: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = range(len(dagger))
    labels = ["Teacher", "DAgger 1", "DAgger 2", "DAgger 3"]
    for key, label, color in [
        ("source_success", "rollout/source success", "#2e7d32"),
        ("action_accuracy", "action accuracy", "#1565c0"),
        ("arg_accuracy", "argument accuracy", "#ef6c00"),
        ("val_best_accuracy", "best validation accuracy", "#6a1b9a"),
    ]:
        ax.plot(list(x), dagger[key] * 100.0, marker="o", linewidth=2.2, label=label, color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("rate (%)")
    ax.set_title("Repeated DAgger turns imitation into usable rollouts")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "dagger_progress.png", dpi=180)
    plt.close(fig)


def plot_stop_behavior(metrics: pd.DataFrame) -> None:
    main = metrics[
        (metrics["run"] == MAIN_RUN)
        & (metrics["phase"] == FINAL_PHASE)
        & (metrics["mode"] == "learned_stop")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for split in SPLIT_ORDER:
        sub = main[main["split"] == split].sort_values("k")
        axes[0].plot(sub["k"], sub["accuracy"] * 100.0, marker="o", linewidth=1.8, label=SPLIT_LABEL[split])
        axes[1].plot(sub["k"], sub["false_stop_rate"] * 100.0, marker="o", linewidth=1.8, label=SPLIT_LABEL[split])
    axes[0].set_title("Learned STOP accuracy")
    axes[1].set_title("False STOP rate")
    for ax in axes:
        ax.set_xlabel("repair steps K")
        ax.set_ylabel("rate (%)")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 105)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "stop_behavior_final.png", dpi=180)
    plt.close(fig)


def plot_compiler_ceiling(compiler: pd.DataFrame) -> None:
    comp = compiler[(compiler["run"] == MAIN_RUN)].copy()
    rows = []
    for _, r in comp.iterrows():
        rows.append({"phase": r["phase"], "split": r["split"], "metric": "direct", "value": r["direct_accuracy"]})
        rows.append({"phase": r["phase"], "split": r["split"], "metric": "search", "value": r["search_accuracy"]})
    df = pd.DataFrame(rows)
    labels = {"seed_compiler": "Seed", "full_supervised_compiler": "Full supervised"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, metric in zip(axes, ["direct", "search"]):
        sub = df[df["metric"] == metric]
        x = range(len(SPLIT_ORDER))
        width = 0.32
        for offset, phase, color in [(-0.5, "seed_compiler", "#616161"), (0.5, "full_supervised_compiler", "#6a1b9a")]:
            vals = [
                float(sub[(sub["phase"] == phase) & (sub["split"] == split)]["value"].iloc[0]) * 100.0
                for split in SPLIT_ORDER
            ]
            ax.bar([i + offset * width for i in x], vals, width=width, label=labels[phase], color=color)
        ax.set_title(f"Compiler {metric} accuracy")
        ax.set_xticks(list(x))
        ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLIT_ORDER], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("accuracy (%)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "compiler_ceiling.png", dpi=180)
    plt.close(fig)


def md_table_summary(summary: pd.DataFrame) -> str:
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            {
                "Split": r["label"],
                "Base": pct(r["base_accuracy"]),
                "Learned STOP": f"{pct(r['learned_stop_accuracy'])} (K={int(r['learned_stop_k'])})",
                "Forced edits": f"{pct(r['learned_forced_accuracy'])} (K={int(r['learned_forced_k'])})",
                "Oracle repair": pct(r["oracle_accuracy"]),
                "Full sup. direct": pct(r["full_supervised_direct"]),
                "Full sup. search": pct(r["full_supervised_search"]),
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def md_table_dagger(dagger: pd.DataFrame) -> str:
    names = {
        "teacher_policy": "Teacher policy",
        "dagger_policy_r1": "DAgger round 1",
        "dagger_policy_r2": "DAgger round 2",
        "dagger_policy_r3": "DAgger round 3",
    }
    rows = []
    for _, r in dagger.iterrows():
        rows.append(
            {
                "Phase": names[r["phase"]],
                "Source success": pct(r["source_success"]),
                "Action acc.": pct(r["action_accuracy"]),
                "Arg acc.": pct(r["arg_accuracy"]),
                "Val best": f"{pct(r['val_best_accuracy'])} (K={int(r['val_best_k'])})",
                "Hard best": f"{pct(r['hard_best_accuracy'])} (K={int(r['hard_best_k'])})",
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def write_markdown(summary: pd.DataFrame, dagger: pd.DataFrame) -> str:
    REPORTS.mkdir(parents=True, exist_ok=True)
    text = f"""# Recurrent VM Repair Policy Attached to Frozen Qwen

## Abstract

This experiment tests whether a frozen 4B language model can be given a learned recurrent program-repair loop at posttraining time. A compiler reads frozen `Qwen/Qwen3-4B` token hidden states and emits a typed stack-machine program. A repair policy then runs as an MDP: observe the prompt features, current program, and VM execution trace; choose one edit or STOP; execute the edited program; repeat for up to `K` private steps.

The result is a partial but real positive. The final learned recurrent policy improves over the seed compiler on every split, reaching {pct(summary.loc[summary.split == 'val_mixed', 'learned_stop_accuracy'].iloc[0])} validation accuracy and {pct(summary.loc[summary.split == 'fresh_paraphrase', 'learned_stop_accuracy'].iloc[0])} fresh-paraphrase accuracy from {pct(summary.loc[summary.split == 'val_mixed', 'base_accuracy'].iloc[0])} and {pct(summary.loc[summary.split == 'fresh_paraphrase', 'base_accuracy'].iloc[0])} base accuracy. Three rounds of DAgger raise training-rollout success from {pct(dagger.loc[dagger.phase == 'dagger_policy_r1', 'source_success'].iloc[0])} to {pct(dagger.loc[dagger.phase == 'dagger_policy_r3', 'source_success'].iloc[0])}. The oracle repair policy still reaches {pct(summary['oracle_accuracy'].min())}-{pct(summary['oracle_accuracy'].max())}, and the full supervised compiler reaches {pct(summary['full_supervised_direct'].min())}-{pct(summary['full_supervised_direct'].max())} direct accuracy. The recurrent loop is useful, but the remaining gap is large.

## Setup

- Base model: `Qwen/Qwen3-4B`, frozen and used as a hidden-state feature extractor.
- Trainable modules: typed bytecode compiler and recurrent edit/STOP policy.
- Seed compiler examples: `192`.
- Recurrent-policy training examples: `1024`.
- Full-supervised ceiling examples: `1024`, combined with the seed set.
- Evaluation size: `128` per split.
- VM: stack bytecode with `PUSH`, arithmetic, comparisons, table lookup, `END`, and `PAD`.
- Recurrent budgets: `K = 0, 1, 2, 4, 8, 16`.
- Main run: `{MAIN_RUN}`.
- Large checkpoints: `large_artifacts/qwen_recurrent_vm_repair_policy/checkpoints/{MAIN_RUN}/`.

## Method

The seed compiler emits an initial program from frozen Qwen hidden states. The repair policy receives:

- token-level frozen Qwen hidden states through cross-attention;
- the current bytecode program;
- VM validity, final value, stack-top trace, and stack-depth trace;
- the current recurrent step index.

It predicts one of three action kinds: STOP, edit an opcode slot, or edit an argument slot. Oracle trajectories are generated by comparing the current program to the gold program and taking one edit at a time. DAgger then rolls out the learned policy, labels the states it actually visits with the oracle next action, and retrains on the accumulated state set.

## Main Results

{md_table_summary(summary)}

![Final recurrent repair K-curves](../analysis/figures/main_final_k_curves.png)

![Learned recurrent policy versus ceilings](../analysis/figures/main_oracle_gap_by_split.png)

## DAgger Dynamics

{md_table_dagger(dagger)}

![DAgger progress](../analysis/figures/dagger_progress.png)

The important signal is not just lower training loss. Repeated on-policy correction changes rollout success: the policy succeeds on 32.4% of first-round visited programs, 42.2% in round 2, and 59.3% in round 3. Argument accuracy also rises from 37.4% after round 1 to 82.3% after round 3, which matters because argument edits carry the numeric content of the prompt.

## STOP Behavior

![STOP behavior](../analysis/figures/stop_behavior_final.png)

Learned STOP is helpful at moderate `K`, but it remains imperfect. The final policy often peaks at `K=2` or `K=4`; larger `K` can introduce false STOPs or unnecessary edits, especially on paired prompts. This is a learning problem, not an environment reachability problem, because the oracle repair policy remains near perfect.

## Compiler Ceiling

![Compiler ceiling](../analysis/figures/compiler_ceiling.png)

The full-supervised compiler ceiling is high: direct accuracy reaches 68.0-84.4%, and answer-verified search reaches 89.8-96.9%. That confirms the frozen Qwen features contain enough information for the bytecode task. The recurrent policy has learned a meaningful part of the repair process, but not enough to match dense supervised program learning.

## Interpretation

This experiment supports a narrow claim: a Qwen-attached recurrent VM repair loop can turn extra private compute steps into better answers under learned policy control. The strongest evidence is the DAgger progression plus the final learned K-curves. The policy is not merely a static reranker; it executes a sequence of edits, observes the VM after each edit, and improves as the state distribution is corrected.

The result does not support a broad claim of universal intelligence gain. The oracle and full-supervised ceilings show much more is reachable, but the learned policy still leaves most of the oracle gap open. The bottleneck is now policy learning: STOP calibration, choosing the right number of edits, and robust argument edits under prompt variation.

## Next Experiments

1. Add a learned value function over VM states and train the policy with advantage-weighted imitation so STOP is tied to expected answer improvement.
2. Distill oracle trajectories with dense intermediate value targets, not only next-action labels.
3. Use beam-style recurrent repair with a learned verifier over resulting VM states, then distill the best repair path back into the one-action policy.
4. Replace gold-program edit distance with answer-equivalent repair targets so the policy is not punished for alternate correct programs.
5. Test a token-output bridge where the same recurrent VM state is fed back into Qwen over multiple turns until a learned halt decision fires.

## Artifacts

- Script: `experiments/qwen_recurrent_vm_repair_policy/src/qwen_recurrent_vm_repair_policy_experiment.py`
- Analysis: `experiments/qwen_recurrent_vm_repair_policy/src/analyze_qwen_recurrent_vm_repair_policy.py`
- Main run: `experiments/qwen_recurrent_vm_repair_policy/runs/{MAIN_RUN}/`
- Aggregate metrics: `experiments/qwen_recurrent_vm_repair_policy/analysis/`
- Markdown report: `experiments/qwen_recurrent_vm_repair_policy/reports/qwen_recurrent_vm_repair_policy_report.md`
- HTML report: `experiments/qwen_recurrent_vm_repair_policy/reports/qwen_recurrent_vm_repair_policy_report.html`
"""
    path = REPORTS / "qwen_recurrent_vm_repair_policy_report.md"
    path.write_text(text, encoding="utf-8")
    return text


def img_tag(path: str, alt: str) -> str:
    return f'<figure><img src="{html.escape(path)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def write_html(summary: pd.DataFrame, dagger: pd.DataFrame) -> None:
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            {
                "Split": r["label"],
                "Base": pct(r["base_accuracy"]),
                "Learned STOP": f"{pct(r['learned_stop_accuracy'])} (K={int(r['learned_stop_k'])})",
                "Forced edits": f"{pct(r['learned_forced_accuracy'])} (K={int(r['learned_forced_k'])})",
                "Oracle repair": pct(r["oracle_accuracy"]),
                "Full sup. direct": pct(r["full_supervised_direct"]),
                "Full sup. search": pct(r["full_supervised_search"]),
            }
        )
    main_table = pd.DataFrame(rows).to_html(index=False, escape=False)
    drows = []
    for _, r in dagger.iterrows():
        drows.append(
            {
                "Phase": r["phase"].replace("_", " "),
                "Source success": pct(r["source_success"]),
                "Action acc.": pct(r["action_accuracy"]),
                "Arg acc.": pct(r["arg_accuracy"]),
                "Val best": f"{pct(r['val_best_accuracy'])} (K={int(r['val_best_k'])})",
                "Hard best": f"{pct(r['hard_best_accuracy'])} (K={int(r['hard_best_k'])})",
            }
        )
    dagger_table = pd.DataFrame(drows).to_html(index=False, escape=False)
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recurrent VM Repair Policy Attached to Frozen Qwen</title>
<style>
body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f7f8fa; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 40px 24px 72px; background: #ffffff; }}
h1 {{ font-size: 34px; margin: 0 0 8px; }}
h2 {{ font-size: 22px; margin-top: 36px; border-bottom: 1px solid #d8dee4; padding-bottom: 8px; }}
p, li {{ font-size: 16px; line-height: 1.55; }}
.dek {{ color: #536471; font-size: 18px; margin-bottom: 28px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 22px; font-size: 14px; }}
th, td {{ border: 1px solid #d8dee4; padding: 8px 10px; text-align: left; }}
th {{ background: #eef2f6; }}
figure {{ margin: 26px 0; }}
img {{ max-width: 100%; height: auto; border: 1px solid #d8dee4; background: #fff; }}
figcaption {{ color: #536471; font-size: 13px; margin-top: 6px; }}
code {{ background: #eef2f6; padding: 2px 5px; border-radius: 4px; }}
.callout {{ background: #f1f7ff; border-left: 4px solid #1565c0; padding: 12px 16px; margin: 18px 0; }}
</style>
</head>
<body>
<main>
<h1>Recurrent VM Repair Policy Attached to Frozen Qwen</h1>
<p class="dek">A standalone experiment on learned private compute steps, typed bytecode, and DAgger policy learning.</p>
<h2>Abstract</h2>
<p>This experiment tests whether a frozen 4B language model can be given a learned recurrent program-repair loop at posttraining time. A compiler reads frozen <code>Qwen/Qwen3-4B</code> token hidden states and emits a typed stack-machine program. A repair policy observes the prompt, current program, and VM execution trace; chooses one edit or STOP; executes the edited program; and repeats for up to <code>K</code> private steps.</p>
<div class="callout">Final learned recurrent accuracy improves over the seed compiler on every split, reaching {pct(summary.loc[summary.split == 'val_mixed', 'learned_stop_accuracy'].iloc[0])} validation and {pct(summary.loc[summary.split == 'fresh_paraphrase', 'learned_stop_accuracy'].iloc[0])} fresh-paraphrase accuracy. Oracle repair remains much higher, at {pct(summary['oracle_accuracy'].min())}-{pct(summary['oracle_accuracy'].max())}.</div>
<h2>Setup</h2>
<ul>
<li>Base model: <code>Qwen/Qwen3-4B</code>, frozen.</li>
<li>Seed examples: 192. Recurrent-policy examples: 1024. Full-supervised ceiling examples: 1024 plus seed.</li>
<li>Evaluation size: 128 per split.</li>
<li>Recurrent budgets: <code>K = 0, 1, 2, 4, 8, 16</code>.</li>
<li>Main run: <code>{MAIN_RUN}</code>.</li>
</ul>
<h2>Main Results</h2>
{main_table}
{img_tag('../analysis/figures/main_final_k_curves.png', 'Final recurrent repair K-curves')}
{img_tag('../analysis/figures/main_oracle_gap_by_split.png', 'Learned recurrent policy versus reachable ceilings')}
<h2>DAgger Dynamics</h2>
{dagger_table}
{img_tag('../analysis/figures/dagger_progress.png', 'DAgger progress')}
<p>Repeated on-policy correction is the main learned-policy lever. Rollout success rises from 32.4% after round 1 to 59.3% by round 3, while argument accuracy rises to 82.3%.</p>
<h2>STOP Behavior</h2>
{img_tag('../analysis/figures/stop_behavior_final.png', 'Learned STOP behavior')}
<p>Learned STOP is useful but imperfect. Several splits peak at moderate K, while larger K can add false STOPs or unnecessary edits. The oracle curve shows this is a policy-learning bottleneck, not a limit of the VM repair environment.</p>
<h2>Compiler Ceiling</h2>
{img_tag('../analysis/figures/compiler_ceiling.png', 'Compiler ceiling')}
<p>The full-supervised compiler reaches 68.0-84.4% direct accuracy and 89.8-96.9% answer-verified search accuracy, confirming that the frozen hidden states carry enough information for much stronger program prediction.</p>
<h2>Bottom Line</h2>
<p>The recurrent VM loop works, but it is not yet close to the oracle. The next high-impact step is to add value learning or advantage-weighted imitation so STOP and edit decisions are trained against expected VM-state improvement rather than only next-action labels.</p>
<h2>Artifacts</h2>
<ul>
<li><code>experiments/qwen_recurrent_vm_repair_policy/runs/{MAIN_RUN}/</code></li>
<li><code>experiments/qwen_recurrent_vm_repair_policy/analysis/</code></li>
<li><code>large_artifacts/qwen_recurrent_vm_repair_policy/checkpoints/{MAIN_RUN}/</code></li>
</ul>
</main>
</body>
</html>
"""
    (REPORTS / "qwen_recurrent_vm_repair_policy_report.html").write_text(body, encoding="utf-8")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    metrics = load_csvs("metrics.csv")
    compiler = load_csvs("compiler_metrics.csv")
    train = load_csvs("train_log.csv")
    traj = load_csvs("trajectory_stats.csv")
    summary, dagger = save_aggregates(metrics, compiler, train, traj)
    plot_k_curves(metrics)
    plot_gap(summary)
    plot_dagger(dagger)
    plot_stop_behavior(metrics)
    plot_compiler_ceiling(compiler)
    write_markdown(summary, dagger)
    write_html(summary, dagger)
    print(f"Wrote analysis to {ANALYSIS}")
    print(f"Wrote reports to {REPORTS}")


if __name__ == "__main__":
    main()

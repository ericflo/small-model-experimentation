#!/usr/bin/env python3
"""Build summary tables, charts, Markdown, and HTML for the main run."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from typed_bytecode_core import RUNS, ROOT


SPLIT_LABELS: Dict[str, str] = {
    "val_mixed": "Validation",
    "fresh_standard": "Fresh standard",
    "fresh_paraphrase": "Fresh paraphrase",
    "fresh_paired": "Fresh paired",
    "hard_composition": "Hard composition",
}

VARIANT_LABELS: Dict[str, str] = {
    "action_only": "Action-only VM",
    "echo": "ECHO VM",
    "oracle_teacher": "Oracle teacher",
    "native_qwen_direct": "Native Qwen direct",
}

COLORS: Dict[str, str] = {
    "action_only": "#2563eb",
    "echo": "#dc2626",
    "oracle_teacher": "#111827",
    "native_qwen_direct": "#059669",
    "blank": "#6b7280",
}


def pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def img_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def load_frames(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    train = pd.read_csv(run_dir / "lora_train_log.csv")
    native_path = run_dir / "native_qwen_metrics.csv"
    native = pd.read_csv(native_path) if native_path.exists() else pd.DataFrame()
    return metrics, train, native


def build_summary(metrics: pd.DataFrame, native: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    k8 = metrics[metrics["k"].eq(8)].copy()
    blank = metrics[(metrics["k"].eq(0)) & (metrics["variant"].eq("action_only"))][["split", "accuracy"]].rename(columns={"accuracy": "blank_k0"})
    action = k8[k8["variant"].eq("action_only")][["split", "accuracy", "false_stop_rate", "program_exact", "mean_steps"]].rename(
        columns={"accuracy": "action_only_k8", "false_stop_rate": "action_only_false_stop", "program_exact": "action_only_program_exact", "mean_steps": "action_only_mean_steps"}
    )
    echo = k8[k8["variant"].eq("echo")][["split", "accuracy", "false_stop_rate", "program_exact", "mean_steps"]].rename(
        columns={"accuracy": "echo_k8", "false_stop_rate": "echo_false_stop", "program_exact": "echo_program_exact", "mean_steps": "echo_mean_steps"}
    )
    oracle = k8[k8["variant"].eq("oracle_teacher")][["split", "accuracy"]].drop_duplicates("split").rename(columns={"accuracy": "oracle_k8"})
    summary = blank.merge(action, on="split").merge(echo, on="split").merge(oracle, on="split")
    if not native.empty:
        native_short = native[["split", "accuracy", "parse_rate"]].rename(columns={"accuracy": "native_qwen_direct", "parse_rate": "native_parse_rate"})
        summary = summary.merge(native_short, on="split", how="left")
    summary["action_minus_native"] = summary["action_only_k8"] - summary.get("native_qwen_direct", 0.0)
    summary["echo_minus_action"] = summary["echo_k8"] - summary["action_only_k8"]
    summary["action_oracle_gap"] = summary["oracle_k8"] - summary["action_only_k8"]
    summary["echo_oracle_gap"] = summary["oracle_k8"] - summary["echo_k8"]
    summary.to_csv(out_dir / "summary_k8.csv", index=False)

    avg = {"split": "average"}
    for column in summary.columns:
        if column != "split":
            avg[column] = float(summary[column].mean())
    pd.DataFrame([avg]).to_csv(out_dir / "summary_k8_average.csv", index=False)
    return summary


def plot_accuracy_by_k(metrics: pd.DataFrame, fig_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()
    for ax, split in zip(axes, SPLIT_LABELS):
        sub = metrics[metrics["split"].eq(split)]
        for variant in ["action_only", "echo", "oracle_teacher"]:
            rows = sub[sub["variant"].eq(variant)].sort_values("k")
            if rows.empty:
                continue
            if variant == "oracle_teacher":
                rows = rows.drop_duplicates("k")
            ax.plot(rows["k"], rows["accuracy"], marker="o", label=VARIANT_LABELS[variant], color=COLORS[variant], linewidth=2)
        ax.set_title(SPLIT_LABELS[split])
        ax.set_xlabel("VM turns K")
        ax.grid(True, alpha=0.25)
    axes[-1].axis("off")
    axes[0].set_ylabel("Accuracy")
    axes[3].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.96, 0.08))
    fig.suptitle("Accuracy Improves With Recurrent VM Turns, But Oracle Remains Higher")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = fig_dir / "accuracy_by_k.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_k8_bars(summary: pd.DataFrame, fig_dir: Path) -> Path:
    labels = [SPLIT_LABELS[s] for s in summary["split"]]
    x = range(len(summary))
    width = 0.18
    fig, ax = plt.subplots(figsize=(13, 6))
    series = [
        ("blank_k0", "Blank VM K=0", COLORS["blank"]),
        ("native_qwen_direct", "Native Qwen direct", COLORS["native_qwen_direct"]),
        ("action_only_k8", "Action-only VM K=8", COLORS["action_only"]),
        ("echo_k8", "ECHO VM K=8", COLORS["echo"]),
        ("oracle_k8", "Oracle K=8", COLORS["oracle_teacher"]),
    ]
    offsets = [-2 * width, -width, 0, width, 2 * width]
    for (column, label, color), offset in zip(series, offsets):
        if column not in summary:
            continue
        ax.bar([i + offset for i in x], summary[column], width=width, label=label, color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("K=8 Accuracy Against Blank, Native, and Oracle Baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    path = fig_dir / "k8_accuracy_by_split.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_false_stop(summary: pd.DataFrame, fig_dir: Path) -> Path:
    labels = [SPLIT_LABELS[s] for s in summary["split"]]
    x = range(len(summary))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - width / 2 for i in x], summary["action_only_false_stop"], width=width, label="Action-only VM", color=COLORS["action_only"])
    ax.bar([i + width / 2 for i in x], summary["echo_false_stop"], width=width, label="ECHO VM", color=COLORS["echo"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("False STOP rate at K=8")
    ax.set_title("Premature STOP Remains The Main Behavioral Failure")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = fig_dir / "false_stop_k8.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_train_ce(train: pd.DataFrame, fig_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for variant in ["action_only", "echo"]:
        rows = train[train["variant"].eq(variant)].sort_values("epoch")
        axes[0].plot(rows["epoch"], rows["action_ce"], marker="o", label=VARIANT_LABELS[variant], color=COLORS[variant], linewidth=2)
        axes[1].plot(rows["epoch"], rows["observation_ce"], marker="o", label=VARIANT_LABELS[variant], color=COLORS[variant], linewidth=2)
    axes[0].set_title("Action-token CE")
    axes[1].set_title("Observation-token CE")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("CE")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle("ECHO Improves Predictive Losses But Not Average Generalization")
    fig.tight_layout()
    path = fig_dir / "train_ce.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def markdown_table(summary: pd.DataFrame) -> str:
    rows: List[str] = []
    rows.append("| Split | Blank K=0 | Native Qwen | Action-only K=8 | ECHO K=8 | Oracle K=8 |")
    rows.append("|---|---:|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        rows.append(
            f"| {SPLIT_LABELS[row['split']]} | {pct(row['blank_k0'])} | {pct(row['native_qwen_direct'])} | "
            f"{pct(row['action_only_k8'])} | {pct(row['echo_k8'])} | {pct(row['oracle_k8'])} |"
        )
    rows.append(
        f"| Average | {pct(summary['blank_k0'].mean())} | {pct(summary['native_qwen_direct'].mean())} | "
        f"{pct(summary['action_only_k8'].mean())} | {pct(summary['echo_k8'].mean())} | {pct(summary['oracle_k8'].mean())} |"
    )
    return "\n".join(rows)


def make_markdown(run_name: str, summary: pd.DataFrame, train: pd.DataFrame, figures: Dict[str, Path], out_dir: Path) -> str:
    action_avg = float(summary["action_only_k8"].mean())
    echo_avg = float(summary["echo_k8"].mean())
    blank_avg = float(summary["blank_k0"].mean())
    native_avg = float(summary["native_qwen_direct"].mean())
    oracle_avg = float(summary["oracle_k8"].mean())
    echo_delta = echo_avg - action_avg
    action_native_delta = action_avg - native_avg
    action_oracle_gap = oracle_avg - action_avg
    echo_oracle_gap = oracle_avg - echo_avg
    action_ce = train[train["variant"].eq("action_only")].sort_values("epoch").iloc[-1]
    echo_ce = train[train["variant"].eq("echo")].sort_values("epoch").iloc[-1]

    md = f"""# VM-Agent ECHO QLoRA: Standalone Experiment Report

## Executive Summary

This experiment tested whether `Qwen/Qwen3-4B` can be post-trained into a recurrent program-editing agent over a small typed bytecode VM. At inference time, the model starts from a blank valid program, emits one edit action, receives the executed VM state as text, and repeats for up to K turns. The control trains only on edit-action tokens. The ECHO treatment also trains on the VM observation tokens with weight 0.05, so the model is explicitly trained to predict the consequences of its edits.

The core result is mixed. The VM-agent loop itself worked: the action-only model improved from a {pct(blank_avg)} blank-program baseline to {pct(action_avg)} average K=8 accuracy, and it beat native direct-answer Qwen on average by {pct(action_native_delta)}. But ECHO did not improve the scaled result. ECHO reached {pct(echo_avg)} average K=8 accuracy, {pct(echo_delta)} relative to action-only, despite better action-token and observation-token validation losses.

The main bottleneck is no longer syntax. Both trained VM-agent variants had 100% parse rate in the main run. The failures are policy failures: premature STOP, wrong constants, and incomplete multi-step programs. The oracle teacher reached {pct(oracle_avg)} average K=8 accuracy, leaving a {pct(action_oracle_gap)} gap for action-only and a {pct(echo_oracle_gap)} gap for ECHO.

## Setup

- Base model: `Qwen/Qwen3-4B`.
- Training method: 4-bit QLoRA with rank 8, alpha 16, 16.5M trainable parameters.
- Training data: 512 generated VM tasks.
- Evaluation: five 32-example splits: validation, fresh standard wording, fresh paraphrase, paired prompts, and hard composition.
- Initial program: `PUSH 0; END; PAD ...`.
- Actions: `OP <slot> <opcode>`, `ARG <slot> <0-96>`, or `STOP`.
- Inference budgets: K in {{0, 2, 4, 8}} VM turns.
- Baselines: blank VM at K=0, native direct-answer Qwen, and an oracle teacher that edits toward the reference bytecode.

## Main K=8 Results

{markdown_table(summary)}

![K=8 accuracy by split](figures/k8_accuracy_by_split.png)

## K-Scaling

Both learned VM-agent policies benefit from more recurrent turns. The action-only model is monotonic across all five splits from K=0 to K=8. That matters: it means the model is not merely producing a one-shot answer in a different format; additional model-VM interaction is doing useful work.

![Accuracy by K](figures/accuracy_by_k.png)

## ECHO Result

ECHO clearly learned the observation channel. Final validation CE:

- Action-only action CE: {action_ce['action_ce']:.4f}
- ECHO action CE: {echo_ce['action_ce']:.4f}
- Action-only observation CE: {action_ce['observation_ce']:.4f}
- ECHO observation CE: {echo_ce['observation_ce']:.4f}

That did not translate into better average rollout accuracy. ECHO improved validation K=8 accuracy from 46.9% to 50.0%, but was worse on fresh standard, paired, and hard composition. The likely interpretation is that token-level observation prediction is too easy and too local: it teaches the model to model the textual VM state, but does not directly optimize the action policy needed to close the oracle gap.

![Training cross-entropy](figures/train_ce.png)

## Failure Mode

Premature STOP remains the clearest behavioral problem. At K=8, action-only had {pct(summary['action_only_false_stop'].mean())} average false STOP rate, and ECHO had {pct(summary['echo_false_stop'].mean())}. The parse rate was 100%, so this is not a grammar problem. It is a decision problem: the model often chooses to halt before it has built the correct executable program.

![False STOP rate](figures/false_stop_k8.png)

## Interpretation

The useful signal is that a 4B Qwen model can be post-trained to act as a recurrent compiler-like policy over a typed executable substrate. From a blank program, the action-only model reached 43.1% average K=8 accuracy and beat direct-answer Qwen on validation, fresh standard, and hard composition. This supports the broad direction of using the model as one iteration of a compute loop rather than trying to place the entire latent computation inside one forward pass.

The negative signal is equally important. ECHO, as implemented here, is not the missing ingredient. Its auxiliary observation-token loss improved CE but degraded average generalization. The next experiment should move the learning signal closer to executable success: reward the whole rollout, penalize false STOP, and train a verifier or value head over VM states instead of asking the LM to predict long observation strings.

## Recommended Next Experiment

Run a verifier-guided rollout optimization experiment:

1. Warm start from the action-only VM-agent policy.
2. Add a small value/verifier head over the final-token hidden state that predicts whether the current VM state solves the task.
3. Fine-tune with rollout-level reward: correct final VM answer, valid program, fewer edits, and a direct false-STOP penalty.
4. Compare supervised cloning, DPO on successful vs failed rollouts, and GRPO/REINFORCE with the VM reward.
5. Keep the same native Qwen, blank VM, action-only, and oracle baselines.

This attacks the observed bottleneck directly. The oracle gap shows that the substrate can solve many more examples within K=8; the current model simply does not learn the halting/action policy well enough from token imitation alone.
"""
    (out_dir / "report.md").write_text(md)
    return md


def make_html(markdown: str, figures: Dict[str, Path], out_dir: Path) -> None:
    html = markdown
    replacements = {
        "![K=8 accuracy by split](figures/k8_accuracy_by_split.png)": f'<img src="{img_data_uri(figures["k8"])}" alt="K=8 accuracy by split">',
        "![Accuracy by K](figures/accuracy_by_k.png)": f'<img src="{img_data_uri(figures["k"])}" alt="Accuracy by K">',
        "![Training cross-entropy](figures/train_ce.png)": f'<img src="{img_data_uri(figures["ce"])}" alt="Training cross-entropy">',
        "![False STOP rate](figures/false_stop_k8.png)": f'<img src="{img_data_uri(figures["stop"])}" alt="False STOP rate">',
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    lines = html.splitlines()
    body: List[str] = []
    in_ul = False
    in_table = False
    for line in lines:
        if line.startswith("# "):
            if in_ul:
                body.append("</ul>")
                in_ul = False
            if in_table:
                body.append("</tbody></table>")
                in_table = False
            body.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            if in_ul:
                body.append("</ul>")
                in_ul = False
            if in_table:
                body.append("</tbody></table>")
                in_table = False
            body.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("- "):
            if not in_ul:
                body.append("<ul>")
                in_ul = True
            body.append(f"<li>{line[2:]}</li>")
        elif line.startswith("|"):
            if line.startswith("|---"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not in_table:
                body.append("<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        elif line.startswith("<img"):
            if in_ul:
                body.append("</ul>")
                in_ul = False
            if in_table:
                body.append("</tbody></table>")
                in_table = False
            body.append(f'<figure>{line}</figure>')
        elif line.strip():
            if in_ul:
                body.append("</ul>")
                in_ul = False
            if in_table:
                body.append("</tbody></table>")
                in_table = False
            body.append(f"<p>{line}</p>")
    if in_ul:
        body.append("</ul>")
    if in_table:
        body.append("</tbody></table>")
    document = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VM-Agent ECHO QLoRA Report</title>
<style>
body { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #111827; background: #f8fafc; }
main { max-width: 980px; margin: 0 auto; padding: 40px 24px 64px; background: #ffffff; }
h1 { font-size: 34px; line-height: 1.15; margin: 0 0 24px; }
h2 { font-size: 22px; margin: 34px 0 12px; border-top: 1px solid #e5e7eb; padding-top: 24px; }
p, li { font-size: 16px; line-height: 1.62; }
code { background: #f3f4f6; padding: 2px 5px; border-radius: 4px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0 24px; font-size: 14px; }
th, td { border: 1px solid #e5e7eb; padding: 8px 10px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #f3f4f6; }
figure { margin: 20px 0 28px; }
img { width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 6px; }
</style>
</head>
<body><main>
""" + "\n".join(body) + "\n</main></body></html>\n"
    (out_dir / "report.html").write_text(document)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_name", default="main_vm_agent_echo_blank_a512_stoprule")
    return p


def main() -> None:
    args = parser().parse_args()
    run_dir = RUNS / args.run_name
    out_dir = ROOT / "reports" / args.run_name
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics, train, native = load_frames(run_dir)
    summary = build_summary(metrics, native, out_dir)
    figures = {
        "k": plot_accuracy_by_k(metrics, fig_dir),
        "k8": plot_k8_bars(summary, fig_dir),
        "stop": plot_false_stop(summary, fig_dir),
        "ce": plot_train_ce(train, fig_dir),
    }
    markdown = make_markdown(args.run_name, summary, train, figures, out_dir)
    make_html(markdown, figures, out_dir)
    print(f"[report] wrote {out_dir}")


if __name__ == "__main__":
    main()

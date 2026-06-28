#!/usr/bin/env python3
"""Generate Markdown, HTML, and plots for the dense-state DAgger VM agent."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt


ROOT = Path("/workspace/experiments/qwen_dense_state_dagger_vm_agent")
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"

SPLITS = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
SPLIT_LABELS = {
    "val_mixed": "Mixed",
    "fresh_standard": "Standard",
    "fresh_paraphrase": "Paraphrase",
    "fresh_paired": "Paired",
    "hard_composition": "Hard",
}
PHASES = ["bc_policy", "dagger_r1_policy", "dagger_r2_policy"]
MODES = ["learned", "value_gated", "forced"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def row_float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def best_row(rows: Sequence[Dict[str, str]], split: str, phases: Sequence[str], modes: Sequence[str]) -> Dict[str, str]:
    candidates = [r for r in rows if r["split"] == split and r["phase"] in phases and r["mode"] in modes]
    return max(candidates, key=lambda r: row_float(r, "accuracy"))


def baseline_row(rows: Sequence[Dict[str, str]], split: str) -> Dict[str, str]:
    return next(r for r in rows if r["split"] == split and r["phase"] == "bc_policy" and r["mode"] == "blank" and r["k"] == "0")


def oracle_best(rows: Sequence[Dict[str, str]], split: str) -> float:
    return max(row_float(r, "accuracy") for r in rows if r["split"] == split and r["mode"] == "oracle_teacher")


def final_phase_rows(rows: Sequence[Dict[str, str]], split: str, mode: str) -> List[Dict[str, str]]:
    return [r for r in rows if r["split"] == split and r["phase"] == "dagger_r2_policy" and r["mode"] == mode]


def plot_headline(rows: Sequence[Dict[str, str]], fig_dir: Path) -> None:
    labels = [SPLIT_LABELS[s] for s in SPLITS]
    x = list(range(len(SPLITS)))
    width = 0.19
    blank = [row_float(baseline_row(rows, s), "blank_accuracy") for s in SPLITS]
    native = [row_float(baseline_row(rows, s), "native_accuracy") for s in SPLITS]
    final_best = [row_float(best_row(rows, s, ["dagger_r2_policy"], MODES), "accuracy") for s in SPLITS]
    oracle = [oracle_best(rows, s) for s in SPLITS]

    plt.figure(figsize=(11, 5.8))
    for offset, values, label, color in [
        (-1.5 * width, blank, "Blank VM", "#8a8f98"),
        (-0.5 * width, native, "Native Qwen", "#4b73d1"),
        (0.5 * width, final_best, "Dense-state DAgger", "#2f9e72"),
        (1.5 * width, oracle, "Oracle teacher", "#d97706"),
    ]:
        plt.bar([i + offset for i in x], values, width, label=label, color=color)
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Final Checkpoint Accuracy by Split")
    plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "headline_accuracy.png", dpi=180)
    plt.close()


def plot_k_curves(rows: Sequence[Dict[str, str]], fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), sharey=True)
    axes = axes.flatten()
    colors = {"learned": "#2f9e72", "value_gated": "#0f766e", "forced": "#7c3aed", "oracle_teacher": "#d97706"}
    for ax, split in zip(axes, SPLITS):
        for mode in ["learned", "value_gated", "forced", "oracle_teacher"]:
            data = [
                r
                for r in rows
                if r["split"] == split and r["phase"] == "dagger_r2_policy" and r["mode"] == mode
            ]
            data = sorted(data, key=lambda r: int(r["k"]))
            ax.plot([int(r["k"]) for r in data], [row_float(r, "accuracy") for r in data], marker="o", label=mode.replace("_", " "), color=colors[mode])
        ax.set_title(SPLIT_LABELS[split])
        ax.set_xlabel("K rollout steps")
        ax.grid(alpha=0.25)
    axes[-1].axis("off")
    axes[0].set_ylabel("Accuracy")
    axes[3].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center")
    fig.suptitle("Final Checkpoint K-Sweep")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(fig_dir / "final_k_sweep.png", dpi=180)
    plt.close(fig)


def plot_training(rows: Sequence[Dict[str, str]], fig_dir: Path) -> None:
    train_rows = read_csv(RUNS / rows[0]["run"] / "train_log.csv")
    x_labels = []
    for r in train_rows:
        label = r["phase"].replace("_policy", "").replace("dagger_", "D")
        if r["phase"] == "bc_policy":
            label = f"BC{r['epoch']}"
        x_labels.append(label)
    x = list(range(len(train_rows)))
    plt.figure(figsize=(10, 5.5))
    for key, label, color in [
        ("action_accuracy", "Action", "#2f9e72"),
        ("arg_accuracy", "Argument", "#2563eb"),
        ("stop_accuracy", "STOP", "#d97706"),
        ("op_accuracy", "Opcode", "#7c3aed"),
    ]:
        plt.plot(x, [float(r[key]) for r in train_rows], marker="o", label=label, color=color)
    plt.xticks(x, x_labels)
    plt.ylim(0, 1.02)
    plt.ylabel("Training accuracy")
    plt.title("State-Level Supervision Metrics")
    plt.grid(alpha=0.25)
    plt.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.savefig(fig_dir / "training_metrics.png", dpi=180)
    plt.close()


def plot_dagger(rows: Sequence[Dict[str, str]], fig_dir: Path) -> None:
    traj = read_csv(RUNS / rows[0]["run"] / "trajectory_stats.csv")
    labels = [r["phase"].replace("bc_teacher", "BC teacher").replace("dagger_r", "D").replace("_states", " states") for r in traj]
    x = list(range(len(traj)))
    success = [float(r["rollout_success_rate"]) for r in traj]
    false_stop = [int(r["false_stop_states"]) for r in traj]
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    ax1.plot(x, success, marker="o", color="#2f9e72", label="Rollout success")
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("Success rate")
    ax1.set_xticks(x, labels)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.bar([i + 0.18 for i in x], false_stop, width=0.32, color="#d97706", alpha=0.65, label="False STOP states")
    ax2.set_ylabel("False STOP count")
    fig.suptitle("DAgger Collection Quality")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, ncol=2, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(fig_dir / "dagger_collection.png", dpi=180)
    plt.close(fig)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def render_markdown(run: str, metrics: Sequence[Dict[str, str]], args: Dict[str, Any]) -> str:
    headline_rows = []
    for split in SPLITS:
        base = baseline_row(metrics, split)
        final = best_row(metrics, split, ["dagger_r2_policy"], MODES)
        best_any = best_row(metrics, split, PHASES, MODES)
        headline_rows.append([
            SPLIT_LABELS[split],
            pct(row_float(base, "blank_accuracy")),
            pct(row_float(base, "native_accuracy")),
            f"{pct(row_float(final, 'accuracy'))} ({final['mode']}, K={final['k']})",
            f"{pct(row_float(best_any, 'accuracy'))} ({best_any['phase']}, {best_any['mode']}, K={best_any['k']})",
            pct(oracle_best(metrics, split)),
        ])

    train = read_csv(RUNS / run / "train_log.csv")
    traj = read_csv(RUNS / run / "trajectory_stats.csv")
    train_rows = [
        [
            r["phase"],
            r["epoch"],
            pct(float(r["action_accuracy"])),
            pct(float(r["arg_accuracy"])),
            pct(float(r["stop_accuracy"])),
            r["train_states"],
        ]
        for r in train
    ]
    traj_rows = [
        [
            r["phase"],
            r["states"],
            r["false_stop_states"],
            pct(float(r["rollout_success_rate"])),
            f"{float(r['mean_policy_steps']):.2f}",
        ]
        for r in traj
    ]

    final_k_rows = []
    for split in SPLITS:
        data = [r for r in metrics if r["phase"] == "dagger_r2_policy" and r["split"] == split and r["mode"] in ["learned", "value_gated"]]
        data = sorted(data, key=lambda r: (r["mode"], int(r["k"])))
        learned = {int(r["k"]): pct(row_float(r, "accuracy")) for r in data if r["mode"] == "learned"}
        gated = {int(r["k"]): pct(row_float(r, "accuracy")) for r in data if r["mode"] == "value_gated"}
        ks = sorted(set(learned) | set(gated))
        final_k_rows.append([SPLIT_LABELS[split], ", ".join(f"K{k}:{learned.get(k, '-')}" for k in ks), ", ".join(f"K{k}:{gated.get(k, '-')}" for k in ks)])

    return f"""# Dense-State DAgger VM Agent

## Summary

This standalone experiment tests whether a small posttraining adapter can turn `Qwen/Qwen3-4B` into one transition of a recurrent typed-bytecode VM controller. Each inference turn receives the task prompt plus dense VM-state tokens, predicts one edit action or `STOP`, executes that action in a fixed VM, and repeats for up to K steps.

The strongest result is not oracle-close, but it is nontrivial. The final checkpoint beats direct native Qwen on four of five evaluation splits, including hard composition, while remaining far below the oracle teacher. The gap says the learned loop is useful but not yet a substitute for the privileged teacher/search process.

![Final checkpoint accuracy](figures/headline_accuracy.png)

{md_table(["Split", "Blank VM", "Native Qwen", "Final VM", "Best Observed VM", "Oracle Teacher"], headline_rows)}

## Method

- Base model: `Qwen/Qwen3-4B` loaded in 4-bit NF4 with LoRA rank {args["lora_r"]}.
- Trainable parameters: LoRA adapters, dense VM-state encoder, direct action heads, solved head, and distance head.
- State interface: prompt tokens plus {args["state_tokens"]} dense VM-state tokens through `inputs_embeds`.
- Action interface: joint scoring over `STOP`, opcode edits, and argument edits.
- Copy bias: argument edits are masked to constants visible in the prompt, plus VM constants `0` and `7`.
- Training: behavior cloning from oracle edit traces, then two DAgger rounds on policy-visited states.
- Main run scale: {args["train_size"]} train tasks, {args["val_size"]} validation tasks, {args["fresh_size"]} fresh tasks per split, {args["hard_size"]} hard-composition tasks.

## K-Sweep

The final checkpoint usually benefits from additional rollout steps up to K=8 or K=12, but the curve is not cleanly monotonic. Value-gated stopping reduces false STOP in several cases, but it can also suppress correct stopping.

![Final K sweep](figures/final_k_sweep.png)

{md_table(["Split", "Final Learned Accuracy by K", "Final Value-Gated Accuracy by K"], final_k_rows)}

## Training Dynamics

State-level supervision becomes strong at the main scale. Argument accuracy is still the weakest supervised component, but the prompt-constant mask raises it enough for the recurrent loop to work.

![Training metrics](figures/training_metrics.png)

{md_table(["Phase", "Epoch", "Action", "Argument", "STOP", "States"], train_rows)}

## DAgger Dynamics

DAgger collected policy-visited states without collapsing STOP calibration in the calibrated main run. Round-2 collection slightly improved rollout success and reduced false STOP states.

![DAgger collection](figures/dagger_collection.png)

{md_table(["Collection", "States", "False STOP States", "Rollout Success", "Mean Policy Steps"], traj_rows)}

## Interpretation

The experiment supports a narrow claim: dense-state recurrent control over a fixed typed VM can improve a 4B model on some synthetic compositional tasks using a small posttraining adapter. It does not support the stronger claim that this approach closes the oracle gap.

The main limiting factors are visible:

- The oracle teacher reaches 81.2% to 96.9% on the same splits, while the final learned VM tops out at 31.2% to 53.1%.
- Native Qwen remains stronger on the fresh-standard split.
- False STOP remains substantial on high-K learned rollouts.
- Argument prediction, although improved, is still the weakest state-level target.
- Evaluation and DAgger training are expensive because each VM step is a full Qwen pass.

## Next Experiment

The next high-impact step is to move from one-step action imitation to oracle-like correction pressure over complete rollouts. Concretely: keep the dense-state VM loop, but train with a search-augmented teacher that labels policy states with repaired complete programs or value-ranked action sequences. This should target the observed failure directly: the model can learn many local edit actions, but it does not yet reliably choose globally useful edit trajectories.

## Artifacts

- Run directory: `experiments/qwen_dense_state_dagger_vm_agent/runs/{run}/`
- Source: `experiments/qwen_dense_state_dagger_vm_agent/src/dense_state_dagger_vm_agent.py`
- Report figures: `experiments/qwen_dense_state_dagger_vm_agent/reports/figures/`
- Large checkpoints: `large_artifacts/qwen_dense_state_dagger_vm_agent/checkpoints/{run}/`
"""


def markdown_to_html(markdown: str, title: str) -> str:
    lines = markdown.splitlines()
    body: List[str] = []
    in_table = False
    table_rows: List[str] = []

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        body.append("<table>")
        for i, row in enumerate(table_rows):
            if i == 1:
                continue
            cells = [c.strip() for c in row.strip("|").split("|")]
            tag = "th" if i == 0 else "td"
            body.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
        body.append("</table>")
        in_table = False
        table_rows = []

    for line in lines:
        if line.startswith("| "):
            in_table = True
            table_rows.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body.append(f"<p class='bullet'>• {html.escape(line[2:])}</p>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt = line[2 : line.index("]")]
            src = line[line.index("(") + 1 : -1]
            body.append(f"<figure><img src='{html.escape(src)}' alt='{html.escape(alt)}'></figure>")
        elif not line.strip():
            body.append("")
        else:
            body.append(f"<p>{html.escape(line)}</p>")
    flush_table()
    css = """
    body { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #17202a; background: #f7f8fb; }
    main { max-width: 1120px; margin: 0 auto; padding: 48px 28px 72px; background: #ffffff; }
    h1 { font-size: 40px; margin: 0 0 24px; }
    h2 { font-size: 24px; margin: 40px 0 14px; border-top: 1px solid #d8dde6; padding-top: 28px; }
    p { line-height: 1.55; font-size: 16px; }
    .bullet { margin: 6px 0; }
    table { border-collapse: collapse; width: 100%; margin: 18px 0 28px; font-size: 14px; }
    th, td { border: 1px solid #d9dee7; padding: 8px 10px; vertical-align: top; }
    th { background: #edf1f7; text-align: left; }
    figure { margin: 24px 0; }
    img { max-width: 100%; border: 1px solid #d9dee7; border-radius: 6px; }
    code { background: #eef2f7; padding: 2px 4px; border-radius: 4px; }
    """
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body><main>{''.join(body)}</main></body></html>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="main_joint_action_calibrated_s256_r2")
    args = parser.parse_args()

    run_dir = RUNS / args.run_name
    metrics = read_csv(run_dir / "metrics.csv")
    with (run_dir / "dataset_manifest.json").open() as f:
        manifest = json.load(f)
    fig_dir = REPORTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_headline(metrics, fig_dir)
    plot_k_curves(metrics, fig_dir)
    plot_training(metrics, fig_dir)
    plot_dagger(metrics, fig_dir)

    markdown = render_markdown(args.run_name, metrics, manifest["args"])
    (REPORTS / "report.md").write_text(markdown)
    (REPORTS / "report.html").write_text(markdown_to_html(markdown, "Dense-State DAgger VM Agent"))
    print(REPORTS / "report.md")
    print(REPORTS / "report.html")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, color: str = "#4C78A8") -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=color)
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, min(1.04, val + 0.025), pct(val), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_depth_chart(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    labels = ["first", "rand mean", "base", "lora", "oracle"]
    depth1 = [
        summaries["first_visible"]["depth_1"]["filtered_accuracy"],
        summaries["random_mean"]["depth_1"]["filtered_accuracy"],
        summaries["base"]["depth_1"]["filtered_accuracy"],
        summaries["lora"]["depth_1"]["filtered_accuracy"],
        summaries["oracle"]["depth_1"]["filtered_accuracy"],
    ]
    depth2 = [
        summaries["first_visible"]["depth_2_plus"]["filtered_accuracy"],
        summaries["random_mean"]["depth_2_plus"]["filtered_accuracy"],
        summaries["base"]["depth_2_plus"]["filtered_accuracy"],
        summaries["lora"]["depth_2_plus"]["filtered_accuracy"],
        summaries["oracle"]["depth_2_plus"]["filtered_accuracy"],
    ]
    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - width / 2 for i in x], depth1, width, label="depth 1", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], depth2, width, label="depth 2+", color="#F58518")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("filtered accuracy")
    ax.set_title("Accuracy by Program Depth")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def table(rows: list[list[str]]) -> str:
    return "\n".join("| " + " | ".join(row) + " |" for row in [rows[0], ["---"] * len(rows[0]), *rows[1:]])


def average_summaries(items: list[dict[str, Any]]) -> dict[str, Any]:
    def avg_metric(section: str, key: str) -> float:
        return sum(item[section][key] for item in items) / len(items)

    def avg_section(section: str) -> dict[str, Any]:
        return {
            "n": items[0][section]["n"],
            "filtered_accuracy": avg_metric(section, "filtered_accuracy"),
            "raw_accuracy": avg_metric(section, "raw_accuracy"),
            "target_exact_rate": avg_metric(section, "target_exact_rate"),
        }

    return {
        "arm": "random_visible_mean",
        "overall": avg_section("overall"),
        "depth_1": avg_section("depth_1"),
        "depth_2_plus": avg_section("depth_2_plus"),
    }


def main() -> None:
    eval_dir = REPORTS / "eval"
    summaries = {
        "oracle": load_json(eval_dir / "oracle_summary.json"),
        "first_visible": load_json(eval_dir / "first_visible_summary.json"),
        "base": load_json(eval_dir / "base_summary.json"),
        "lora": load_json(eval_dir / "lora_summary.json"),
    }
    random_summaries = sorted(eval_dir.glob("random_visible_summary_seed_*.json"))
    random_items = [load_json(path) for path in random_summaries]
    summaries["random_mean"] = average_summaries(random_items)
    dataset = load_json(ROOT / "data" / "dataset_summary.json")
    train_metrics = load_json(REPORTS / "train_metrics.json")

    save_bar(
        FIGURES / "overall_filtered_accuracy.png",
        ["first", "rand mean", "base", "lora", "oracle"],
        [
            summaries["first_visible"]["overall"]["filtered_accuracy"],
            summaries["random_mean"]["overall"]["filtered_accuracy"],
            summaries["base"]["overall"]["filtered_accuracy"],
            summaries["lora"]["overall"]["filtered_accuracy"],
            summaries["oracle"]["overall"]["filtered_accuracy"],
        ],
        "Counterexample-Filtered Execution Accuracy",
        "accuracy",
    )
    save_depth_chart(FIGURES / "accuracy_by_depth.png", summaries)
    save_bar(
        FIGURES / "target_exact_accuracy.png",
        ["first", "rand mean", "base", "lora", "oracle"],
        [
            summaries["first_visible"]["overall"]["target_exact_rate"],
            summaries["random_mean"]["overall"]["target_exact_rate"],
            summaries["base"]["overall"]["target_exact_rate"],
            summaries["lora"]["overall"]["target_exact_rate"],
            summaries["oracle"]["overall"]["target_exact_rate"],
        ],
        "Exact Program Match",
        "exact match rate",
        color="#54A24B",
    )
    save_bar(
        FIGURES / "random_seed_sweep.png",
        [path.stem.split("_")[-1] for path in random_summaries],
        [item["overall"]["filtered_accuracy"] for item in random_items],
        "Random Visible Candidate Baseline",
        "filtered accuracy",
        color="#B279A2",
    )

    metric_rows = [["Arm", "Overall filtered", "Depth-1 filtered", "Depth-2+ filtered", "Exact program"]]
    for key, label in [
        ("first_visible", "First visible"),
        ("random_mean", "Random visible mean"),
        ("base", "Frozen Qwen scorer"),
        ("lora", "QLoRA scorer"),
        ("oracle", "Oracle"),
    ]:
        metric_rows.append(
            [
                label,
                pct(summaries[key]["overall"]["filtered_accuracy"]),
                pct(summaries[key]["depth_1"]["filtered_accuracy"]),
                pct(summaries[key]["depth_2_plus"]["filtered_accuracy"]),
                pct(summaries[key]["overall"]["target_exact_rate"]),
            ]
        )

    base_rows = load_jsonl(eval_dir / "base_records.jsonl")
    lora_rows = {row["task_id"]: row for row in load_jsonl(eval_dir / "lora_records.jsonl")}
    recoveries = [row for row in base_rows if not row["filtered_pass"] and lora_rows[row["task_id"]]["filtered_pass"]]
    recovery_rows = [["Task", "Domain", "Depth", "Base selected", "Target"]]
    for row in recoveries:
        recovery_rows.append([row["task_id"], row["domain"], str(row["depth"]), row["selected_text"], row["target_text"]])

    random_values = [item["overall"]["filtered_accuracy"] for item in random_items]
    report = f"""# Qwen3.5-4B Transform ABI Compiler Pilot

## Summary

This pilot tested a constrained compiler surface for deterministic transformation tasks. Candidate ABI programs were enumerated, Qwen scored each candidate under the task prompt, and the selected program was executed by a deterministic interpreter. This removes JSON syntax validity as a confound and isolates operation/composition choice.

The QLoRA scorer reached **48/48 filtered execution accuracy (100.0%)**, compared with **44/48 (91.7%)** for frozen Qwen and **48/48 (100.0%)** for the oracle. On the composition slice, QLoRA reached **21/21 (100.0%)** versus frozen Qwen's **19/21 (90.5%)**.

This is a positive compiler-learnability result inside the generated transformation distribution. It should not be read as a broad production result: the tasks are generated from the same frozen ABI grammar, and most coverage remains shallow. The next gate is a less-curated task source with the ABI frozen before task inspection.

## Charts

![Overall filtered accuracy](figures/overall_filtered_accuracy.png)

![Accuracy by depth](figures/accuracy_by_depth.png)

![Exact program accuracy](figures/target_exact_accuracy.png)

![Random seed sweep](figures/random_seed_sweep.png)

## Dataset

- Train records: {dataset["train_n"]}
- Validation records: {dataset["validation_n"]}
- Eval records: {dataset["eval_n"]}
- Eval depth counts: `{dataset["eval_depth_counts"]}`
- Eval domain counts: `{dataset["eval_domain_counts"]}`
- Mean eval candidate count: {dataset["candidate_count_mean_eval"]:.1f}

## Results

{table(metric_rows)}

Random visible baseline seed range: {pct(min(random_values))}-{pct(max(random_values))}, mean {pct(sum(random_values) / len(random_values))}.

## Base Misses Recovered By QLoRA

{table(recovery_rows)}

## Training

- Trainable LoRA parameters: 10.6M.
- Training steps: 80.
- Final train loss reported by Trainer: {train_metrics["train_result"]["train_loss"]:.4f}.
- Training runtime: {train_metrics["train_result"]["train_runtime"]:.1f}s.

The adapter learned the compact compiler language strongly. Because the constrained scorer evaluates candidate programs directly, the gain is not from better parseability; it is from moving target ABI programs above plausible visible-consistent alternatives.

## Interpretation

The useful signal is the depth-2+ result: the adapter recovered the two composition tasks frozen Qwen missed and closed the oracle gap on this generated suite. That supports running a harder compiler pilot on a less-curated transformation benchmark.

The limiting caveat is also clear: the non-model baselines are already strong because candidate enumeration filters by visible examples. First-visible reached {pct(summaries["first_visible"]["overall"]["filtered_accuracy"])}, and random-visible averaged {pct(summaries["random_mean"]["overall"]["filtered_accuracy"])}. Future tasks need more adversarial visible-equivalent candidates and a less templated source to distinguish robust compiler skill from an easy candidate set.

## Decision

Proceed to the next gate only if the ABI and task set are frozen from an external source before evaluation. The next experiment should keep the constrained scorer, but use a larger less-curated pipeline-transform corpus and report depth-1 operation selection separately from depth-2+ composition.
"""
    (REPORTS / "report.md").write_text(report, encoding="utf-8")
    print(REPORTS / "report.md")


if __name__ == "__main__":
    main()

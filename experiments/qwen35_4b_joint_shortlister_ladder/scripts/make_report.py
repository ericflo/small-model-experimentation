#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "reports" / "joint_shortlister_results.json"
LOSS_PATH = ROOT / "reports" / "training_losses.json"
REPORT_PATH = ROOT / "reports" / "qwen35_4b_joint_shortlister_ladder_report.md"
FIG_DIR = ROOT / "reports" / "figures"


def pct(metric: dict[str, Any]) -> float:
    return 100.0 * float(metric["rate"])


def flatten_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, dict) and {"successes", "records", "rate"} <= set(value):
                flat[f"{key}_successes"] = value["successes"]
                flat[f"{key}_records"] = value["records"]
                flat[f"{key}_pct"] = round(100.0 * value["rate"], 3)
            else:
                flat[key] = value
        out.append(flat)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_for(rows: list[dict[str, Any]], control: str, observation: str) -> dict[str, Any] | None:
    for row in rows:
        if row["control"] == control and row["observation"] == observation:
            return row
    return None


def plot_loss(losses: list[dict[str, Any]]) -> None:
    if not losses:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot([row["step"] for row in losses], [row["loss"] for row in losses], marker=".", linewidth=1)
    plt.xlabel("optimizer step")
    plt.ylabel("loss")
    plt.title("QLoRA Joint Pair Training Loss")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_loss.png", dpi=180)
    plt.close()


def plot_control_recall(result: dict[str, Any]) -> None:
    rows = result["prediction_summary_by_control"]
    labels = [f"{row['control']}\n{row['observation']}" for row in rows]
    ks = [int(k) for k in result["recall_ks"]]
    plt.figure(figsize=(10, 5.2))
    for row in rows:
        ys = [pct(row[f"pair_recall_at_{k}"]) for k in ks]
        plt.plot(ks, ys, marker="o", label=f"{row['control']} / {row['observation']}")
    plt.xscale("log", base=2)
    plt.xticks(ks, [str(k) for k in ks])
    plt.ylim(0, 100)
    plt.xlabel("top-k joint pairs")
    plt.ylabel("target pair recall (%)")
    plt.title("Joint Pair Recall Curves")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "joint_pair_recall_curves.png", dpi=180)
    plt.close()
    _ = labels


def plot_ladder(result: dict[str, Any]) -> None:
    max_k = max(int(k) for k in result["recall_ks"])
    rows = [
        row
        for row in result["prediction_summary_by_ladder"]
        if row["control"] == "trained_model" and row["observation"] == "designed"
    ]
    if not rows:
        rows = [
            row
            for row in result["prediction_summary_by_ladder"]
            if row["control"] == "trained_model" and row["observation"] == "random"
        ]
    templates = sorted({row["template"] for row in rows})
    sizes = sorted({int(row["library_size"]) for row in rows})
    fig, axes = plt.subplots(1, len(templates), figsize=(5.2 * len(templates), 4.5), squeeze=False)
    for ax, template in zip(axes[0], templates):
        subset = {int(row["library_size"]): row for row in rows if row["template"] == template}
        ys = [pct(subset[size][f"pair_recall_at_{max_k}"]) if size in subset else 0.0 for size in sizes]
        ax.bar([str(size) for size in sizes], ys, color="#4f7d95")
        ax.set_ylim(0, 100)
        ax.set_title(template)
        ax.set_xlabel("library size")
        ax.set_ylabel(f"pair recall@{max_k} (%)")
        ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ladder_pair_recall.png", dpi=180)
    plt.close()


def plot_observation(result: dict[str, Any]) -> None:
    rows = result["observation_summary_by_ladder"]
    labels = [f"{row['library_size']}\n{row['template'].replace('pair_', '')}" for row in rows]
    random_counts = [row["random_consistent_avg"] for row in rows]
    designed_counts = [row["designed_consistent_avg"] for row in rows]
    xs = range(len(rows))
    width = 0.38
    plt.figure(figsize=(11, 5.5))
    plt.bar([x - width / 2 for x in xs], random_counts, width=width, label="random six")
    plt.bar([x + width / 2 for x in xs], designed_counts, width=width, label="designed six")
    plt.yscale("log")
    plt.xticks(list(xs), labels, rotation=35, ha="right")
    plt.ylabel("visible-consistent candidates")
    plt.title("Observation Design Reduces Candidate Ambiguity")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "observation_candidate_counts.png", dpi=180)
    plt.close()


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _key in columns) + " |"
    sep = "| " + " | ".join("---" for _label, _key in columns) + " |"
    body = []
    for row in rows:
        vals = []
        for _label, key in columns:
            value = row[key]
            if isinstance(value, float):
                vals.append(f"{value:.1f}")
            else:
                vals.append(str(value))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def control_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    max_k = max(int(k) for k in result["recall_ks"])
    out: list[dict[str, Any]] = []
    for row in result["prediction_summary_by_control"]:
        out.append(
            {
                "condition": f"{row['control']} / {row['observation']}",
                "records": row["records"],
                "pair@1": round(pct(row["pair_recall_at_1"]), 1),
                "pair@8": round(pct(row["pair_recall_at_8"]), 1) if 8 <= max_k else None,
                f"pair@{max_k}": round(pct(row[f"pair_recall_at_{max_k}"]), 1),
                f"left@{max_k}": round(pct(row[f"left_recall_at_{max_k}"]), 1),
                f"right@{max_k}": round(pct(row[f"right_recall_at_{max_k}"]), 1),
            }
        )
    return out


def ladder_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    max_k = max(int(k) for k in result["recall_ks"])
    out: list[dict[str, Any]] = []
    for row in result["prediction_summary_by_ladder"]:
        if row["control"] != "trained_model":
            continue
        out.append(
            {
                "observation": row["observation"],
                "library": row["library_size"],
                "template": row["template"],
                "records": row["records"],
                f"pair@{max_k}": round(pct(row[f"pair_recall_at_{max_k}"]), 1),
                f"left@{max_k}": round(pct(row[f"left_recall_at_{max_k}"]), 1),
                f"right@{max_k}": round(pct(row[f"right_recall_at_{max_k}"]), 1),
            }
        )
    return out


def observation_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in result["observation_summary_by_ladder"]:
        out.append(
            {
                "library": row["library_size"],
                "template": row["template"],
                "records": row["records"],
                "random survivors": round(float(row["random_consistent_avg"]), 1),
                "designed survivors": round(float(row["designed_consistent_avg"]), 1),
                "random select %": round(pct(row["random_selected_hidden_all"]), 1),
                "designed select %": round(pct(row["designed_selected_hidden_all"]), 1),
            }
        )
    return out


def report_text(result: dict[str, Any], losses: list[dict[str, Any]]) -> str:
    max_k = max(int(k) for k in result["recall_ks"])
    trained_random = row_for(result["prediction_summary_by_control"], "trained_model", "random")
    trained_designed = row_for(result["prediction_summary_by_control"], "trained_model", "designed")
    shuffled_random = row_for(result["prediction_summary_by_control"], "trained_model_shuffled_inventory", "random")
    base_random = row_for(result["prediction_summary_by_control"], "base_model", "random")
    final_loss = losses[-1]["loss"] if losses else None
    first_loss = losses[0]["loss"] if losses else None
    loss_sentence = ""
    if final_loss is not None and first_loss is not None:
        loss_sentence = f" The logged loss moved from `{first_loss:.4f}` to `{final_loss:.4f}` over `{len(losses)}` optimizer steps."
    trained_random_pair = pct(trained_random[f"pair_recall_at_{max_k}"]) if trained_random else 0.0
    trained_designed_pair = pct(trained_designed[f"pair_recall_at_{max_k}"]) if trained_designed else 0.0
    shuffled_pair = pct(shuffled_random[f"pair_recall_at_{max_k}"]) if shuffled_random else 0.0
    base_pair = pct(base_random[f"pair_recall_at_{max_k}"]) if base_random else 0.0
    observation_overall = result["observation_summary_overall"][0] if result["observation_summary_overall"] else {}
    random_survivors = observation_overall.get("random_consistent_avg", 0.0)
    designed_survivors = observation_overall.get("designed_consistent_avg", 0.0)

    return f"""# Qwen3.5-4B Joint Shortlister Ladder Report

## Summary

This standalone experiment tests whether Qwen3.5-4B can emit a joint two-operator shortlist when operator aliases are record-local and must be read from an inventory. The task ladder crosses library sizes `64, 128, 256, 512` with a higher-information numeric template and a lower-information comparison template. The adapter predicts the pair as `LLL,RRR`; evaluation reports exact recall@k over joint pairs, plus marginal LEFT and RIGHT recall.

Training produced a LoRA adapter outside this package under `/workspace/large_artifacts/qwen35_4b_joint_shortlister_ladder/models/joint_pair_lora`.{loss_sentence}

The main pilot result is trained random-observation pair recall@{max_k} of `{trained_random_pair:.1f}%`, trained designed-observation pair recall@{max_k} of `{trained_designed_pair:.1f}%`, base random pair recall@{max_k} of `{base_pair:.1f}%`, and shuffled-inventory random pair recall@{max_k} of `{shuffled_pair:.1f}%`. A beam-32 run hit CUDA OOM in the trained 512-operator condition; exact model recall is therefore reported through beam `{max_k}`.

## Joint Pair Recall

{md_table(control_rows(result), [
    ('condition', 'condition'),
    ('records', 'records'),
    ('pair@1 %', 'pair@1'),
    ('pair@8 %', 'pair@8'),
    (f'pair@{max_k} %', f'pair@{max_k}'),
    (f'left@{max_k} %', f'left@{max_k}'),
    (f'right@{max_k} %', f'right@{max_k}'),
])}

![Joint pair recall curves](figures/joint_pair_recall_curves.png)

## Difficulty Ladder

{md_table(ladder_rows(result), [
    ('observation', 'observation'),
    ('library', 'library'),
    ('template', 'template'),
    ('records', 'records'),
    (f'pair@{max_k} %', f'pair@{max_k}'),
    (f'left@{max_k} %', f'left@{max_k}'),
    (f'right@{max_k} %', f'right@{max_k}'),
])}

![Ladder pair recall](figures/ladder_pair_recall.png)

## Observation Design

The max-split probe diagnostic measures whether six designed cases carry more identifying information than six random visible cases before any model shortlist is applied. Across the diagnostic subset, random six-case observations leave `{random_survivors}` visible-consistent candidates on average; designed six-case observations leave `{designed_survivors}`.

{md_table(observation_rows(result), [
    ('library', 'library'),
    ('template', 'template'),
    ('records', 'records'),
    ('random survivors', 'random survivors'),
    ('designed survivors', 'designed survivors'),
    ('random select %', 'random select %'),
    ('designed select %', 'designed select %'),
])}

![Observation candidate counts](figures/observation_candidate_counts.png)

## Training Loss

![Training loss](figures/training_loss.png)

## Interpretation

This experiment is designed to avoid a floored binary result: it reports marginal recalls, joint recall curves, shuffled-inventory controls, and a library-by-template ladder. A useful model-side effect should show up as trained recall separating from both base and shuffled controls, with the gap changing smoothly across the ladder.

That separation did not appear. Pair recall stayed at zero in every model condition. The adapter moved marginal recall above base, but shuffled inventory retained nearly the same marginal signal, so this is not evidence that the model is reading the alias-description binding. Designed observations reduced exhaustive ambiguity, but did not change model pair recall.

The result points away from simply training the same prompt longer as the next step. The more useful next test is a lower-entropy action interface: score or classify candidate pairs produced by executable filtering, or train a reranker over a compact candidate set, while keeping the designed-observation machinery because it continues to reduce ambiguity in the executable substrate.

## Artifacts

- Dataset manifest: `data/dataset_manifest.json`
- Train pairs: `data/train_pairs.jsonl`
- Eval records: `data/eval_records.jsonl`
- Results: `reports/joint_shortlister_results.json`
- Training losses: `reports/training_losses.json`
- Large adapter: `/workspace/large_artifacts/qwen35_4b_joint_shortlister_ladder/models/joint_pair_lora`
"""


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    losses = json.loads(LOSS_PATH.read_text(encoding="utf-8")) if LOSS_PATH.exists() else []

    plot_loss(losses)
    plot_control_recall(result)
    plot_ladder(result)
    plot_observation(result)

    write_csv(ROOT / "reports" / "prediction_summary_by_control.csv", flatten_metric_rows(result["prediction_summary_by_control"]))
    write_csv(ROOT / "reports" / "prediction_summary_by_ladder.csv", flatten_metric_rows(result["prediction_summary_by_ladder"]))
    write_csv(ROOT / "reports" / "observation_summary_by_ladder.csv", flatten_metric_rows(result["observation_summary_by_ladder"]))
    write_csv(ROOT / "reports" / "observation_rows.csv", result["observation_rows"])

    REPORT_PATH.write_text(report_text(result, losses), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_PATH), "figures": str(FIG_DIR)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

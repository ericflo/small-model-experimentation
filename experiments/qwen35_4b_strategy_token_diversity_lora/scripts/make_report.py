#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diversity_utils import summarize_records  # noqa: E402
from src.jsonl import load_jsonl  # noqa: E402


BASELINE_PATHS = {
    "base_k4": "data/main_base_k4_records.jsonl",
    "default_k32": "data/main_default_extra_k32_records.jsonl",
    "hot_k32": "data/main_hot_extra_k32_records.jsonl",
    "diverse_k32": "data/main_diverse_extra_k32_records.jsonl",
    "union_k32": "data/main_union_k32_records.jsonl",
    "union_k128": "data/main_union_hot_extra_k128_records.jsonl",
}

STRATEGY_PATHS = {
    "semantic_strategy_k32": "data/main_strategy_semantic_k32_records.jsonl",
    "shuffled_strategy_k32": "data/main_strategy_shuffled_k32_records.jsonl",
    "shuffled_strategy_k32_base_missed": "data/main_strategy_shuffled_k32_base_missed_records.jsonl",
    "semantic_strategy_k32_base_missed": "data/main_strategy_semantic_k32_base_missed_records.jsonl",
    "base_plus_semantic_strategy_k32": "data/main_base_plus_semantic_strategy_k32_records.jsonl",
    "base_plus_shuffled_strategy_k32": "data/main_base_plus_shuffled_strategy_k32_records.jsonl",
}


def load_manifest_for_record(path: Path) -> dict[str, Any]:
    manifest = path.with_suffix(".manifest.json")
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    return {}


def task_sets(records: list[dict[str, Any]], base_records: list[dict[str, Any]]) -> dict[str, Any]:
    base_by_id = {record["record_id"]: record for record in base_records}
    zero_base = [record for record in records if record["record_id"] in base_by_id and not base_by_id[record["record_id"]].get("coverage")]
    recovered = sorted(record["task_id"] for record in zero_base if record.get("coverage"))
    covered = sorted(record["task_id"] for record in records if record.get("coverage"))
    return {"covered": covered, "recovered": recovered}


def summarize_arm(root: Path, name: str, path_text: str, base_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    path = root / path_text
    if not path.exists():
        return None
    records = load_jsonl(path)
    manifest = load_manifest_for_record(path)
    summary = summarize_records(records, base_records=base_records)
    usage = manifest.get("token_usage", {})
    sets = task_sets(records, base_records)
    return {
        "name": name,
        "path": path_text,
        "records": len(records),
        "summary": summary,
        "usage": usage,
        "covered_task_ids": sets["covered"],
        "recovered_task_ids": sets["recovered"],
    }


def read_sft_manifest(path: Path) -> dict[str, Any]:
    manifest = path.with_suffix(".manifest.json")
    return json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}


def loss_tail(path: Path) -> tuple[float | None, list[dict[str, Any]]]:
    if not path.exists():
        return None, []
    payload = json.loads(path.read_text(encoding="utf-8"))
    losses = payload if isinstance(payload, list) else payload.get("losses", [])
    if not losses:
        return None, []
    last = losses[-1]
    value = last.get("loss") if isinstance(last, dict) else last
    return float(value), losses


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%"


def fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def forward_tokens_for_display(arm: dict[str, Any], by_name: dict[str, dict[str, Any]]) -> int | None:
    usage = arm["usage"]
    raw = usage.get("forward_tokens")
    if raw is None:
        return None
    name = arm["name"]
    base = by_name.get("base_k4")
    union = by_name.get("union_k32")
    if base and name in {"default_k32", "hot_k32", "diverse_k32"}:
        return int(raw) + int(base["usage"].get("forward_tokens", 0))
    if union and name == "union_k128":
        return int(raw) + int(union["usage"].get("forward_tokens", 0))
    return int(raw)


def add_bar_plot(out: Path, arms: list[dict[str, Any]]) -> None:
    labels = [arm["name"].replace("_", "\n") for arm in arms]
    coverage = [arm["summary"].get("coverage", 0.0) for arm in arms]
    zero = [arm["summary"].get("zero_to_one_rate", 0.0) for arm in arms]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(arms))
    ax.bar([i - 0.18 for i in x], coverage, width=0.36, label="coverage")
    ax.bar([i + 0.18 for i in x], zero, width=0.36, label="base-miss recovery rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("Coverage and base-miss recovery")
    ax.set_xticks(list(x), labels, rotation=0, fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def add_loss_plot(out: Path, series: dict[str, list[dict[str, Any]]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, rows in series.items():
        if not rows:
            continue
        xs = [row.get("step", idx + 1) if isinstance(row, dict) else idx + 1 for idx, row in enumerate(rows)]
        ys = [row.get("loss", row) if isinstance(row, dict) else row for row in rows]
        ax.plot(xs, ys, label=name)
    ax.set_xlabel("logged step")
    ax.set_ylabel("training loss")
    ax.set_title("Adapter training losses")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def overlap_table(arms: list[dict[str, Any]]) -> list[str]:
    lines = ["| arm A | arm B | recovered overlap | A only | B only |", "|---|---|---:|---:|---:|"]
    for index, arm_a in enumerate(arms):
        set_a = set(arm_a["recovered_task_ids"])
        for arm_b in arms[index + 1 :]:
            set_b = set(arm_b["recovered_task_ids"])
            lines.append(
                f"| {arm_a['name']} | {arm_b['name']} | {len(set_a & set_b)} | {len(set_a - set_b)} | {len(set_b - set_a)} |"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/final_report.md"))
    args = parser.parse_args()

    root = ROOT
    reports = root / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    base_records = load_jsonl(root / BASELINE_PATHS["base_k4"])

    arms: list[dict[str, Any]] = []
    for name, path in {**BASELINE_PATHS, **STRATEGY_PATHS}.items():
        arm = summarize_arm(root, name, path, base_records)
        if arm is not None:
            arms.append(arm)

    plot_arms = [
        arm
        for arm in arms
        if arm["name"]
        in {
            "base_k4",
            "hot_k32",
            "union_k32",
            "union_k128",
            "base_plus_semantic_strategy_k32",
            "base_plus_shuffled_strategy_k32",
        }
    ]
    add_bar_plot(figures / "coverage_and_recovery.png", plot_arms)
    semantic_loss, semantic_rows = loss_tail(root / "reports/semantic_strategy_lora_losses.json")
    shuffled_loss, shuffled_rows = loss_tail(root / "reports/shuffled_strategy_lora_losses.json")
    add_loss_plot(figures / "training_losses.png", {"semantic": semantic_rows, "shuffled": shuffled_rows})

    semantic_sft = read_sft_manifest(root / "data/main_strategy_semantic_sft.jsonl")
    shuffled_sft = read_sft_manifest(root / "data/main_strategy_shuffled_sft.jsonl")
    semantic_counts = Counter(semantic_sft.get("train_strategy_counts", {}))
    shuffled_counts = Counter(shuffled_sft.get("train_strategy_counts", {}))

    by_name = {arm["name"]: arm for arm in arms}
    semantic = by_name.get("base_plus_semantic_strategy_k32") or by_name.get("semantic_strategy_k32")
    hot = by_name.get("hot_k32")
    union = by_name.get("union_k32")
    shuffled = by_name.get("base_plus_shuffled_strategy_k32") or by_name.get("shuffled_strategy_k32") or by_name.get("shuffled_strategy_k32_base_missed")

    conclusion = []
    if semantic and hot and union:
        sem_recovery = semantic["summary"].get("zero_to_one", 0)
        hot_recovery = hot["summary"].get("zero_to_one", 0)
        union_recovery = union["summary"].get("zero_to_one", 0)
        if sem_recovery > hot_recovery and sem_recovery >= union_recovery:
            conclusion.append("The semantic strategy-token adapter cleared the efficiency target: it beat hot K32 at matched single-arm sampling and reached the K32 union recovery level.")
        elif sem_recovery > hot_recovery:
            conclusion.append("The semantic strategy-token adapter improved over hot K32 at matched budget but did not reach the multi-policy union.")
        else:
            conclusion.append("The semantic strategy-token adapter did not beat the hot K32 inference baseline, so the training objective did not buy the desired sampling-efficiency win.")
        if shuffled:
            shuf_recovery = shuffled["summary"].get("zero_to_one", 0)
            if sem_recovery <= shuf_recovery:
                conclusion.append("The shuffled-key control matched or beat the semantic adapter on base-miss recovery, so any recovery is not attributable to meaningful strategy-key semantics.")
            else:
                conclusion.append("The semantic adapter beat the shuffled-key control on base-miss recovery, which supports a real key-to-mode effect.")

    lines: list[str] = []
    lines.append("# qwen35_4b_strategy_token_diversity_lora")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(
        "Can a small QLoRA adapter with explicit strategy tokens make extra samples on base-missed MBPP tasks behave like a more complementary ensemble, recovering misses at roughly the cost of one hot K32 arm instead of a three-policy union?"
    )
    lines.append("")
    lines.append("## Result")
    lines.append("")
    if conclusion:
        for item in conclusion:
            lines.append(f"- {item}")
    else:
        lines.append("- The final semantic/shuffled evaluation artifacts were not both available when this report was generated.")
    lines.append("")
    lines.append("## Arms")
    lines.append("")
    lines.append("| arm | records | coverage | base-miss recovered | base-miss recovery | mean candidates | functional diversity | forward tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    display_order = [
        "base_k4",
        "default_k32",
        "hot_k32",
        "diverse_k32",
        "union_k32",
        "union_k128",
        "base_plus_semantic_strategy_k32",
        "base_plus_shuffled_strategy_k32",
        "semantic_strategy_k32_base_missed",
        "shuffled_strategy_k32_base_missed",
        "semantic_strategy_k32",
    ]
    arms_by_name = {arm["name"]: arm for arm in arms}
    displayed_arms = [arms_by_name[name] for name in display_order if name in arms_by_name]
    displayed_arms.extend(arm for arm in arms if arm["name"] not in set(display_order))
    for arm in displayed_arms:
        summary = arm["summary"]
        usage = arm["usage"]
        lines.append(
            f"| {arm['name']} | {arm['records']} | {fmt_pct(summary.get('coverage'))} | "
            f"{summary.get('zero_to_one', 0)}/{summary.get('zero_base_records', 0)} | {fmt_pct(summary.get('zero_to_one_rate'))} | "
            f"{summary.get('candidate_count_mean', 0.0):.2f} | {fmt_pct(summary.get('distinct_functional_rate_mean'))} | "
            f"{fmt_int(forward_tokens_for_display(arm, arms_by_name))} |"
        )
    lines.append("")
    lines.append("![Coverage and recovery](figures/coverage_and_recovery.png)")
    lines.append("")
    lines.append("## Recovered Task IDs")
    lines.append("")
    for arm in arms:
        if arm["summary"].get("zero_base_records", 0):
            ids = ", ".join(str(task_id) for task_id in arm["recovered_task_ids"]) or "none"
            lines.append(f"- `{arm['name']}`: {ids}")
    lines.append("")
    lines.append("## Recovery Overlap")
    lines.append("")
    recovery_arms = [arm for arm in arms if arm["summary"].get("zero_base_records", 0) and arm["summary"].get("zero_to_one", 0)]
    lines.extend(overlap_table(recovery_arms) if recovery_arms else ["No recovered-task overlap table available."])
    lines.append("")
    lines.append("## Training Data")
    lines.append("")
    lines.append(
        f"- Semantic SFT rows: {semantic_sft.get('rows', 'n/a')} from {semantic_sft.get('tasks_used', 'n/a')} tasks; final logged loss {semantic_loss if semantic_loss is not None else 'n/a'}."
    )
    lines.append(
        f"- Shuffled SFT rows: {shuffled_sft.get('rows', 'n/a')} from {shuffled_sft.get('tasks_used', 'n/a')} tasks; final logged loss {shuffled_loss if shuffled_loss is not None else 'n/a'}."
    )
    lines.append("- Semantic row counts by assigned strategy: " + ", ".join(f"{key}={value}" for key, value in sorted(semantic_counts.items())))
    lines.append("- Shuffled row counts by assigned strategy: " + ", ".join(f"{key}={value}" for key, value in sorted(shuffled_counts.items())))
    lines.append("")
    lines.append("![Training losses](figures/training_losses.png)")
    lines.append("")
    lines.append("## Design Notes")
    lines.append("")
    lines.append("- The adapters were trained only on verified hidden-correct samples from MBPP train tasks.")
    lines.append("- The semantic adapter maps each correct sample to a structural strategy token; the shuffled control keeps the same target programs but breaks the mapping between token and program mode.")
    lines.append("- The primary comparison is base-miss recovery at K32-equivalent sampling cost: base + semantic strategy K32 on misses vs hot K32 vs the more expensive K32 union.")
    lines.append("- Forward-token totals are cumulative for full 80-task arms; base-missed-only diagnostic rows show the extra strategy-token sampling cost on the 24 missed tasks.")
    lines.append("- The all-80 semantic strategy pass is reported as a diagnostic only; it is not the fair budget comparison because it also spends strategy-token samples on tasks the base K4 pool already solved.")
    lines.append("- Large adapter artifacts are stored outside this experiment package under `/workspace/large_artifacts/qwen35_4b_strategy_token_diversity_lora`.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- Config: `configs/experiment.json`")
    lines.append("- Log: `logs/experiment_log.md`")
    lines.append("- Records and manifests: `data/`")
    lines.append("- Scripts: `scripts/`")
    lines.append("- Figures: `reports/figures/`")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "arms": {arm["name"]: arm["summary"] for arm in arms},
        "semantic_final_loss": semantic_loss,
        "shuffled_final_loss": shuffled_loss,
        "report": str(args.out),
    }
    (args.out.parent / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

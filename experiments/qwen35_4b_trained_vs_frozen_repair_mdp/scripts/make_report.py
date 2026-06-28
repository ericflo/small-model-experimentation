#!/usr/bin/env python3
"""Build the experiment report from saved manifests and eval summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


ARMS = [
    ("base_direct_heldout", "Direct x4"),
    ("frozen_repair_heldout", "Frozen repair"),
    ("repair_sft_heldout", "SFT repair"),
    ("sample_more_token_matched_heldout", "Sample more"),
]

POLICIES = [
    ("first_visible", "First visible-pass"),
    ("public_signature_majority", "Public-signature majority"),
    ("shortest_visible", "Shortest visible-pass"),
    ("oracle_coverage", "Oracle coverage"),
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def manifest_name(arm: str) -> Path:
    return DATA / f"{arm}_records.manifest.json"


def records_name(arm: str) -> Path:
    return DATA / f"{arm}_records.jsonl"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def task_id(record: dict) -> str:
    return str(record.get("task_id", record.get("record_id", "")))


def has_hidden_pass(record: dict) -> bool:
    return any(c.get("hidden_all_pass") or c.get("full_pass") for c in record.get("candidates", []))


def zero_to_one_task_ids(base_records: list[dict], arm_records: list[dict]) -> list[str]:
    out = []
    for base, arm in zip(base_records, arm_records):
        if not has_hidden_pass(base) and has_hidden_pass(arm):
            out.append(task_id(arm))
    return out


def final_summary(eval_path: Path) -> dict:
    data = load_json(eval_path)
    summaries = data["summary"]
    return summaries[-1]


def save_bar(path: Path, title: str, labels: list[str], values: list[float], ylabel: str, ylim: tuple[float, float] | None = None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#4c78a8", "#f58518", "#e45756", "#54a24b"]
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_line(path: Path, title: str, xs: list[float], ys: list[float], xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o", color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    manifests = {arm: load_json(manifest_name(arm)) for arm, _ in ARMS}
    records = {arm: load_jsonl(records_name(arm)) for arm, _ in ARMS}
    base_records = records["base_direct_heldout"]
    base_zero_count = sum(1 for r in base_records if not has_hidden_pass(r))

    arm_rows = []
    labels = []
    coverage_values = []
    zero_values = []
    false_values = []
    diversity_values = []
    token_values = []
    zero_task_rows = []
    for arm, label in ARMS:
        m = manifests[arm]
        rec = m["records"]
        tokens = m.get("token_usage", {}).get("forward_tokens", 0)
        z_ids = zero_to_one_task_ids(base_records, records[arm])
        arm_rows.append(
            [
                label,
                f"{rec['records']}",
                pct(rec["coverage"]),
                f"{len(z_ids)} / {base_zero_count}",
                pct((len(z_ids) / base_zero_count) if base_zero_count else 0.0),
                pct(rec.get("false_repair_rate", 0.0)) if rec.get("visible_repair_pass_count", 0) else "-",
                f"{rec['candidate_count_mean']:.2f}",
                f"{rec['distinct_behavior_rate_mean']:.3f}",
                f"{tokens:,}",
            ]
        )
        labels.append(label)
        coverage_values.append(rec["coverage"])
        zero_values.append((len(z_ids) / base_zero_count) if base_zero_count else 0.0)
        false_values.append(rec.get("false_repair_rate", 0.0) if rec.get("visible_repair_pass_count", 0) else 0.0)
        diversity_values.append(rec["distinct_behavior_rate_mean"])
        token_values.append(tokens)
        zero_task_rows.append([label, ", ".join(z_ids) if z_ids else "-"])

    eval_rows = []
    for arm, label in ARMS:
        for policy, policy_label in POLICIES:
            path = REPORTS / "eval" / f"{arm}_{policy}.json"
            s = final_summary(path)
            eval_rows.append(
                [
                    label,
                    policy_label,
                    s["budget"],
                    pct(s["coverage"]),
                    pct(s["selected_hidden_all"]),
                    pct(s["coverage_captured"]),
                ]
            )

    loss_path = REPORTS / "repair_sft_losses.json"
    loss_payload = load_json(loss_path) if loss_path.exists() else []
    if isinstance(loss_payload, dict):
        losses = loss_payload.get("losses", [])
        sft_meta = loss_payload
    else:
        losses = loss_payload
        meta_path = Path("/workspace/large_artifacts/qwen35_4b_trained_vs_frozen_repair_mdp/models/repair_sft_lora/training_metadata.json")
        sft_meta = load_json(meta_path) if meta_path.exists() else {}

    save_bar(FIGURES / "coverage_by_arm.png", "Held-Out Coverage by Arm", labels, coverage_values, "Hidden coverage", (0.55, 0.68))
    save_bar(FIGURES / "zero_to_one_by_arm.png", "Zero-to-One Lift by Arm", labels, zero_values, "Fraction of zero-base tasks", (0.0, max(0.12, max(zero_values) * 1.25)))
    save_bar(FIGURES / "false_repair_by_arm.png", "False Repair Rate", labels, false_values, "Visible-pass repairs failing hidden", (0.0, max(0.35, max(false_values) * 1.25)))
    save_bar(FIGURES / "diversity_by_arm.png", "Distinct Behavior Rate", labels, diversity_values, "Mean distinct behavior rate", (0.65, 0.80))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(token_values, zero_values, s=80, color=["#4c78a8", "#f58518", "#e45756", "#54a24b"])
    for label, x, y in zip(labels, token_values, zero_values):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_title("Zero-to-One Lift vs Estimated Forward Tokens")
    ax.set_xlabel("Estimated forward tokens")
    ax.set_ylabel("Zero-to-one rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "tokens_vs_zero_to_one.png", dpi=160)
    plt.close(fig)

    if losses:
        save_line(
            FIGURES / "repair_sft_loss.png",
            "Repair SFT Training Loss",
            [x["step"] for x in losses],
            [x["loss"] for x in losses],
            "Step",
            "Loss",
        )

    main_table = md_table(
        [
            "Arm",
            "N",
            "Coverage",
            "Zero-to-one",
            "Zero-to-one rate",
            "False repair rate",
            "Candidates/task",
            "Distinct behavior",
            "Forward tokens",
        ],
        arm_rows,
    )
    zero_task_table = md_table(["Arm", "Zero-to-one task IDs"], zero_task_rows)
    eval_table = md_table(
        ["Arm", "Commit policy", "Budget", "Coverage ceiling", "Selected hidden-pass", "Coverage captured"],
        eval_rows,
    )

    text = f"""# Qwen3.5-4B Trained vs Frozen Repair MDP Report

Date: 2026-06-25

## Summary

This experiment tested whether a trained repair policy can expand held-out coding coverage beyond frozen Qwen self-repair, under a fair comparison against spending the same estimated model-forward-token budget on more direct samples.

The result is negative for trained repair. On 150 held-out MBPP tasks, direct sampling covered 62.0% of tasks, leaving 57 zero-coverage tasks. Frozen repair recovered 3 of those 57 tasks. The SFT repair adapter recovered only 2 of 57 and had a higher false-repair rate. A token-matched sample-more baseline recovered 5 of 57, beating both repair arms at essentially the same estimated model-forward-token cost.

The practical read is: in this setup, the best use of extra model budget was more diverse direct generation, not trained repair. Frozen repair produced a small useful lift, but trained repair did not improve it.

## Main Held-Out Results

{main_table}

## Zero-To-One Tasks

{zero_task_table}

## Commit / Selection Summaries

The following table uses the final budget available in each candidate pool. `Oracle coverage` is a ceiling: it selects a hidden-correct candidate if one exists in the pool. Other policies use only public/visible candidate behavior.

{eval_table}

## Training Details

- SFT training examples: {sft_meta.get('train_examples')}.
- Max steps: {sft_meta.get('max_steps')}.
- Batch size / grad accumulation: {sft_meta.get('batch_size')} / {sft_meta.get('grad_accum')}.
- Learning rate: {sft_meta.get('learning_rate')}.
- Final logged SFT loss: {losses[-1]['loss'] if losses else 'n/a'}.
- DPO was skipped because the SFT repair arm failed the held-out gate.

## Figures

- [Coverage by arm](figures/coverage_by_arm.png)
- [Zero-to-one by arm](figures/zero_to_one_by_arm.png)
- [False repair rate](figures/false_repair_by_arm.png)
- [Diversity by arm](figures/diversity_by_arm.png)
- [Tokens vs zero-to-one](figures/tokens_vs_zero_to_one.png)
- [Repair SFT loss](figures/repair_sft_loss.png)

## Interpretation

The headline test was trained repair versus frozen repair on tasks with no hidden-correct direct sample. Trained repair did not pass that test: it recovered fewer zero-base tasks than frozen repair and produced a worse visible-pass-but-hidden-fail profile.

The sample-more baseline is the decisive comparator. It spent approximately the same model-forward-token budget as frozen repair and recovered more zero-base tasks. That means the repair loop did not justify its extra prompt structure or training in this run.

The false-repair rates matter. Frozen repair had 28 visible-passing repair candidates, 7 of which failed hidden tests. SFT repair had 24 visible-passing repair candidates, also with 7 hidden failures. Repair can create plausible candidates that satisfy public evidence but do not generalize, so aggregate visible pass rates would overstate its value.

## Limitations

- This is one held-out MBPP run, not a multi-seed estimate.
- The SFT adapter trained on only 17 mined repair examples, so the trained-arm negative should be read as a result for this small verified-repair recipe, not as a proof that repair training cannot work.
- Repair was conservative: it repaired visible-failing parsed candidates and did not repair candidates that already passed visible tests but failed hidden tests.
- No transfer benchmark was run in this package; the held-out MBPP comparison is the primary readout.
- Hidden tests were used for evaluation and train-side label mining, but not included in repair prompts.

## Conclusion

The experiment does not support trained repair as the next deployable posttraining lever. The best observed intervention was to preserve generation diversity and spend the matched budget on more direct samples. A stronger future repair experiment would need either a much larger verified repair set, a process objective that reduces false repairs, or a repair policy aimed at visible-pass hidden-fail near misses rather than only visible failures.
"""

    (REPORTS / "qwen35_4b_trained_vs_frozen_repair_mdp_report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

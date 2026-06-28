#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
CODER_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
CODER_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"
METRIC = "repair_after_first_failure@1"


def read_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def summary_row(label: str, payload: dict | None) -> dict:
    if not payload:
        return {"condition": label, "status": "missing"}
    summary = payload.get("summary", payload)
    row = {"condition": label, "status": "ok"}
    for key in [
        METRIC,
        "patch_apply_rate",
        "syntax_valid_rate",
        "visible_pass_hidden_fail_rate",
        "records",
        "successes",
    ]:
        row[key] = summary.get(key)
    return row


def load_table(reports_dir: Path, specs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [summary_row(label, read_json(reports_dir / filename)) for label, filename in specs]
    )


def load_split_table(reports_dir: Path, specs: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for split, label, filename in specs:
        row = summary_row(label, read_json(reports_dir / filename))
        row["split"] = split
        rows.append(row)
    columns = [
        "split",
        "condition",
        "status",
        METRIC,
        "patch_apply_rate",
        "syntax_valid_rate",
        "visible_pass_hidden_fail_rate",
        "records",
        "successes",
    ]
    return pd.DataFrame(rows)[columns]


def sampled_row(label: str, payload: dict | None) -> dict:
    if not payload:
        return {"condition": label, "status": "missing"}
    summary = payload.get("summary", payload)
    return {
        "condition": label,
        "status": "ok",
        "repair_after_first_failure@1": summary.get("repair_after_first_failure@1"),
        "repair_after_first_failure@3": summary.get("repair_after_first_failure@3"),
        "patch_apply_rate": summary.get("patch_apply_rate"),
        "syntax_valid_rate": summary.get("syntax_valid_rate"),
        "visible_pass_hidden_fail_rate": summary.get("visible_pass_hidden_fail_rate"),
        "records": summary.get("records"),
        "successes": summary.get("successes"),
    }


def load_sampled_table(reports_dir: Path, specs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [sampled_row(label, read_json(reports_dir / filename)) for label, filename in specs]
    )


def plot_metric(table: pd.DataFrame, title: str, output: Path) -> None:
    if METRIC not in table or not table[METRIC].notna().any():
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=table.dropna(subset=[METRIC]), x=METRIC, y="condition", color="#4c78a8")
    plt.xlim(0, 1)
    plt.title(title)
    plt.xlabel("repair_after_first_failure@1")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def format_manifest(manifest: dict) -> list[str]:
    return [
        f"- Dataset: `{manifest.get('dataset', 'missing')}`.",
        f"- Tasks: `{manifest.get('num_tasks', 'missing')}`.",
        f"- Episodes: `{manifest.get('num_episodes', 'missing')}`.",
        f"- Splits: `{manifest.get('splits', {})}`.",
        f"- Failure classes: `{manifest.get('failure_classes', {})}`.",
        f"- Bug families: `{manifest.get('bug_families', {})}`.",
    ]


def ablation_table(payload: dict | None) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    rows = []
    for mode, summary in payload.get("summaries", {}).items():
        rows.append(
            {
                "prompt_mode": mode,
                METRIC: summary.get(METRIC),
                "patch_apply_rate": summary.get("patch_apply_rate"),
                "syntax_valid_rate": summary.get("syntax_valid_rate"),
                "visible_pass_hidden_fail_rate": summary.get("visible_pass_hidden_fail_rate"),
                "successes": summary.get("successes"),
                "records": summary.get("records"),
            }
        )
    order = ["trace", "no_trace", "wrong_patch_only", "trace_only", "gold_file_removed"]
    if rows:
        df = pd.DataFrame(rows)
        df["order"] = df["prompt_mode"].map({mode: i for i, mode in enumerate(order)})
        return df.sort_values(["order", "prompt_mode"]).drop(columns=["order"])
    return pd.DataFrame()


def compact_code(text: str, *, max_lines: int = 28) -> str:
    lines = text.strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return "\n".join(lines)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def task_ids_for(manifest: dict, prefix: str) -> list[str]:
    return sorted(task_id for task_id in manifest.get("task_ids", []) if task_id.startswith(prefix))


def lora_metadata_table(models_dir: Path) -> pd.DataFrame:
    rows = []
    paths = list(models_dir.glob("v2_*_lora/experiment_metadata.json"))
    paths += list(models_dir.glob("coder_v2_*_lora/experiment_metadata.json"))
    for path in sorted(paths):
        data = read_json(path) or {}
        rows.append(
            {
                "adapter": path.parent.name,
                "model": data.get("model_id"),
                "mode": data.get("mode"),
                "shuffle_traces": data.get("shuffle_traces"),
                "rank": data.get("lora_rank"),
                "alpha": data.get("lora_alpha"),
                "dropout": data.get("lora_dropout"),
                "epochs": data.get("epochs"),
                "lr": data.get("learning_rate"),
                "max_length": data.get("max_length"),
                "train_records": data.get("train_records"),
            }
        )
    return pd.DataFrame(rows)


def result_by_episode(path: Path) -> dict[str, dict]:
    payload = read_json(path) or {}
    return {row["episode_id"]: row for row in payload.get("records", [])}


def records_by_episode(path: Path) -> dict[str, dict]:
    return {row["episode_id"]: row for row in load_jsonl(path)}


def representative_example(
    title: str,
    dataset_records: dict[str, dict],
    result_records: dict[str, dict],
    episode_id: str,
) -> list[str]:
    record = dataset_records.get(episode_id)
    result = result_records.get(episode_id)
    if not record or not result:
        return [f"### {title}", "", f"Example `{episode_id}` is missing from current artifacts.", ""]
    lines = [
        f"### {title}",
        "",
        f"- Episode: `{episode_id}`.",
        f"- Task: `{record.get('task_id')}`.",
        f"- Bug family: `{record.get('metadata', {}).get('bug_family')}`.",
        f"- Failure class: `{record.get('metadata', {}).get('failure_class')}`.",
        f"- Outcome: patch_applied=`{result.get('patch_applied')}`, visible_passed=`{result.get('visible_passed')}`, hidden_passed=`{result.get('hidden_passed')}`.",
        "",
        "Issue:",
        "",
        "```text",
        compact_code(record.get("issue", ""), max_lines=8),
        "```",
        "",
        "Target corrective diff:",
        "",
        "```diff",
        compact_code(record.get("target_next_diff", ""), max_lines=24),
        "```",
        "",
        "Generated diff:",
        "",
        "```diff",
        compact_code(result.get("extracted_patch", ""), max_lines=24),
        "```",
        "",
    ]
    return lines


def plot_failure_classes(manifest: dict, output: Path) -> None:
    counts = manifest.get("failure_classes", {})
    if not counts:
        return
    df = pd.DataFrame([{"failure_class": key, "count": value} for key, value in counts.items()])
    plt.figure(figsize=(7, 3.5))
    sns.barplot(data=df, x="count", y="failure_class", color="#59a14f")
    plt.title("v2 Wrong-Patch Failure-Class Breakdown")
    plt.xlabel("episodes")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_transfer_gap(v2_iid_table: pd.DataFrame, v2_holdout_table: pd.DataFrame, output: Path) -> None:
    if v2_iid_table.empty or v2_holdout_table.empty:
        return
    rows = []
    for split, table in [("IID", v2_iid_table), ("Held-out family", v2_holdout_table)]:
        for _, row in table.iterrows():
            rows.append({"split": split, "condition": row["condition"], METRIC: row.get(METRIC)})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=df, x=METRIC, y="condition", hue="split")
    plt.xlim(0, 1)
    plt.title("v2 Synthetic Transfer Gap")
    plt.xlabel("repair_after_first_failure@1")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_split_metric(table: pd.DataFrame, title: str, output: Path) -> None:
    if table.empty or METRIC not in table or not table[METRIC].notna().any():
        return
    plt.figure(figsize=(8, 3.8))
    sns.barplot(data=table.dropna(subset=[METRIC]), x=METRIC, y="condition", hue="split")
    plt.xlim(0, 1)
    plt.title(title)
    plt.xlabel("repair_after_first_failure@1")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def direct_swebench_table(payloads: list[dict]) -> pd.DataFrame:
    if not payloads:
        return pd.DataFrame()
    rows = []
    for payload in payloads:
        summary = payload.get("summary", {})
        repair = summary.get("repair_after_first_failure@1", {})
        end_to_end = summary.get("end_to_end_resolved@2", {})
        rows.append(
            {
                "instance": summary.get("instance_id"),
                "repo": summary.get("repo"),
                "condition": "A. frozen first patch",
                "initial_resolved@1": summary.get("initial_resolved@1"),
                "repair_after_first_failure@1": "n/a",
                "end_to_end_resolved@2": summary.get("initial_resolved@1"),
            }
        )
        for condition, value in repair.items():
            rows.append(
                {
                    "instance": summary.get("instance_id"),
                    "repo": summary.get("repo"),
                    "condition": condition,
                    "initial_resolved@1": summary.get("initial_resolved@1"),
                    "repair_after_first_failure@1": value,
                    "end_to_end_resolved@2": end_to_end.get(condition),
                }
            )
    return pd.DataFrame(rows)


def direct_real_cases(payloads: list[dict]) -> list[str]:
    if not payloads:
        return ["### Direct Real-Task Probes", "", "No direct real-task probe artifact was found.", ""]
    lines = ["### Direct Real-Task Probe Failures", ""]
    for payload in payloads:
        lines.extend(direct_real_case(payload))
    return lines


def direct_real_case(payload: dict) -> list[str]:
    summary = payload.get("summary", {})
    initial = payload.get("initial", {})
    repairs = {row.get("condition"): row for row in payload.get("repairs", [])}
    trace = repairs.get("E_trace_repair_sft", {})
    return [
        f"#### `{summary.get('instance_id')}`",
        "",
        f"- Instance: `{summary.get('instance_id')}` from `{summary.get('repo')}`.",
        f"- Harness: `{summary.get('harness')}`.",
        f"- Base failed official FAIL_TO_PASS test: `{summary.get('base_failed')}`.",
        f"- Gold patch passed the same direct test: `{summary.get('gold_passed')}`.",
        f"- Frozen first patch resolved: `{bool(summary.get('initial_resolved@1'))}`.",
        f"- Trace repair resolved after first failure: `{summary.get('repair_after_first_failure@1', {}).get('E_trace_repair_sft')}`.",
        "",
        "Frozen first patch:",
        "",
        "```diff",
        compact_code(initial.get("extracted_patch", ""), max_lines=24),
        "```",
        "",
        "Trace repair generated diff:",
        "",
        "```diff",
        compact_code(trace.get("extracted_patch", ""), max_lines=24),
        "```",
        "",
        "Observed failure:",
        "",
        "```text",
        compact_code(trace.get("test_output", ""), max_lines=18),
        "```",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--output", type=Path, default=Path("reports/execution_conditioned_repair_paper.md"))
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    v1_manifest = read_json(args.data_dir / "dataset_manifest.json") or {}
    v2_manifest = read_json(args.data_dir / "v2" / "dataset_manifest.json") or {}
    swebench = (
        read_json(args.reports_dir / "swebench_slice_results.json")
        or read_json(args.reports_dir / "swebench_preflight.json")
    )
    swebench_direct_paths = sorted(args.reports_dir.glob("swebench_direct_*_qwen3_results.json"))
    swebench_direct_payloads = [
        payload for payload in (read_json(path) for path in swebench_direct_paths) if payload
    ]

    v1_table = load_table(
        args.reports_dir,
        [
            ("B. frozen second attempt", "frozen_second_attempt_results.json"),
            ("C. final-patch SFT", "final_patch_sft_results.json"),
            ("D. no-trace repair SFT", "failure_conditioned_no_trace_results.json"),
            ("E. trace repair SFT", "failure_conditioned_trace_results.json"),
            ("F. shuffled-trace repair SFT", "failure_conditioned_shuffled_trace_results.json"),
        ],
    )
    v2_iid_table = load_table(
        args.reports_dir,
        [
            ("B. frozen second attempt", "v2_frozen_second_attempt_iid_results.json"),
            ("C. final-patch SFT", "v2_final_patch_sft_iid_results.json"),
            ("D. no-trace repair SFT", "v2_failure_conditioned_no_trace_iid_results.json"),
            ("E. trace repair SFT", "v2_failure_conditioned_trace_iid_results.json"),
            ("F. shuffled-trace repair SFT", "v2_failure_conditioned_shuffled_trace_iid_results.json"),
        ],
    )
    v2_holdout_table = load_table(
        args.reports_dir,
        [
            ("B. frozen second attempt", "v2_frozen_second_attempt_family_holdout_results.json"),
            ("C. final-patch SFT", "v2_final_patch_sft_family_holdout_results.json"),
            ("D. no-trace repair SFT", "v2_failure_conditioned_no_trace_family_holdout_results.json"),
            ("E. trace repair SFT", "v2_failure_conditioned_trace_family_holdout_results.json"),
            ("F. shuffled-trace repair SFT", "v2_failure_conditioned_shuffled_trace_family_holdout_results.json"),
        ],
    )
    v2_holdout_at3_table = load_sampled_table(
        args.reports_dir,
        [
            ("B. frozen second attempt", "v2_frozen_second_attempt_family_holdout_at3_results.json"),
            ("C. final-patch SFT", "v2_final_patch_sft_family_holdout_at3_results.json"),
            ("D. no-trace repair SFT", "v2_failure_conditioned_no_trace_family_holdout_at3_results.json"),
            ("E. trace repair SFT", "v2_failure_conditioned_trace_family_holdout_at3_results.json"),
            ("F. shuffled-trace repair SFT", "v2_failure_conditioned_shuffled_trace_family_holdout_at3_results.json"),
        ],
    )
    coder_table = load_split_table(
        args.reports_dir,
        [
            ("IID", "C'. coder final-patch SFT", "coder_v2_final_patch_sft_iid_results.json"),
            ("IID", "D'. coder no-trace repair SFT", "coder_v2_failure_conditioned_no_trace_iid_results.json"),
            ("IID", "E'. coder trace repair SFT", "coder_v2_failure_conditioned_trace_iid_results.json"),
            ("IID", "F'. coder shuffled-trace repair SFT", "coder_v2_failure_conditioned_shuffled_trace_iid_results.json"),
            (
                "Held-out family",
                "C'. coder final-patch SFT",
                "coder_v2_final_patch_sft_family_holdout_results.json",
            ),
            (
                "Held-out family",
                "D'. coder no-trace repair SFT",
                "coder_v2_failure_conditioned_no_trace_family_holdout_results.json",
            ),
            (
                "Held-out family",
                "E'. coder trace repair SFT",
                "coder_v2_failure_conditioned_trace_family_holdout_results.json",
            ),
            (
                "Held-out family",
                "F'. coder shuffled-trace repair SFT",
                "coder_v2_failure_conditioned_shuffled_trace_family_holdout_results.json",
            ),
        ],
    )
    v1_ablation = ablation_table(read_json(args.reports_dir / "trace_ablation_results.json"))
    v2_ablation = ablation_table(
        read_json(args.reports_dir / "v2_trace_ablation_family_holdout_results.json")
    )
    swebench_direct_table = direct_swebench_table(swebench_direct_payloads)
    lora_table = lora_metadata_table(args.models_dir)
    v2_iid_records = records_by_episode(args.data_dir / "v2" / "repair_val_synth_iid.jsonl")
    v2_holdout_records = records_by_episode(
        args.data_dir / "v2" / "repair_val_synth_family_holdout.jsonl"
    )
    trace_iid_results = result_by_episode(
        args.reports_dir / "v2_failure_conditioned_trace_iid_results.json"
    )
    trace_holdout_results = result_by_episode(
        args.reports_dir / "v2_failure_conditioned_trace_family_holdout_results.json"
    )

    plot_metric(v1_table, "v1 Synthetic Held-Out Repair", args.figures_dir / "v1_repair_rate.png")
    plot_metric(v2_iid_table, "v2 IID Synthetic Repair", args.figures_dir / "v2_iid_repair_rate.png")
    plot_metric(
        v2_holdout_table,
        "v2 Held-Out-Family Synthetic Repair",
        args.figures_dir / "v2_family_holdout_repair_rate.png",
    )
    plot_failure_classes(v2_manifest, args.figures_dir / "v2_failure_class_breakdown.png")
    plot_transfer_gap(v2_iid_table, v2_holdout_table, args.figures_dir / "v2_transfer_gap.png")
    plot_split_metric(
        coder_table,
        "Coding-Specialist v2 Repair Controls",
        args.figures_dir / "coder_v2_ablation_repair_rate.png",
    )

    synthetic_iid_gain = None
    synthetic_holdout_gain = None
    direct_real_gain = None
    direct_transfer_ratio = None
    direct_probe_count = len(swebench_direct_payloads)
    direct_trace_rate = None
    direct_frozen_second_rate = None
    if not v2_iid_table.empty:
        frozen = v2_iid_table.loc[v2_iid_table["condition"].str.startswith("B."), METRIC].iloc[0]
        trace = v2_iid_table.loc[v2_iid_table["condition"].str.startswith("E."), METRIC].iloc[0]
        synthetic_iid_gain = trace - frozen
    if not v2_holdout_table.empty:
        frozen = v2_holdout_table.loc[v2_holdout_table["condition"].str.startswith("B."), METRIC].iloc[0]
        trace = v2_holdout_table.loc[v2_holdout_table["condition"].str.startswith("E."), METRIC].iloc[0]
        synthetic_holdout_gain = trace - frozen
    direct_frozen_values = []
    direct_trace_values = []
    for payload in swebench_direct_payloads:
        direct_summary = payload.get("summary", {})
        repair = direct_summary.get("repair_after_first_failure@1", {})
        if "B_frozen_second_attempt" in repair and "E_trace_repair_sft" in repair:
            direct_frozen_values.append(repair["B_frozen_second_attempt"])
            direct_trace_values.append(repair["E_trace_repair_sft"])
    if direct_frozen_values and direct_trace_values:
        direct_frozen_second_rate = sum(direct_frozen_values) / len(direct_frozen_values)
        direct_trace_rate = sum(direct_trace_values) / len(direct_trace_values)
        direct_real_gain = direct_trace_rate - direct_frozen_second_rate
        if synthetic_iid_gain not in (None, 0):
            direct_transfer_ratio = direct_real_gain / synthetic_iid_gain
    direct_probe_label = {0: "No", 1: "One", 2: "Two", 3: "Three"}.get(
        direct_probe_count, str(direct_probe_count)
    )
    direct_trace_successes = int(sum(direct_trace_values)) if direct_trace_values else 0
    direct_result_refs = ", ".join(
        f"`reports/{path.name}`" for path in swebench_direct_paths
    ) or "`reports/swebench_direct_*_qwen3_results.json`"

    lines = [
        "# Execution-Conditioned Repair LoRA: Negative Transfer Result",
        "",
        "## Abstract",
        "",
        (
            "We tested whether a single QLoRA adapter trained to repair after a failed patch "
            "outperforms a single final-patch SFT adapter. On an expanded synthetic IID split, "
            "failure-conditioned adapters solved all examples while the final-patch SFT repaired "
            "11/60 and the frozen second-attempt baseline repaired 6/60. However, this success did "
            "not transfer to held-out synthetic bug families: every condition repaired 0/27 examples. "
            "A secondary coding-specialist ablation with Qwen2.5-Coder-3B reproduced the IID result "
            "and found a small held-out-family trace result, 3/27, compared with 2/27 for no-trace, "
            "0/27 for shuffled-trace, and 0/27 for final-patch. "
            f"{direct_probe_label} non-official direct pytest probes on SWE-bench Verified Flask/Requests "
            "tasks had base-fails/gold-passes behavior; the trace repair condition scored "
            f"{direct_trace_successes}/{direct_probe_count}, and no Qwen3 repair condition solved any probe. "
            "Trace controls did not establish a causal benefit from execution traces. Official "
            "SWE-bench-style Docker execution is unavailable in this runner because Docker cannot "
            "register container layers (`unshare: operation not permitted`), so no valid real-task "
            "transfer ratio is claimed."
        ),
        "",
        "## Main Claim",
        "",
        (
            "The experiment does not support the hypothesis that failure-conditioned trace SFT "
            "learns transferable execution-conditioned repair. The stronger observed effect is "
            "template-family memorization: the repair adapters master same-family synthetic "
            "validation but the primary Qwen3 adapters fail completely on unseen synthetic families. "
            "The coding-specialist ablation weakens the broadest version of that negative claim, but "
            "its 3/27 held-out-family result is small and only one success above the no-trace control."
        ),
        "",
        "## Model and Training",
        "",
        f"- Base model: `{MODEL_ID}`.",
        f"- Revision: `{REVISION}`.",
        "- Quantization/training: 4-bit NF4 QLoRA, frozen base, BF16, rank 32, alpha 64.",
        "- Loss: assistant diff tokens only.",
        "- Compared adapters: final-patch SFT, no-trace repair SFT, trace repair SFT, shuffled-trace repair SFT.",
        f"- Secondary model ablation: `{CODER_MODEL_ID}` at revision `{CODER_REVISION}` for C/D/E/F.",
        "- Evaluation applies generated diffs to the wrong-patched tree and then runs visible and hidden tests.",
        "",
        "v2 adapter metadata:",
        "",
        lora_table.to_markdown(index=False) if not lora_table.empty else "Missing.",
        "",
        "## Prompt",
        "",
        "System prompt:",
        "",
        "```text",
        "You are a coding agent repairing a repository. Output only a unified diff.",
        "Do not explain. Do not include markdown fences.",
        "```",
        "",
        "User prompt template:",
        "",
        "```text",
        "<ISSUE>",
        "{issue_text}",
        "</ISSUE>",
        "",
        "<REPO_CONTEXT>",
        "{opened_files_or_current_files}",
        "</REPO_CONTEXT>",
        "",
        "<CURRENT_DIFF>",
        "{wrong_patch}",
        "</CURRENT_DIFF>",
        "",
        "<TEST_OUTPUT_AFTER_CURRENT_DIFF>",
        "{traceback_stdout_stderr_or_apply_error}",
        "</TEST_OUTPUT_AFTER_CURRENT_DIFF>",
        "",
        "Task:",
        "Produce the minimal corrective unified diff to make the repository pass the tests.",
        "```",
        "",
        "Prompt modes: final-patch SFT uses the original buggy files and targets `base_buggy_diff`; repair SFT uses the wrong-patched files and targets `target_next_diff`; no-trace blanks test output; shuffled-trace replaces test output with another training trace; wrong-patch-only removes repository file context and blanks test output; trace-only removes the current diff; gold-file-removed replaces file contents with a withholding marker.",
        "",
        "## Dataset",
        "",
        "### v1 Pilot",
        "",
        *format_manifest(v1_manifest),
        "",
        "### v2 Expanded Synthetic",
        "",
        *format_manifest(v2_manifest),
        "",
        "v2 includes an IID validation split from trained bug families and a held-out-family split using `path_norm` and `tie_breaking`, which are absent from training.",
        "",
        "![v2 failure-class breakdown](../figures/v2_failure_class_breakdown.png)",
        "",
        "Exact v2 train task IDs:",
        "",
        "```text",
        ", ".join(task_ids_for(v2_manifest, "train_")),
        "```",
        "",
        "Exact v2 IID validation task IDs:",
        "",
        "```text",
        ", ".join(task_ids_for(v2_manifest, "val_synth_iid_")),
        "```",
        "",
        "Exact v2 held-out-family validation task IDs:",
        "",
        "```text",
        ", ".join(task_ids_for(v2_manifest, "val_synth_family_holdout_")),
        "```",
        "",
        "## Results",
        "",
        "### v1 Pilot Held-Out Synthetic",
        "",
        v1_table.to_markdown(index=False),
        "",
        "### v2 IID Synthetic",
        "",
        v2_iid_table.to_markdown(index=False),
        "",
        "![v2 IID repair rate](../figures/v2_iid_repair_rate.png)",
        "",
        "### v2 Held-Out-Family Synthetic",
        "",
        v2_holdout_table.to_markdown(index=False),
        "",
        "![v2 held-out-family repair rate](../figures/v2_family_holdout_repair_rate.png)",
        "",
        "### v2 Held-Out-Family Synthetic, Best-of-3 Sampling",
        "",
        "Sampling used `temperature=0.2`, `top_p=0.95`, and three candidate diffs per episode. The `@1` column is the first sampled candidate; `@3` is best-of-three by hidden-test pass.",
        "",
        v2_holdout_at3_table.to_markdown(index=False),
        "",
        "### Coding-Specialist Repair-Control Ablation",
        "",
        (
            "To test whether the primary held-out-family collapse was specific to Qwen3-4B, I trained "
            "the same v2 final-patch, no-trace repair, trace repair, and shuffled-trace repair adapters "
            "on `Qwen/Qwen2.5-Coder-3B-Instruct`. This is still a secondary ablation because the frozen "
            "B baseline was not rerun for the coder model, but it includes the trace-specific D/E/F "
            "controls needed to interpret the small held-out-family trace result."
        ),
        "",
        coder_table.to_markdown(index=False),
        "",
        "![coding-specialist v2 ablation](../figures/coder_v2_ablation_repair_rate.png)",
        "",
        "### v2 Transfer Gap View",
        "",
        "![v2 transfer gap](../figures/v2_transfer_gap.png)",
        "",
        "## Trace Controls",
        "",
        "### v1 Trace Adapter on v1 Held-Out Synthetic",
        "",
        v1_ablation.to_markdown(index=False) if not v1_ablation.empty else "Missing.",
        "",
        "### v2 Trace Adapter on v2 Held-Out Families",
        "",
        v2_ablation.to_markdown(index=False) if not v2_ablation.empty else "Missing.",
        "",
        (
            "The requested evidence pattern `normal trace > no trace`, `normal trace > shuffled trace`, "
            "and `normal trace > wrong-patch-only` is not present. On v2 IID, no-trace, trace, and "
            "shuffled-trace adapters all repair 60/60. On v2 held-out families, all five ablation "
            "prompt modes repair 0/27."
        ),
        "",
        "## Representative Cases",
        "",
        *representative_example(
            "Same-Family Success",
            v2_iid_records,
            trace_iid_results,
            "val_synth_iid_clamp_8::near_miss",
        ),
        *representative_example(
            "Held-Out-Family Failure",
            v2_holdout_records,
            trace_holdout_results,
            "val_synth_family_holdout_normalize_path_0::near_miss",
        ),
        *direct_real_cases(swebench_direct_payloads),
        "",
        "## Synthetic-to-Real Transfer Status",
        "",
        f"- v2 IID trace gain over frozen second attempt: `{synthetic_iid_gain}`.",
        f"- v2 held-out-family trace gain over frozen second attempt: `{synthetic_holdout_gain}`.",
        f"- Direct non-Docker SWE-bench Verified probe count: `{direct_probe_count}`.",
        f"- Direct non-Docker frozen second-attempt repair rate: `{direct_frozen_second_rate}`.",
        f"- Direct non-Docker trace repair rate: `{direct_trace_rate}`.",
        f"- Direct non-Docker real-task gain: `{direct_real_gain}`.",
        f"- Direct non-Docker transfer ratio: `{direct_transfer_ratio}`.",
        "- Official Docker SWE-bench real-task gain: `not measured`.",
        "- Official Docker SWE-bench transfer ratio: `not defined`.",
        "",
        "Direct non-Docker SWE-bench Verified probes:",
        "",
        swebench_direct_table.to_markdown(index=False) if not swebench_direct_table.empty else "Missing.",
        "",
        "Direct-probe selection note: I preflighted every SWE-bench Verified task in the currently supported `pallets/flask` and `psf/requests` profiles. The validated direct slice includes tasks with base-fails/gold-passes behavior. Excluded same-profile candidates either failed gold validation under the local profile (`psf__requests-2931`) or could not install on Python 3.12 because older vendored urllib3 code imports removed stdlib symbols (`psf__requests-1142`, `psf__requests-1724`, `psf__requests-1766`, `psf__requests-1921`, `psf__requests-2317`).",
        "",
        "Docker/SWE-bench preflight:",
        "",
        "```json",
        json.dumps(swebench or {"status": "missing"}, indent=2, sort_keys=True),
        "```",
        "",
        "The official real-task portion is blocked by the runner, not by a modeling decision. The direct probes are useful negative evidence, but they are not a replacement for the official Docker SWE-bench harness or a statistically meaningful real slice.",
        "",
        "## Interpretation",
        "",
        "1. Final-patch SFT is weaker than repair-conditioned SFT on same-family synthetic validation.",
        "2. The apparent same-family synthetic gain does not survive held-out-family validation.",
        "3. Execution traces are not shown to be causally useful in the current setup.",
        "4. A coding-specialist model can recover a small amount of held-out-family repair under the trace-conditioned objective, but the effect is weak: 3/27 for trace versus 2/27 for no-trace and 0/27 for shuffled-trace.",
        "5. Patch application and visible-test pass rates can be high while hidden repair remains zero, so executable hidden tests are essential.",
        "6. The current result is negative and should not be framed as successful synthetic-to-real transfer.",
        "",
        "## Reproducibility Artifacts",
        "",
        "- Dataset builders: `scripts/build_repair_dataset.py`, `scripts/build_repair_dataset_v2.py`.",
        "- Training: `scripts/train_repair_lora.py`.",
        "- Evaluation: `scripts/eval_repair_synthetic.py`, `scripts/run_trace_ablation.py`.",
        "- Direct real-task probe: `scripts/eval_repair_swebench_direct.py`.",
        "- v2 adapters: `models/v2_final_patch_sft_lora`, `models/v2_failure_conditioned_no_trace_lora`, `models/v2_failure_conditioned_trace_lora`, `models/v2_failure_conditioned_shuffled_trace_lora`.",
        "- Coder ablation adapters: `models/coder_v2_final_patch_sft_lora`, `models/coder_v2_failure_conditioned_no_trace_lora`, `models/coder_v2_failure_conditioned_trace_lora`, `models/coder_v2_failure_conditioned_shuffled_trace_lora`.",
        "- v2 result JSONs: `reports/v2_*_results.json`, `reports/v2_trace_ablation_family_holdout_results.json`.",
        "- Coder ablation result JSONs: `reports/coder_v2_*_results.json`.",
        f"- Direct real-task result JSONs: {direct_result_refs}.",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(line) for line in lines) + "\n"
    args.output.write_text(text, encoding="utf-8")

    # Keep the historical report path current for convenience.
    legacy = args.reports_dir / "transfer_gap_report.md"
    legacy.write_text(text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

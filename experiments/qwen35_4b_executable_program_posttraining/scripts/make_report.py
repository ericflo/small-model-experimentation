#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(item: dict) -> str:
    return f"{100 * item['rate']:.1f}% ({item['successes']}/{item['records']})"


def main() -> None:
    manifest = load_json(ROOT / "data" / "dataset_manifest.json")
    eval_files = sorted(path for path in (REPORTS / "eval").glob("*.json") if not path.name.startswith("_"))
    lines = [
        "# Qwen 3.5 4B Executable Program Posttraining",
        "",
        "## Question",
        "",
        "Can a Qwen 3.5 4B adapter trained to emit executable DSL repair programs produce programs that generalize to held-out composition families, and does visible-test reranking improve hidden-case success?",
        "",
        "## Design",
        "",
        "- Base model: `Qwen/Qwen3.5-4B`.",
        "- Model output: one executable DSL expression.",
        "- Evaluator: parses and executes generated programs on visible and hidden cases.",
        "- Reranking: samples candidate programs and selects the valid candidate with the most visible-case passes.",
        "- Main held-out families: `modulo_sum_label`, `length_contains_code`, and `tuple_branch_label`.",
        "- Adapter weights and checkpoints are stored outside the compact directory under `/workspace/large_artifacts/qwen35_4b_executable_program_posttraining/models/`.",
        "",
        "## Dataset",
        "",
        f"- Train records: {manifest['train_records']}.",
        f"- IID eval records: {manifest['iid_eval_records']}.",
        f"- Holdout eval records: {manifest['holdout_records']}.",
        f"- Visible cases per record: {manifest['visible_cases_per_record']}.",
        f"- Hidden cases per record: {manifest['hidden_cases_per_record']}.",
        "",
        "## Iteration Readout",
        "",
        "The first trace-trained executable-program adapter transferred cleanly on two held-out families but failed the length+contains family completely. Inspection showed that failed generations repeatedly substituted `count_eq text needle` for the needed `len text` predicate inside a conjunction. A second adapter was trained from scratch with three training-only conjunction families added under the same 240-record budget.",
        "",
        "Key held-out results:",
        "",
        "| Condition | Prompt | Samples | Greedy Hidden | Rerank Hidden | modulo_sum_label | length_contains_code | tuple_branch_label |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    key_files = [
        "trace_lora_trace_holdout.json",
        "trace_and_bridge_lora_trace_holdout_samples3.json",
        "trace_and_bridge_lora_no_trace_holdout_greedy.json",
        "trace_and_bridge_lora_shuffled_trace_holdout_greedy.json",
    ]
    for name in key_files:
        path = REPORTS / "eval" / name
        if not path.exists():
            continue
        result = load_json(path)
        summary = result["summary"]
        by_family = summary["by_family"]
        label = {
            "trace_lora_trace_holdout.json": "Initial trace adapter",
            "trace_and_bridge_lora_trace_holdout_samples3.json": "Conjunction-support trace adapter",
            "trace_and_bridge_lora_no_trace_holdout_greedy.json": "Conjunction-support adapter",
            "trace_and_bridge_lora_shuffled_trace_holdout_greedy.json": "Conjunction-support adapter",
        }[name]
        lines.append(
            "| "
            + label
            + " | `"
            + result["prompt_mode"]
            + "` | "
            + str(result["num_samples"])
            + " | "
            + pct(summary["overall"]["greedy_hidden_all"])
            + " | "
            + pct(summary["overall"]["rerank_hidden_all"])
            + " | "
            + pct(by_family["modulo_sum_label"]["rerank_hidden_all"])
            + " | "
            + pct(by_family["length_contains_code"]["rerank_hidden_all"])
            + " | "
            + pct(by_family["tuple_branch_label"]["rerank_hidden_all"])
            + " |"
        )
    lines.extend(
        [
            "",
            "Readout:",
            "",
            "- Executable DSL posttraining produced a large held-out signal on `modulo_sum_label` and `tuple_branch_label`.",
            "- The initial failure on `length_contains_code` was not random formatting noise; it was a specific mechanism error.",
            "- Adding non-held-out conjunction training families moved `length_contains_code` from 0/24 to 7/24 under visible reranking, while preserving 24/24 on modulo and 23/24 on tuple.",
            "- Aligned visible traces mattered: the conjunction-support adapter scored 54/72 with aligned trace plus 3 samples, 24/72 with no trace greedy, and 27/72 with shuffled trace greedy.",
            "- A full 12-sample evaluation of the second adapter was started but stopped after two records because generations were taking over 90 seconds per record. The reported second-adapter rerank condition uses 3 samples and a 64-token cap.",
            "",
        ]
    )
    lines.extend(
        [
        "## Results",
        "",
        ]
    )
    if not eval_files:
        lines.extend(["No evaluation files were found under `reports/eval/`.", ""])
    for path in eval_files:
        result = load_json(path)
        summary = result["summary"]["overall"]
        lines.extend(
            [
                f"### {path.stem}",
                "",
                f"- Adapter: `{result.get('adapter')}`.",
                f"- Prompt mode: `{result['prompt_mode']}`.",
                f"- Records: {result['records']}.",
                f"- Greedy hidden all-pass: {pct(summary['greedy_hidden_all'])}.",
                f"- Visible-rerank hidden all-pass: {pct(summary['rerank_hidden_all'])}.",
                f"- Greedy visible all-pass: {pct(summary['greedy_visible_all'])}.",
                f"- Rerank visible all-pass: {pct(summary['rerank_visible_all'])}.",
                "",
                "| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for family, fam_summary in result["summary"]["by_family"].items():
            lines.append(
                "| "
                + family
                + " | "
                + pct(fam_summary["greedy_hidden_all"])
                + " | "
                + pct(fam_summary["rerank_hidden_all"])
                + " | "
                + pct(fam_summary["greedy_visible_all"])
                + " | "
                + pct(fam_summary["rerank_visible_all"])
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Artifact Layout",
            "",
            "- Compact artifacts: `/workspace/experiments/qwen35_4b_executable_program_posttraining/`.",
            "- Large artifacts: `/workspace/large_artifacts/qwen35_4b_executable_program_posttraining/`.",
            "- Dataset manifest: `data/dataset_manifest.json`.",
            "- Evaluation JSON files: `reports/eval/`.",
            "",
        ]
    )
    (REPORTS / "qwen35_4b_executable_program_posttraining_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(REPORTS / "qwen35_4b_executable_program_posttraining_report.md")


if __name__ == "__main__":
    main()

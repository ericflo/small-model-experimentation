#!/usr/bin/env python
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(metric: dict) -> str:
    return f"{100 * metric['rate']:.1f}% ({metric['successes']}/{metric['records']})"


def condition_label(name: str) -> str:
    labels = {
        "random_lora_random_holdout.json": "Random-trace adapter on random traces",
        "random_lora_counterexample_holdout.json": "Random-trace adapter on counterexample traces",
        "counterexample_lora_counterexample_holdout.json": "Counterexample-trace adapter on counterexample traces",
        "counterexample_lora_no_trace_holdout.json": "Counterexample-trace adapter, no trace",
        "counterexample_lora_shuffled_trace_holdout.json": "Counterexample-trace adapter, shuffled trace",
    }
    return labels.get(name, name.replace(".json", ""))


def metric(result: dict, name: str = "rerank_hidden_all") -> dict:
    return result["summary"]["overall"][name]


def fam_metric(result: dict, family: str, name: str = "rerank_hidden_all") -> dict:
    return result["summary"]["by_family"][family][name]


def successes(result: dict, name: str = "rerank_hidden_all") -> int:
    return int(metric(result, name)["successes"])


def top_programs(result: dict, family: str, field: str, limit: int = 3) -> list[tuple[str, int]]:
    counts = Counter(row[field] for row in result["rows"] if row["family"] == family)
    return [(program, count) for program, count in counts.most_common(limit)]


def main() -> None:
    manifest = load_json(ROOT / "data" / "dataset_manifest.json")
    eval_files = sorted(path for path in (REPORTS / "eval").glob("*.json") if not path.name.startswith("_"))
    lines = [
        "# Qwen 3.5 4B Counterexample-Directed DSL",
        "",
        "## Question",
        "",
        "Can visible traces chosen as counterexamples to plausible wrong programs improve executable DSL repair compared with random visible traces?",
        "",
        "## Design",
        "",
        "- Base model: `Qwen/Qwen3.5-4B`.",
        "- Model output: one executable DSL expression.",
        "- Training: 4-bit NF4 QLoRA adapters.",
        "- Evaluation: parse and execute generated programs on visible and hidden cases.",
        "- Candidate selection: choose the valid candidate with the most visible-case passes.",
        "- Main held-out families: `modulo_sum_label`, `length_contains_code`, and `tuple_branch_label`.",
        "- Large adapter/checkpoint files are stored outside the compact experiment directory.",
        "",
        "## Dataset",
        "",
        f"- Random trace train records: {manifest['datasets']['random']['train_records']}.",
        f"- Counterexample trace train records: {manifest['datasets']['counterexample']['train_records']}.",
        f"- Holdout records per trace regime: {manifest['datasets']['counterexample']['holdout_records']}.",
        f"- Visible cases per record: {manifest['datasets']['counterexample']['visible_cases_per_record']}.",
        f"- Hidden cases per record: {manifest['datasets']['counterexample']['hidden_cases_per_record']}.",
        "",
        "## Main Results",
        "",
        "| Condition | Prompt | Samples | Greedy Hidden | Rerank Hidden | modulo_sum_label | length_contains_code | tuple_branch_label |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    preferred = [
        "random_lora_random_holdout.json",
        "random_lora_counterexample_holdout.json",
        "counterexample_lora_counterexample_holdout.json",
        "counterexample_lora_no_trace_holdout.json",
        "counterexample_lora_shuffled_trace_holdout.json",
    ]
    ordered = [REPORTS / "eval" / name for name in preferred if (REPORTS / "eval" / name).exists()]
    ordered += [path for path in eval_files if path not in ordered]
    for path in ordered:
        result = load_json(path)
        summary = result["summary"]
        by_family = summary["by_family"]
        lines.append(
            "| "
            + condition_label(path.name)
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
    results = {path.name: load_json(path) for path in ordered}
    random_counterexample = results.get("random_lora_counterexample_holdout.json")
    counterexample_main = results.get("counterexample_lora_counterexample_holdout.json")
    counterexample_no_trace = results.get("counterexample_lora_no_trace_holdout.json")
    counterexample_shuffled = results.get("counterexample_lora_shuffled_trace_holdout.json")
    if random_counterexample and counterexample_main and counterexample_no_trace and counterexample_shuffled:
        length_top = top_programs(counterexample_main, "length_contains_code", "greedy_program", 1)[0]
        random_length_top = top_programs(random_counterexample, "length_contains_code", "greedy_program", 3)
        lines.extend(
            [
                "",
                "## Readout",
                "",
                "- Counterexample-directed training improved greedy hidden all-pass on the counterexample holdout from "
                + f"{pct(metric(random_counterexample, 'greedy_hidden_all'))} to {pct(metric(counterexample_main, 'greedy_hidden_all'))}, "
                + "but did not improve reranked hidden all-pass: "
                + f"{pct(metric(random_counterexample))} to {pct(metric(counterexample_main))}.",
                "- Coherent traces mattered for the counterexample-trained adapter: no-trace greedy hidden all-pass was "
                + f"{pct(metric(counterexample_no_trace, 'greedy_hidden_all'))}, while shuffled traces fell to "
                + f"{pct(metric(counterexample_shuffled, 'greedy_hidden_all'))}.",
                "- The effect was family-specific. The counterexample-trained adapter reached "
                + f"{pct(fam_metric(counterexample_main, 'modulo_sum_label'))} on `modulo_sum_label` and "
                + f"{pct(fam_metric(counterexample_main, 'tuple_branch_label'))} on `tuple_branch_label`, "
                + "but stayed at "
                + f"{pct(fam_metric(counterexample_main, 'length_contains_code'))} on `length_contains_code`.",
                "- The main failure was not syntax. On `length_contains_code`, the counterexample-trained adapter generated "
                + f"`{length_top[0]}` on {length_top[1]}/24 holdout records. This valid program confuses text length with needle count; "
                + "sampling produced no useful diversity for reranking.",
                "- The random-trace adapter was less collapsed on the same `length_contains_code` holdout. Its top greedy programs were: "
                + "; ".join(f"{count}/24 `{program}`" for program, count in random_length_top)
                + ".",
                "",
                "## Interpretation",
                "",
                "This experiment gives a mixed but useful answer. Counterexample-directed visible traces are strong supervision when the model has already learned the right primitive composition, as shown by the tuple-family rescue from sampled reranking. They are not sufficient by themselves to force the model to learn the correct latent primitive binding: the length family collapsed into a stable `count_eq` alias even though the selected traces were intended to distinguish plausible wrong programs.",
                "",
                "The next iteration should make the counterexamples adaptive to the model's actual wrong program, not just to hand-authored distractors. In this run the selector distinguished the target from planned distractors, but it did not anticipate the learned `count_eq` alias. A stronger loop would sample candidate model programs during training-data construction, execute them, add traces that separate those candidates from the target, and then retrain or continue training on those model-specific counterexamples.",
            ]
        )
    lines.extend(
        [
            "",
            "## Per-Condition Details",
            "",
        ]
    )
    for path in ordered:
        result = load_json(path)
        summary = result["summary"]
        lines.extend(
            [
                f"### {path.stem}",
                "",
                f"- Adapter: `{result.get('adapter')}`.",
                f"- Data: `{result['data']}`.",
                f"- Prompt mode: `{result['prompt_mode']}`.",
                f"- Samples: {result['num_samples']}.",
                f"- Greedy hidden all-pass: {pct(summary['overall']['greedy_hidden_all'])}.",
                f"- Rerank hidden all-pass: {pct(summary['overall']['rerank_hidden_all'])}.",
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
            "- Compact artifacts: `/workspace/experiments/qwen35_4b_counterexample_directed_dsl/`.",
            "- Large artifacts: `/workspace/large_artifacts/qwen35_4b_counterexample_directed_dsl/`.",
            "- Dataset manifest: `data/dataset_manifest.json`.",
            "- Evaluation JSON files: `reports/eval/`.",
            "",
        ]
    )
    output = REPORTS / "qwen35_4b_counterexample_directed_dsl_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

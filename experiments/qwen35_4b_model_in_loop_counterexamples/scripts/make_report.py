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
        "seed_lora_challenge.json": "Seed adapter",
        "static_bridge_lora_challenge.json": "Static bridge adapter",
        "model_loop_lora_challenge.json": "Model-loop bridge adapter",
        "model_loop_lora_no_trace_challenge.json": "Model-loop bridge adapter, no trace",
        "model_loop_lora_shuffled_trace_challenge.json": "Model-loop bridge adapter, shuffled trace",
        "seed_lora_iid.json": "Seed adapter, IID",
        "static_bridge_lora_iid.json": "Static bridge adapter, IID",
        "model_loop_lora_iid.json": "Model-loop bridge adapter, IID",
    }
    return labels.get(name, name.replace(".json", ""))


def top_programs(result: dict, family: str, field: str = "greedy_program", limit: int = 5) -> list[tuple[str, int]]:
    counts = Counter(row[field] for row in result["rows"] if row["family"] == family)
    return counts.most_common(limit)


def metric(result: dict, name: str = "rerank_hidden_all") -> dict:
    return result["summary"]["overall"][name]


def fam_metric(result: dict, family: str, name: str = "rerank_hidden_all") -> dict:
    return result["summary"]["by_family"][family][name]


def fam_pct_or_na(by_family: dict, family: str, name: str = "rerank_hidden_all") -> str:
    if family not in by_family:
        return "n/a"
    return pct(by_family[family][name])


def maybe(path: Path):
    return load_json(path) if path.exists() else None


def main() -> None:
    manifest = load_json(ROOT / "data" / "dataset_manifest.json")
    mining = maybe(REPORTS / "mining" / "seed_model_mining.json")
    eval_files = sorted(path for path in (REPORTS / "eval").glob("*.json") if not path.name.startswith("_"))
    preferred = [
        "seed_lora_challenge.json",
        "static_bridge_lora_challenge.json",
        "model_loop_lora_challenge.json",
        "model_loop_lora_no_trace_challenge.json",
        "model_loop_lora_shuffled_trace_challenge.json",
        "seed_lora_iid.json",
        "static_bridge_lora_iid.json",
        "model_loop_lora_iid.json",
    ]
    ordered = [REPORTS / "eval" / name for name in preferred if (REPORTS / "eval" / name).exists()]
    ordered += [path for path in eval_files if path not in ordered]

    lines = [
        "# Qwen 3.5 4B Model-In-Loop Counterexamples",
        "",
        "## Question",
        "",
        "Can counterexamples selected against Qwen-generated wrong DSL programs improve executable program repair beyond static counterexample traces under the same training budget?",
        "",
        "## Design",
        "",
        "- Base model: `Qwen/Qwen3.5-4B`.",
        "- Model output: one executable DSL expression.",
        "- Training: 4-bit NF4 QLoRA adapters.",
        "- Training budget: 240 records per trained adapter.",
        "- Seed adapter: 240 base-family random-trace records.",
        "- Static bridge adapter: 180 base-family records plus 60 challenge-family static counterexample records.",
        "- Model-loop bridge adapter: 180 base-family records plus 60 challenge-family records whose traces were selected against seed-adapter wrong programs.",
        "- Evaluation: parse and execute generated programs on visible and hidden cases.",
        "- Candidate selection: choose the valid candidate with the most visible-case passes.",
        "- Large adapter/checkpoint files are stored outside the compact experiment directory.",
        "",
        "## Dataset",
        "",
        f"- Seed train records: {manifest['seed_train_records']}.",
        f"- Static bridge train records: {manifest['static_bridge_train_records']}.",
        f"- Model-loop bridge allocation: {manifest['bridge_allocation']}.",
        f"- IID eval records: {manifest['iid_eval_records']}.",
        f"- Challenge eval records: {manifest['challenge_eval_records']}.",
        f"- Mining pool records: {manifest['mining_pool_records']}.",
        f"- Visible cases per record: {manifest['visible_cases_per_record']}.",
        f"- Hidden cases per record: {manifest['hidden_cases_per_record']}.",
    ]
    if mining:
        lines.extend(["", "## Mining Summary", ""])
        for family, summary in mining["family_summaries"].items():
            top = "; ".join(
                f"{item['count']}x `{item['program']}`"
                for item in summary["top_wrong_programs"][:3]
            )
            lines.append(
                f"- `{family}`: {summary['bridge_records_with_model_wrong']}/{summary['bridge_records_requested']} bridge records had sampled model wrong programs; "
                f"{summary['unique_wrong_programs']} unique wrong programs mined. Top: {top if top else 'none'}."
            )
    if ordered:
        lines.extend(
            [
                "",
                "## Main Results",
                "",
                "| Condition | Data | Prompt | Samples | Greedy Hidden | Rerank Hidden | modulo_sum_label | length_contains_code | tuple_branch_label |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for path in ordered:
            result = load_json(path)
            by_family = result["summary"]["by_family"]
            data_name = Path(result["data"]).name
            lines.append(
                "| "
                + condition_label(path.name)
                + " | `"
                + data_name
                + "` | `"
                + result["prompt_mode"]
                + "` | "
                + str(result["num_samples"])
                + " | "
                + pct(result["summary"]["overall"]["greedy_hidden_all"])
                + " | "
                + pct(result["summary"]["overall"]["rerank_hidden_all"])
                + " | "
                + fam_pct_or_na(by_family, "modulo_sum_label")
                + " | "
                + fam_pct_or_na(by_family, "length_contains_code")
                + " | "
                + fam_pct_or_na(by_family, "tuple_branch_label")
                + " |"
            )
    results = {path.name: load_json(path) for path in ordered}
    seed = results.get("seed_lora_challenge.json")
    static = results.get("static_bridge_lora_challenge.json")
    model_loop = results.get("model_loop_lora_challenge.json")
    no_trace = results.get("model_loop_lora_no_trace_challenge.json")
    shuffled = results.get("model_loop_lora_shuffled_trace_challenge.json")
    if seed and static and model_loop:
        lines.extend(
            [
                "",
                "## Readout",
                "",
                "- Challenge reranked hidden all-pass: seed "
                + pct(metric(seed))
                + ", static bridge "
                + pct(metric(static))
                + ", model-loop bridge "
                + pct(metric(model_loop))
                + ".",
                "- `length_contains_code` reranked hidden all-pass: seed "
                + pct(fam_metric(seed, "length_contains_code"))
                + ", static bridge "
                + pct(fam_metric(static, "length_contains_code"))
                + ", model-loop bridge "
                + pct(fam_metric(model_loop, "length_contains_code"))
                + ".",
                "- `tuple_branch_label` reranked hidden all-pass: seed "
                + pct(fam_metric(seed, "tuple_branch_label"))
                + ", static bridge "
                + pct(fam_metric(static, "tuple_branch_label"))
                + ", model-loop bridge "
                + pct(fam_metric(model_loop, "tuple_branch_label"))
                + ".",
                "- `modulo_sum_label` reranked hidden all-pass: seed "
                + pct(fam_metric(seed, "modulo_sum_label"))
                + ", static bridge "
                + pct(fam_metric(static, "modulo_sum_label"))
                + ", model-loop bridge "
                + pct(fam_metric(model_loop, "modulo_sum_label"))
                + ".",
            ]
        )
        if no_trace and shuffled:
            lines.extend(
                [
                    "- Model-loop trace ablations: aligned trace "
                    + pct(metric(model_loop, "greedy_hidden_all"))
                    + ", no trace "
                    + pct(metric(no_trace, "greedy_hidden_all"))
                    + ", shuffled trace "
                    + pct(metric(shuffled, "greedy_hidden_all"))
                    + ".",
                ]
            )
        lines.extend(["", "## Failure Signatures", ""])
        for label, result in [
            ("Seed", seed),
            ("Static bridge", static),
            ("Model-loop bridge", model_loop),
        ]:
            top = "; ".join(f"{count}/24 `{program}`" for program, count in top_programs(result, "length_contains_code", limit=4))
            lines.append(f"- {label} `length_contains_code` greedy programs: {top}.")
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "- Static bridge and model-loop bridge both solved the challenge set under reranking, while static bridge was cleaner under greedy decoding.",
                "- Model-loop mining was still useful diagnostically: it exposed the seed adapter's stable wrong hypotheses, especially the `count_eq` substitute for string length.",
                "- Trace ablations show the symbolic trace is semantically active. Removing it mainly hurts `length_contains_code`; shuffling it collapses all challenge families.",
                "- For this task shape, the strongest training recipe is not yet the extra active-mining loop. It is targeted bridge coverage with an execution-based verifier.",
                "- The next higher-leverage experiment should make bridge selection adaptive only after expanding the held-out challenge space enough that static bridge records no longer saturate it.",
            ]
        )
    if ordered:
        lines.extend(["", "## Per-Condition Details", ""])
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
            "- Compact artifacts: `/workspace/experiments/qwen35_4b_model_in_loop_counterexamples/`.",
            "- Large artifacts: `/workspace/large_artifacts/qwen35_4b_model_in_loop_counterexamples/`.",
            "- Dataset manifest: `data/dataset_manifest.json`.",
            "- Mining report: `reports/mining/seed_model_mining.json`.",
            "- Evaluation JSON files: `reports/eval/`.",
            "",
        ]
    )
    output = REPORTS / "qwen35_4b_model_in_loop_counterexamples_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

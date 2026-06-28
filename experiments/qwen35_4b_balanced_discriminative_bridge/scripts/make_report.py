#!/usr/bin/env python
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe(path: Path) -> dict | None:
    return load_json(path) if path.exists() else None


def pct(metric: dict) -> str:
    return f"{100 * metric['rate']:.1f}% ({metric['successes']}/{metric['records']})"


def condition_label(name: str) -> str:
    labels = {
        "seed_lora_frontier.json": "Seed adapter, frontier",
        "static_bridge_lora_frontier.json": "Static bridge adapter, frontier",
        "alias_discriminative_bridge_lora_frontier.json": "Alias-discriminative bridge adapter, frontier",
        "model_discriminative_bridge_lora_frontier.json": "Model-discriminative bridge adapter, frontier",
        "seed_lora_hard_frontier.json": "Seed adapter, hard frontier",
        "static_bridge_lora_hard_frontier.json": "Static bridge adapter, hard frontier",
        "alias_discriminative_bridge_lora_hard_frontier.json": "Alias-discriminative bridge adapter, hard frontier",
        "model_discriminative_bridge_lora_hard_frontier.json": "Model-discriminative bridge adapter, hard frontier",
        "static_bridge_lora_no_trace_hard_frontier.json": "Static bridge adapter, no trace hard frontier",
        "static_bridge_lora_shuffled_trace_hard_frontier.json": "Static bridge adapter, shuffled trace hard frontier",
        "alias_discriminative_bridge_lora_no_trace_hard_frontier.json": "Alias-discriminative bridge adapter, no trace hard frontier",
        "alias_discriminative_bridge_lora_shuffled_trace_hard_frontier.json": "Alias-discriminative bridge adapter, shuffled trace hard frontier",
        "model_discriminative_bridge_lora_no_trace_hard_frontier.json": "Model-discriminative bridge adapter, no trace hard frontier",
        "model_discriminative_bridge_lora_shuffled_trace_hard_frontier.json": "Model-discriminative bridge adapter, shuffled trace hard frontier",
        "seed_lora_iid.json": "Seed adapter, IID",
        "static_bridge_lora_iid.json": "Static bridge adapter, IID",
        "alias_discriminative_bridge_lora_iid.json": "Alias-discriminative bridge adapter, IID",
        "model_discriminative_bridge_lora_iid.json": "Model-discriminative bridge adapter, IID",
    }
    return labels.get(name, name.replace(".json", "").replace("_", " "))


def metric(result: dict, name: str = "rerank_hidden_all") -> dict:
    return result["summary"]["overall"][name]


def fam_pct(result: dict, family: str, name: str = "rerank_hidden_all") -> str:
    by_family = result["summary"]["by_family"]
    return pct(by_family[family][name]) if family in by_family else "n/a"


def top_programs(result: dict, family: str, field: str = "greedy_program", limit: int = 3) -> list[tuple[str, int]]:
    counts = Counter(row[field] for row in result["rows"] if row["family"] == family)
    return counts.most_common(limit)


def mining_lines(mining: dict | None) -> list[str]:
    if not mining:
        return []
    lines = ["", "## Model-Discriminative Mining", ""]
    lines.append(f"- Allocation mode: `{mining.get('allocation_mode')}`.")
    lines.append(f"- Selector case mode: `{mining.get('selector_case_mode')}`.")
    lines.append(f"- Bridge allocation: `{mining.get('bridge_allocation')}`.")
    for family, summary in mining["family_summaries"].items():
        top = "; ".join(
            f"{item['count']}x `{item['program']}`"
            for item in summary["top_wrong_programs"][:3]
        )
        lines.append(
            f"- `{family}`: {summary['bridge_records_with_model_wrong']}/{summary['bridge_records_requested']} selected records had seed-adapter wrong programs; "
            f"wrong-candidate score {summary.get('wrong_candidate_score', 0)}; "
            f"{summary['unique_wrong_programs']} unique model wrong programs. Top: {top if top else 'none'}."
        )
    return lines


def result_table(lines: list[str], title: str, ordered: list[Path], results: dict[str, dict]) -> None:
    present = [path for path in ordered if path.name in results]
    if not present:
        return
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            "| Condition | Data | Prompt | Samples | Greedy Hidden | Rerank Hidden |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for path in present:
        result = results[path.name]
        lines.append(
            "| "
            + condition_label(path.name)
            + " | `"
            + Path(result["data"]).name
            + "` | `"
            + result["prompt_mode"]
            + "` | "
            + str(result["num_samples"])
            + " | "
            + pct(result["summary"]["overall"]["greedy_hidden_all"])
            + " | "
            + pct(result["summary"]["overall"]["rerank_hidden_all"])
            + " |"
        )


def family_table(lines: list[str], title: str, names: list[str], results: dict[str, dict], families: list[str]) -> None:
    present = [name for name in names if name in results]
    if not present:
        return
    lines.extend(["", f"## {title}", ""])
    lines.append("| Family | " + " | ".join(condition_label(name) for name in present) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in present) + " |")
    for family in families:
        lines.append("| `" + family + "` | " + " | ".join(fam_pct(results[name], family) for name in present) + " |")


def main() -> None:
    manifest = load_json(ROOT / "data" / "dataset_manifest.json")
    mining = maybe(REPORTS / "mining" / "model_discriminative_mining.json")
    preferred = [
        "seed_lora_frontier.json",
        "static_bridge_lora_frontier.json",
        "alias_discriminative_bridge_lora_frontier.json",
        "model_discriminative_bridge_lora_frontier.json",
        "seed_lora_hard_frontier.json",
        "static_bridge_lora_hard_frontier.json",
        "alias_discriminative_bridge_lora_hard_frontier.json",
        "model_discriminative_bridge_lora_hard_frontier.json",
        "static_bridge_lora_no_trace_hard_frontier.json",
        "static_bridge_lora_shuffled_trace_hard_frontier.json",
        "alias_discriminative_bridge_lora_no_trace_hard_frontier.json",
        "alias_discriminative_bridge_lora_shuffled_trace_hard_frontier.json",
        "model_discriminative_bridge_lora_no_trace_hard_frontier.json",
        "model_discriminative_bridge_lora_shuffled_trace_hard_frontier.json",
        "seed_lora_iid.json",
        "static_bridge_lora_iid.json",
        "alias_discriminative_bridge_lora_iid.json",
        "model_discriminative_bridge_lora_iid.json",
    ]
    eval_files = sorted(path for path in (REPORTS / "eval").glob("*.json") if not path.name.startswith("_"))
    ordered = [REPORTS / "eval" / name for name in preferred if (REPORTS / "eval" / name).exists()]
    ordered += [path for path in eval_files if path not in ordered]
    results = {path.name: load_json(path) for path in ordered}
    frontier_families = manifest["frontier_families"]

    lines = [
        "# Qwen 3.5 4B Balanced Discriminative Bridge",
        "",
        "## Question",
        "",
        "Can equal frontier-family coverage improve when visible traces are chosen to discriminate against harder aliases and seed-adapter mistakes, while keeping the same 240-record posttraining budget?",
        "",
        "## Design",
        "",
        "- Base model: `Qwen/Qwen3.5-4B`.",
        "- Model output: one executable DSL expression.",
        "- Training: 4-bit NF4 QLoRA adapters.",
        "- Training budget: 240 records per trained adapter.",
        "- Seed adapter: 240 base-family random-trace records.",
        "- Static bridge adapter: 180 base-family records plus 60 equally allocated normal frontier bridge records.",
        "- Alias-discriminative bridge adapter: 180 base-family records plus 60 equally allocated hard-case frontier records selected against an expanded alias bank.",
        "- Model-discriminative bridge adapter: 180 base-family records plus 60 equally allocated hard-case frontier records selected against seed-adapter wrong programs plus the alias bank.",
        "- Evaluation: normal frontier, harder frontier, trace controls, and IID retention.",
        "- Candidate selection: choose the valid candidate with the most visible-case passes.",
        "- Large adapter/checkpoint files are stored outside the compact experiment directory.",
        "",
        "## Dataset",
        "",
        f"- Seed train records: {manifest['seed_train_records']}.",
        f"- Bridge anchor records per bridge condition: {manifest['bridge_base_records']}.",
        f"- Bridge records per bridge condition: {manifest['bridge_total']}.",
        f"- Static bridge train records: {manifest['static_bridge_train_records']}.",
        f"- Alias-discriminative train records: {manifest['alias_discriminative_train_records']}.",
        f"- Frontier eval records: {manifest['frontier_eval_records']}.",
        f"- Hard frontier eval records: {manifest['hard_frontier_eval_records']}.",
        f"- IID eval records: {manifest['iid_eval_records']}.",
        f"- Mining pool records: {manifest['mining_pool_records']}.",
        f"- Frontier families: {len(frontier_families)}.",
        f"- Visible cases per record: {manifest['visible_cases_per_record']}.",
        f"- Hidden cases per record: {manifest['hidden_cases_per_record']}.",
    ]
    lines.extend(mining_lines(mining))

    result_table(
        lines,
        "Normal Frontier Results",
        [REPORTS / "eval" / name for name in preferred[:4]],
        results,
    )
    result_table(
        lines,
        "Hard Frontier Results",
        [REPORTS / "eval" / name for name in preferred[4:8]],
        results,
    )
    result_table(
        lines,
        "Trace Control Results",
        [REPORTS / "eval" / name for name in preferred[8:14]],
        results,
    )
    result_table(
        lines,
        "IID Retention Results",
        [REPORTS / "eval" / name for name in preferred[14:18]],
        results,
    )

    family_table(lines, "Normal Frontier By Family", preferred[:4], results, frontier_families)
    family_table(lines, "Hard Frontier By Family", preferred[4:8], results, frontier_families)

    seed = results.get("seed_lora_hard_frontier.json")
    static = results.get("static_bridge_lora_hard_frontier.json")
    alias = results.get("alias_discriminative_bridge_lora_hard_frontier.json")
    model = results.get("model_discriminative_bridge_lora_hard_frontier.json")
    no_trace = results.get("static_bridge_lora_no_trace_hard_frontier.json")
    shuffled_trace = results.get("static_bridge_lora_shuffled_trace_hard_frontier.json")
    if seed and static and alias and model:
        lines.extend(
            [
                "",
                "## Readout",
                "",
                "- Hard frontier reranked hidden all-pass: seed "
                + pct(metric(seed))
                + ", static bridge "
                + pct(metric(static))
                + ", alias-discriminative bridge "
                + pct(metric(alias))
                + ", model-discriminative bridge "
                + pct(metric(model))
                + ".",
                "- Hard frontier greedy hidden all-pass: seed "
                + pct(metric(seed, "greedy_hidden_all"))
                + ", static bridge "
                + pct(metric(static, "greedy_hidden_all"))
                + ", alias-discriminative bridge "
                + pct(metric(alias, "greedy_hidden_all"))
                + ", model-discriminative bridge "
                + pct(metric(model, "greedy_hidden_all"))
                + ".",
            ]
        )
        if no_trace and shuffled_trace:
            lines.extend(
                [
                    "- Static bridge trace controls on hard frontier: correct trace "
                    + pct(metric(static, "greedy_hidden_all"))
                    + ", no trace "
                    + pct(metric(no_trace, "greedy_hidden_all"))
                    + ", shuffled trace "
                    + pct(metric(shuffled_trace, "greedy_hidden_all"))
                    + ".",
                ]
            )
        lines.extend(
            [
                "",
                "## Next Experiment Options",
                "",
                "1. Recommended: run a static-normal bridge ceiling breaker. Keep `Qwen/Qwen3.5-4B`, keep equal family allocation, and replace selector hardness with harder held-out family construction: more unseen compositions, longer inputs, adversarial edge cases, and trace controls. This directly tests whether the 119/120 result is a real bridge-interface gain or an evaluation ceiling.",
                "2. Run a bridge-budget and case-count ablation around the static recipe: 20/40/60/80 bridge records and 2/4/6/8 visible cases per record. This identifies whether the gain is coming from family coverage, trace density, or sheer bridge-token exposure.",
                "3. Run a mild hard-case mixture instead of fully hard discriminative selection: 75% normal static records and 25% hard selector records within each family. This tests whether the regression came from hard-case distribution shift rather than discriminative selection itself.",
                "4. Run trace-semantic regularization only after the ceiling breaker: train with a small fraction of corrupted or missing traces labeled by the correct program. This is higher risk, but the shuffled-trace collapse shows the interface is semantically sensitive enough to justify a targeted robustness experiment.",
            ]
        )
        lines.extend(["", "## Failure Signatures", ""])
        for family in frontier_families:
            seed_top = "; ".join(f"{count}x `{program}`" for program, count in top_programs(seed, family, limit=2))
            model_top = "; ".join(f"{count}x `{program}`" for program, count in top_programs(model, family, limit=2))
            lines.append(f"- `{family}` seed greedy top: {seed_top if seed_top else 'none'}; model-discriminative greedy top: {model_top if model_top else 'none'}.")

    if ordered:
        lines.extend(["", "## Per-Condition Details", ""])
        for path in ordered:
            result = results[path.name]
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
            "- Compact artifacts: `/workspace/experiments/qwen35_4b_balanced_discriminative_bridge/`.",
            "- Large artifacts: `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/`.",
            "- Dataset manifest: `data/dataset_manifest.json`.",
            "- Mining reports: `reports/mining/`.",
            "- Evaluation JSON files: `reports/eval/`.",
            "",
        ]
    )
    output = REPORTS / "qwen35_4b_balanced_discriminative_bridge_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

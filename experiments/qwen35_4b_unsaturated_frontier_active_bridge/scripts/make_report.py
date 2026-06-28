#!/usr/bin/env python
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def maybe(path: Path):
    return load_json(path) if path.exists() else None


def pct(metric: dict) -> str:
    return f"{100 * metric['rate']:.1f}% ({metric['successes']}/{metric['records']})"


def condition_label(name: str) -> str:
    labels = {
        "seed_lora_frontier.json": "Seed adapter",
        "static_bridge_lora_frontier.json": "Static bridge adapter",
        "seed_mined_bridge_lora_frontier.json": "Seed-mined bridge adapter",
        "adaptive_bridge_lora_frontier.json": "Adaptive bridge adapter",
        "adaptive_bridge_lora_no_trace_frontier.json": "Adaptive bridge adapter, no trace",
        "adaptive_bridge_lora_shuffled_trace_frontier.json": "Adaptive bridge adapter, shuffled trace",
        "seed_lora_iid.json": "Seed adapter, IID",
        "static_bridge_lora_iid.json": "Static bridge adapter, IID",
        "seed_mined_bridge_lora_iid.json": "Seed-mined bridge adapter, IID",
        "adaptive_bridge_lora_iid.json": "Adaptive bridge adapter, IID",
    }
    return labels.get(name, name.replace(".json", ""))


def metric(result: dict, name: str = "rerank_hidden_all") -> dict:
    return result["summary"]["overall"][name]


def fam_pct(result: dict, family: str, name: str = "rerank_hidden_all") -> str:
    by_family = result["summary"]["by_family"]
    if family not in by_family:
        return "n/a"
    return pct(by_family[family][name])


def top_programs(result: dict, family: str, field: str = "greedy_program", limit: int = 4) -> list[tuple[str, int]]:
    counts = Counter(row[field] for row in result["rows"] if row["family"] == family)
    return counts.most_common(limit)


def mining_lines(title: str, mining: dict | None) -> list[str]:
    if not mining:
        return []
    lines = ["", f"## {title}", ""]
    lines.append(f"- Allocation mode: `{mining.get('allocation_mode')}`.")
    lines.append(f"- Bridge allocation: `{mining.get('bridge_allocation')}`.")
    for family, summary in mining["family_summaries"].items():
        top = "; ".join(
            f"{item['count']}x `{item['program']}`"
            for item in summary["top_wrong_programs"][:3]
        )
        lines.append(
            f"- `{family}`: {summary['bridge_records_with_model_wrong']}/{summary['bridge_records_requested']} selected records had model-generated wrong programs; "
            f"wrong-candidate score {summary.get('wrong_candidate_score', 0)}; "
            f"{summary['unique_wrong_programs']} unique wrong programs. Top: {top if top else 'none'}."
        )
    return lines


def main() -> None:
    manifest = load_json(ROOT / "data" / "dataset_manifest.json")
    seed_mining = maybe(REPORTS / "mining" / "seed_mined_mining.json")
    adaptive_mining = maybe(REPORTS / "mining" / "adaptive_mining.json")
    preferred = [
        "seed_lora_frontier.json",
        "static_bridge_lora_frontier.json",
        "seed_mined_bridge_lora_frontier.json",
        "adaptive_bridge_lora_frontier.json",
        "adaptive_bridge_lora_no_trace_frontier.json",
        "adaptive_bridge_lora_shuffled_trace_frontier.json",
        "seed_lora_iid.json",
        "static_bridge_lora_iid.json",
        "seed_mined_bridge_lora_iid.json",
        "adaptive_bridge_lora_iid.json",
    ]
    eval_files = sorted(path for path in (REPORTS / "eval").glob("*.json") if not path.name.startswith("_"))
    ordered = [REPORTS / "eval" / name for name in preferred if (REPORTS / "eval" / name).exists()]
    ordered += [path for path in eval_files if path not in ordered]
    results = {path.name: load_json(path) for path in ordered}
    frontier_families = manifest["challenge_families"]

    lines = [
        "# Qwen 3.5 4B Unsaturated Frontier Active Bridge",
        "",
        "## Question",
        "",
        "Can active bridge allocation outperform uniform static bridge coverage on a frontier suite broad enough that static bridge examples do not automatically saturate the target space?",
        "",
        "## Design",
        "",
        "- Base model: `Qwen/Qwen3.5-4B`.",
        "- Model output: one executable DSL expression.",
        "- Training: 4-bit NF4 QLoRA adapters.",
        "- Training budget: 240 records per trained adapter.",
        "- Seed adapter: 240 base-family random-trace records.",
        "- Static bridge adapter: 180 base-family records plus 60 uniformly allocated static frontier bridge records.",
        "- Seed-mined bridge adapter: 180 base-family records plus 60 uniformly allocated bridge records selected against seed-adapter wrong programs.",
        "- Adaptive bridge adapter: 180 base-family records plus 60 bridge records allocated toward wrong programs generated after static bridge training.",
        "- Evaluation: parse and execute generated programs on visible and hidden cases.",
        "- Candidate selection: choose the valid candidate with the most visible-case passes.",
        "- Large adapter/checkpoint files are stored outside the compact experiment directory.",
        "",
        "## Dataset",
        "",
        f"- Seed train records: {manifest['seed_train_records']}.",
        f"- Static bridge train records: {manifest['static_bridge_train_records']}.",
        f"- Bridge total: {manifest['bridge_total']}.",
        f"- Frontier families: {len(frontier_families)}.",
        f"- Frontier eval records: {manifest['challenge_eval_records']}.",
        f"- IID eval records: {manifest['iid_eval_records']}.",
        f"- Mining pool records: {manifest['mining_pool_records']}.",
        f"- Visible cases per record: {manifest['visible_cases_per_record']}.",
        f"- Hidden cases per record: {manifest['hidden_cases_per_record']}.",
    ]
    lines.extend(mining_lines("Seed-Adapter Mining Summary", seed_mining))
    lines.extend(mining_lines("Static-Adapter Adaptive Mining Summary", adaptive_mining))

    if ordered:
        lines.extend(
            [
                "",
                "## Main Results",
                "",
                "| Condition | Data | Prompt | Samples | Greedy Hidden | Rerank Hidden |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for path in ordered:
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

    frontier_paths = [
        "seed_lora_frontier.json",
        "static_bridge_lora_frontier.json",
        "seed_mined_bridge_lora_frontier.json",
        "adaptive_bridge_lora_frontier.json",
    ]
    if any(name in results for name in frontier_paths):
        lines.extend(["", "## Frontier By Family", ""])
        header = "| Family | " + " | ".join(condition_label(name) for name in frontier_paths if name in results) + " |"
        lines.append(header)
        lines.append("| --- | " + " | ".join("---:" for name in frontier_paths if name in results) + " |")
        for family in frontier_families:
            cells = [fam_pct(results[name], family) for name in frontier_paths if name in results]
            lines.append("| `" + family + "` | " + " | ".join(cells) + " |")

    seed = results.get("seed_lora_frontier.json")
    static = results.get("static_bridge_lora_frontier.json")
    seed_mined = results.get("seed_mined_bridge_lora_frontier.json")
    adaptive = results.get("adaptive_bridge_lora_frontier.json")
    no_trace = results.get("adaptive_bridge_lora_no_trace_frontier.json")
    shuffled = results.get("adaptive_bridge_lora_shuffled_trace_frontier.json")
    if seed and static and seed_mined and adaptive:
        lines.extend(
            [
                "",
                "## Readout",
                "",
                "- Frontier reranked hidden all-pass: seed "
                + pct(metric(seed))
                + ", static bridge "
                + pct(metric(static))
                + ", seed-mined bridge "
                + pct(metric(seed_mined))
                + ", adaptive bridge "
                + pct(metric(adaptive))
                + ".",
                "- Static bridge greedy hidden all-pass: "
                + pct(metric(static, "greedy_hidden_all"))
                + ".",
                "- Adaptive bridge greedy hidden all-pass: "
                + pct(metric(adaptive, "greedy_hidden_all"))
                + ".",
            ]
        )
        if no_trace and shuffled:
            lines.append(
                "- Adaptive bridge prompt controls: aligned trace "
                + pct(metric(adaptive, "greedy_hidden_all"))
                + ", no trace "
                + pct(metric(no_trace, "greedy_hidden_all"))
                + ", shuffled trace "
                + pct(metric(shuffled, "greedy_hidden_all"))
                + "."
            )
        lines.extend(["", "## Failure Signatures", ""])
        for family in frontier_families:
            seed_top = "; ".join(f"{count}x `{program}`" for program, count in top_programs(seed, family, limit=2))
            adaptive_top = "; ".join(f"{count}x `{program}`" for program, count in top_programs(adaptive, family, limit=2))
            lines.append(f"- `{family}` seed greedy top: {seed_top if seed_top else 'none'}; adaptive greedy top: {adaptive_top if adaptive_top else 'none'}.")

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
            "- Compact artifacts: `/workspace/experiments/qwen35_4b_unsaturated_frontier_active_bridge/`.",
            "- Large artifacts: `/workspace/large_artifacts/qwen35_4b_unsaturated_frontier_active_bridge/`.",
            "- Dataset manifest: `data/dataset_manifest.json`.",
            "- Mining reports: `reports/mining/`.",
            "- Evaluation JSON files: `reports/eval/`.",
            "",
        ]
    )
    output = REPORTS / "qwen35_4b_unsaturated_frontier_active_bridge_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(metric: dict[str, Any]) -> str:
    return f"{100 * metric['rate']:.1f}% ({metric['successes']}/{metric['records']})"


def label(name: str) -> str:
    labels = {
        "dsl_lora_ceiling.json": "DSL baseline, ceiling",
        "dsl_lora_support.json": "DSL baseline, support",
        "dsl_lora_iid.json": "DSL baseline, IID",
        "graphir_construct_ceiling.json": "GraphIR construct, ceiling",
        "graphir_construct_support.json": "GraphIR construct, support",
        "graphir_construct_iid.json": "GraphIR construct, IID",
        "graphir_pipeline_ceiling.json": "GraphIR construct+repair, ceiling",
        "graphir_pipeline_support.json": "GraphIR construct+repair, support",
        "graphir_pipeline_iid.json": "GraphIR construct+repair, IID",
        "graphir_pipeline_no_trace_ceiling.json": "GraphIR pipeline, no trace ceiling",
        "graphir_pipeline_shuffled_trace_ceiling.json": "GraphIR pipeline, shuffled trace ceiling",
        "graphir_repair_corrupt_ceiling.json": "GraphIR repair, corrupted ceiling",
    }
    return labels.get(name, name.removesuffix(".json").replace("_", " "))


def dsl_metric(result: dict[str, Any], key: str = "rerank_hidden_all") -> dict[str, Any]:
    return result["summary"]["overall"][key]


def graph_metric(result: dict[str, Any], key: str = "repair_hidden_all") -> dict[str, Any]:
    return result["summary"]["overall"][key]


def result_metric(name: str, result: dict[str, Any]) -> dict[str, Any]:
    if "construct_adapter" in result:
        return graph_metric(result, "repair_hidden_all")
    if "candidate_graph" in (result.get("rows") or [{}])[0]:
        return graph_metric(result, "repair_hidden_all")
    return dsl_metric(result, "rerank_hidden_all")


def table(lines: list[str], title: str, names: list[str], results: dict[str, dict[str, Any]]) -> None:
    present = [name for name in names if name in results]
    if not present:
        return
    lines.extend(["", f"## {title}", ""])
    lines.append("| Condition | Data | Prompt | Samples | Main Hidden | Secondary Hidden |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    for name in present:
        result = results[name]
        if "construct_adapter" in result:
            main = pct(graph_metric(result, "repair_hidden_all"))
            secondary = pct(graph_metric(result, "construct_rerank_hidden_all"))
        elif "candidate_graph" in (result.get("rows") or [{}])[0]:
            main = pct(graph_metric(result, "repair_hidden_all"))
            secondary = pct(graph_metric(result, "input_hidden_all"))
        else:
            main = pct(dsl_metric(result, "rerank_hidden_all"))
            secondary = pct(dsl_metric(result, "greedy_hidden_all"))
        lines.append(
            f"| {label(name)} | `{Path(result['data']).name}` | `{result['prompt_mode']}` | "
            f"{result['num_samples']} | {main} | {secondary} |"
        )


def family_table(lines: list[str], title: str, names: list[str], results: dict[str, dict[str, Any]], families: list[str]) -> None:
    present = [name for name in names if name in results]
    if not present:
        return
    lines.extend(["", f"## {title}", ""])
    lines.append("| Family | " + " | ".join(label(name) for name in present) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in present) + " |")
    for family in families:
        cells = []
        for name in present:
            result = results[name]
            fam = result["summary"]["by_family"].get(family)
            if not fam:
                cells.append("n/a")
            elif "construct_adapter" in result:
                cells.append(pct(fam["repair_hidden_all"]))
            elif "candidate_graph" in (result.get("rows") or [{}])[0]:
                cells.append(pct(fam["repair_hidden_all"]))
            else:
                cells.append(pct(fam["rerank_hidden_all"]))
        lines.append("| `" + family + "` | " + " | ".join(cells) + " |")


def top_outputs(result: dict[str, Any], family: str, field: str, limit: int = 2) -> str:
    counts = Counter(row.get(field, "") for row in result["rows"] if row["family"] == family)
    return "; ".join(f"{count}x `{value}`" for value, count in counts.most_common(limit) if value) or "none"


def write_charts(results: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    names = ["dsl_lora_ceiling.json", "graphir_construct_ceiling.json", "graphir_pipeline_ceiling.json"]
    present = [name for name in names if name in results]
    if len(present) >= 2:
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        values = [100 * result_metric(name, results[name])["rate"] for name in present]
        ax.bar([label(name) for name in present], values, color=["#6b7280", "#2563eb", "#0f766e"][: len(present)])
        ax.set_ylabel("Hidden all-pass (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Ceiling Hidden Success")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = FIGURES / "ceiling_hidden_success.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path.relative_to(ROOT)))

    names = [
        "graphir_pipeline_ceiling.json",
        "graphir_pipeline_no_trace_ceiling.json",
        "graphir_pipeline_shuffled_trace_ceiling.json",
    ]
    present = [name for name in names if name in results]
    if len(present) >= 2:
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        values = [100 * graph_metric(results[name], "repair_hidden_all")["rate"] for name in present]
        ax.bar([label(name) for name in present], values, color="#7c3aed")
        ax.set_ylabel("Pipeline hidden all-pass (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Trace Controls")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = FIGURES / "trace_controls.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path.relative_to(ROOT)))

    names = ["dsl_lora_ceiling.json", "graphir_construct_ceiling.json", "graphir_pipeline_ceiling.json"]
    present = [name for name in names if name in results]
    if present:
        families = manifest["ceiling_families"]
        fig, ax = plt.subplots(figsize=(11, 5.8))
        width = 0.24
        x = list(range(len(families)))
        for offset, name in enumerate(present):
            values = []
            for family in families:
                fam = results[name]["summary"]["by_family"][family]
                if "construct_adapter" in results[name]:
                    values.append(100 * fam["repair_hidden_all"]["rate"])
                else:
                    values.append(100 * fam["rerank_hidden_all"]["rate"])
            ax.bar([item + (offset - 1) * width for item in x], values, width=width, label=label(name))
        ax.set_ylabel("Hidden all-pass (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Ceiling By Family")
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = FIGURES / "ceiling_by_family.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path.relative_to(ROOT)))

    return written


def main() -> None:
    manifest = load_json(ROOT / "data" / "dataset_manifest.json")
    eval_files = sorted((REPORTS / "eval").glob("*.json"))
    results = {path.name: load_json(path) for path in eval_files}
    chart_paths = write_charts(results, manifest)

    lines = [
        "# Qwen 3.5 4B GraphIR Self Repair",
        "",
        "## Question",
        "",
        "Can a Qwen 3.5 4B adapter improve held-out executable repair by configuring a typed register graph, then applying a verifier-guided repair step?",
        "",
        "## Design",
        "",
        "- Base model: `Qwen/Qwen3.5-4B`.",
        "- Training: 4-bit NF4 QLoRA adapters.",
        "- Fixed budget: 240 records per adapter.",
        "- DSL baseline: emits one prefix DSL expression.",
        "- GraphIR construct adapter: emits typed register assignments ending in `out`.",
        "- GraphIR repair adapter: receives a candidate graph plus visible execution mismatches and emits a corrected graph.",
        "- Inference policy: generate configured construction candidates, execute visible cases, keep the best graph, optionally repair it, and score hidden cases.",
        "- Large adapter/checkpoint files are stored outside the compact experiment directory.",
        "",
        "## Dataset",
        "",
        f"- Base train records: {manifest['base_train_records']}.",
        f"- Support bridge train records: {manifest['bridge_train_records']}.",
        f"- Train records per adapter: {manifest['train_records_per_adapter']}.",
        f"- IID eval records: {manifest['iid_eval_records']}.",
        f"- Support eval records: {manifest['support_eval_records']}.",
        f"- Ceiling eval records: {manifest['ceiling_eval_records']}.",
        f"- Visible cases per record: {manifest['visible_cases_per_record']}.",
        f"- Hidden cases per record: {manifest['hidden_cases_per_record']}.",
        f"- Support bridge families: {len(manifest['support_bridge_families'])}.",
        f"- Ceiling families: {len(manifest['ceiling_families'])}.",
    ]

    table(
        lines,
        "Ceiling Results",
        ["dsl_lora_ceiling.json", "graphir_construct_ceiling.json", "graphir_pipeline_ceiling.json"],
        results,
    )
    table(
        lines,
        "Support Results",
        ["dsl_lora_support.json", "graphir_construct_support.json", "graphir_pipeline_support.json"],
        results,
    )
    table(
        lines,
        "IID Results",
        ["dsl_lora_iid.json", "graphir_construct_iid.json", "graphir_pipeline_iid.json"],
        results,
    )
    if any(
        name in results
        for name in ["graphir_pipeline_no_trace_ceiling.json", "graphir_pipeline_shuffled_trace_ceiling.json"]
    ):
        table(
            lines,
            "Trace Controls",
            [
                "graphir_pipeline_ceiling.json",
                "graphir_pipeline_no_trace_ceiling.json",
                "graphir_pipeline_shuffled_trace_ceiling.json",
            ],
            results,
        )
    table(lines, "Repair Diagnostic", ["graphir_repair_corrupt_ceiling.json"], results)
    family_table(
        lines,
        "Ceiling By Family",
        ["dsl_lora_ceiling.json", "graphir_construct_ceiling.json", "graphir_pipeline_ceiling.json"],
        results,
        manifest["ceiling_families"],
    )

    dsl = results.get("dsl_lora_ceiling.json")
    construct = results.get("graphir_construct_ceiling.json")
    pipeline = results.get("graphir_pipeline_ceiling.json")
    diagnostic = results.get("graphir_repair_corrupt_ceiling.json")
    if dsl and construct and pipeline:
        lines.extend(
            [
                "",
                "## Readout",
                "",
                "- Ceiling hidden all-pass: DSL baseline "
                + pct(dsl_metric(dsl, "rerank_hidden_all"))
                + ", GraphIR construction "
                + pct(graph_metric(construct, "construct_rerank_hidden_all"))
                + ", GraphIR construction plus repair "
                + pct(graph_metric(pipeline, "repair_hidden_all"))
                + ".",
                "- GraphIR construction greedy ceiling hidden all-pass: "
                + pct(graph_metric(construct, "construct_greedy_hidden_all"))
                + ".",
                "- GraphIR pipeline construction-only selected ceiling hidden all-pass: "
                + pct(graph_metric(pipeline, "construct_rerank_hidden_all"))
                + ".",
                "- The GraphIR repair stage improved the actual ceiling pipeline from "
                + pct(graph_metric(pipeline, "construct_rerank_hidden_all"))
                + " to "
                + pct(graph_metric(pipeline, "repair_hidden_all"))
                + ", but did not beat the DSL baseline.",
            ]
        )
        if diagnostic:
            lines.append(
                "- On synthetic corrupted ceiling GraphIR candidates, repair improved hidden all-pass from "
                + pct(graph_metric(diagnostic, "input_hidden_all"))
                + " to "
                + pct(graph_metric(diagnostic, "repair_hidden_all"))
                + ", indicating repair skill exists but does not transfer enough to actual construction errors."
            )
    aligned = results.get("graphir_pipeline_ceiling.json")
    no_trace = results.get("graphir_pipeline_no_trace_ceiling.json")
    shuffled = results.get("graphir_pipeline_shuffled_trace_ceiling.json")
    if aligned and no_trace and shuffled:
        lines.append(
            "- Trace controls for the GraphIR pipeline: aligned "
            + pct(graph_metric(aligned, "repair_hidden_all"))
            + ", no trace "
            + pct(graph_metric(no_trace, "repair_hidden_all"))
            + ", shuffled trace "
            + pct(graph_metric(shuffled, "repair_hidden_all"))
            + "."
        )

    if chart_paths:
        lines.extend(["", "## Figures", ""])
        for path in chart_paths:
            lines.append(f"- `{path}`")

    if dsl and construct and pipeline:
        lines.extend(["", "## Failure Signatures", ""])
        for family in manifest["ceiling_families"]:
            lines.append(
                f"- `{family}` DSL top: {top_outputs(dsl, family, 'selected_program')}; "
                f"GraphIR construct top: {top_outputs(construct, family, 'construct_selected_graph', 1)}; "
                f"GraphIR pipeline top: {top_outputs(pipeline, family, 'repair_graph', 1)}."
            )

    if results:
        lines.extend(["", "## Per-Condition Details", ""])
        for path in eval_files:
            result = results[path.name]
            lines.extend(
                [
                    f"### {path.stem}",
                    "",
                    f"- Data: `{result['data']}`.",
                    f"- Prompt mode: `{result['prompt_mode']}`.",
                    f"- Samples: {result['num_samples']}.",
                    "",
                    "| Family | Main Hidden | Main Visible |",
                    "| --- | ---: | ---: |",
                ]
            )
            for family, summary in result["summary"]["by_family"].items():
                if "construct_adapter" in result:
                    hidden = summary["repair_hidden_all"]
                    visible = summary["repair_visible_all"]
                elif "candidate_graph" in (result.get("rows") or [{}])[0]:
                    hidden = summary["repair_hidden_all"]
                    visible = summary["repair_visible_all"]
                else:
                    hidden = summary["rerank_hidden_all"]
                    visible = summary["rerank_visible_all"]
                lines.append(f"| {family} | {pct(hidden)} | {pct(visible)} |")
            lines.append("")

    out = REPORTS / "qwen35_4b_graphir_self_repair_report.md"
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

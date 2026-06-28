#!/usr/bin/env python
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(metric: dict) -> str:
    return f"{100 * metric['rate']:.1f}% ({metric['successes']}/{metric['records']})"


def metric(result: dict, name: str = "rerank_hidden_all") -> dict:
    return result["summary"]["overall"][name]


def label(name: str) -> str:
    labels = {
        "seed_lora_support.json": "Seed, support",
        "static60_lora_support.json": "Static 60, support",
        "static80_lora_support.json": "Static 80, support",
        "seed_lora_ceiling.json": "Seed, ceiling",
        "static60_lora_ceiling.json": "Static 60, ceiling",
        "static80_lora_ceiling.json": "Static 80, ceiling",
        "static60_lora_no_trace_ceiling.json": "Static 60, no trace ceiling",
        "static60_lora_shuffled_trace_ceiling.json": "Static 60, shuffled trace ceiling",
        "static80_lora_no_trace_ceiling.json": "Static 80, no trace ceiling",
        "static80_lora_shuffled_trace_ceiling.json": "Static 80, shuffled trace ceiling",
        "seed_lora_iid.json": "Seed, IID",
        "static60_lora_iid.json": "Static 60, IID",
        "static80_lora_iid.json": "Static 80, IID",
    }
    return labels.get(name, name.removesuffix(".json").replace("_", " "))


def result_table(lines: list[str], title: str, names: list[str], results: dict[str, dict]) -> None:
    present = [name for name in names if name in results]
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
    for name in present:
        result = results[name]
        lines.append(
            "| "
            + label(name)
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
    lines.append("| Family | " + " | ".join(label(name) for name in present) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in present) + " |")
    for family in families:
        cells = []
        for name in present:
            fam = results[name]["summary"]["by_family"].get(family)
            cells.append(pct(fam["rerank_hidden_all"]) if fam else "n/a")
        lines.append("| `" + family + "` | " + " | ".join(cells) + " |")


def top_programs(result: dict, family: str, field: str = "greedy_program", limit: int = 2) -> str:
    counts = Counter(row[field] for row in result["rows"] if row["family"] == family)
    return "; ".join(f"{count}x `{program}`" for program, count in counts.most_common(limit)) or "none"


def write_charts(results: dict[str, dict], manifest: dict) -> list[str]:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    main_names = [
        "seed_lora_support.json",
        "static60_lora_support.json",
        "static80_lora_support.json",
        "seed_lora_ceiling.json",
        "static60_lora_ceiling.json",
        "static80_lora_ceiling.json",
    ]
    present = [name for name in main_names if name in results]
    if present:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        values = [100 * metric(results[name])["rate"] for name in present]
        ax.bar([label(name) for name in present], values, color=["#6b7280", "#2563eb", "#0f766e", "#9ca3af", "#3b82f6", "#14b8a6"][: len(present)])
        ax.set_ylabel("Rerank hidden all-pass (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Support and Ceiling Generalization")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = FIGURES / "support_ceiling_rerank_hidden.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path.relative_to(ROOT)))

    control_names = [
        "static60_lora_ceiling.json",
        "static60_lora_no_trace_ceiling.json",
        "static60_lora_shuffled_trace_ceiling.json",
    ]
    if (
        "static80_lora_no_trace_ceiling.json" in results
        or "static80_lora_shuffled_trace_ceiling.json" in results
    ):
        control_names.extend(
            [
                "static80_lora_ceiling.json",
                "static80_lora_no_trace_ceiling.json",
                "static80_lora_shuffled_trace_ceiling.json",
            ]
        )
    present = [name for name in control_names if name in results]
    if present:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        values = [100 * metric(results[name], "greedy_hidden_all")["rate"] for name in present]
        ax.bar([label(name) for name in present], values, color="#7c3aed")
        ax.set_ylabel("Greedy hidden all-pass (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Ceiling Trace Controls")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = FIGURES / "ceiling_trace_controls.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path.relative_to(ROOT)))

    family_names = [
        name for name in ["seed_lora_ceiling.json", "static60_lora_ceiling.json", "static80_lora_ceiling.json"] if name in results
    ]
    if family_names:
        families = manifest["ceiling_families"]
        fig, ax = plt.subplots(figsize=(11, 5.8))
        width = 0.24
        x = list(range(len(families)))
        for offset, name in enumerate(family_names):
            values = [
                100 * results[name]["summary"]["by_family"][family]["rerank_hidden_all"]["rate"]
                for family in families
            ]
            ax.bar([item + (offset - 1) * width for item in x], values, width=width, label=label(name))
        ax.set_ylabel("Rerank hidden all-pass (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Ceiling Split By Family")
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

    support_names = ["seed_lora_support.json", "static60_lora_support.json", "static80_lora_support.json"]
    ceiling_names = ["seed_lora_ceiling.json", "static60_lora_ceiling.json", "static80_lora_ceiling.json"]
    control_names = [
        "static60_lora_no_trace_ceiling.json",
        "static60_lora_shuffled_trace_ceiling.json",
        "static80_lora_no_trace_ceiling.json",
        "static80_lora_shuffled_trace_ceiling.json",
    ]
    iid_names = ["seed_lora_iid.json", "static60_lora_iid.json", "static80_lora_iid.json"]

    lines = [
        "# Qwen 3.5 4B Static Bridge Ceiling Breaker",
        "",
        "## Question",
        "",
        "Can fixed-budget static bridge posttraining learn a trace-conditioned repair interface that transfers from support bridge families to deeper held-out composition families?",
        "",
        "## Design",
        "",
        "- Base model: `Qwen/Qwen3.5-4B`.",
        "- Model output: one executable DSL expression.",
        "- Training: 4-bit NF4 QLoRA adapters.",
        "- Candidate selection: choose the valid candidate with the most visible-case passes.",
        "- Large adapter/checkpoint files are stored outside the compact experiment directory.",
        "- Seed adapter: 240 base-family random-trace records.",
        "- Static 60 adapter: 180 base-family records plus 60 equal support bridge records.",
        "- Static 80 adapter: 160 base-family records plus 80 equal support bridge records.",
        "- Main test: held-out ceiling families absent from bridge training.",
        "",
        "## Dataset",
        "",
        f"- Seed train records: {manifest['seed_train_records']}.",
        f"- Static 60 train records: {manifest['static60_train_records']} ({manifest['static60_base_records']} base + {manifest['static60_bridge_records']} bridge).",
        f"- Static 80 train records: {manifest['static80_train_records']} ({manifest['static80_base_records']} base + {manifest['static80_bridge_records']} bridge).",
        f"- IID eval records: {manifest['iid_eval_records']}.",
        f"- Support eval records: {manifest['support_eval_records']}.",
        f"- Ceiling eval records: {manifest['ceiling_eval_records']}.",
        f"- Visible cases per record: {manifest['visible_cases_per_record']}.",
        f"- Hidden cases per record: {manifest['hidden_cases_per_record']}.",
        f"- Support bridge families: {len(manifest['support_bridge_families'])}.",
        f"- Ceiling families: {len(manifest['ceiling_families'])}.",
        f"- Static 60 selector summary: `{manifest['static60_selector_summary']}`.",
        f"- Static 80 selector summary: `{manifest['static80_selector_summary']}`.",
    ]

    result_table(lines, "Support Split Results", support_names, results)
    result_table(lines, "Ceiling Split Results", ceiling_names, results)
    result_table(lines, "Ceiling Trace Controls", control_names, results)
    result_table(lines, "IID Retention Results", iid_names, results)
    family_table(lines, "Ceiling By Family", ceiling_names, results, manifest["ceiling_families"])
    family_table(lines, "Support By Family", support_names, results, manifest["support_bridge_families"])

    seed = results.get("seed_lora_ceiling.json")
    static60 = results.get("static60_lora_ceiling.json")
    static80 = results.get("static80_lora_ceiling.json")
    if seed and static60 and static80:
        lines.extend(
            [
                "",
                "## Readout",
                "",
                "- Ceiling reranked hidden all-pass: seed "
                + pct(metric(seed))
                + ", static 60 "
                + pct(metric(static60))
                + ", static 80 "
                + pct(metric(static80))
                + ".",
                "- Ceiling greedy hidden all-pass: seed "
                + pct(metric(seed, "greedy_hidden_all"))
                + ", static 60 "
                + pct(metric(static60, "greedy_hidden_all"))
                + ", static 80 "
                + pct(metric(static80, "greedy_hidden_all"))
                + ".",
            ]
        )
    for name in ["static60_lora_ceiling.json", "static80_lora_ceiling.json"]:
        no_trace = results.get(name.replace("_ceiling.json", "_no_trace_ceiling.json"))
        shuffled = results.get(name.replace("_ceiling.json", "_shuffled_trace_ceiling.json"))
        aligned = results.get(name)
        if aligned and no_trace and shuffled:
            lines.append(
                "- "
                + label(name)
                + " trace controls: aligned "
                + pct(metric(aligned, "greedy_hidden_all"))
                + ", no trace "
                + pct(metric(no_trace, "greedy_hidden_all"))
                + ", shuffled trace "
                + pct(metric(shuffled, "greedy_hidden_all"))
                + "."
            )

    if chart_paths:
        lines.extend(["", "## Figures", ""])
        for path in chart_paths:
            lines.append(f"- `{path}`")

    if seed and static60 and static80:
        lines.extend(["", "## Failure Signatures", ""])
        for family in manifest["ceiling_families"]:
            lines.append(
                f"- `{family}` seed greedy top: {top_programs(seed, family)}; "
                f"static 60 greedy top: {top_programs(static60, family)}; "
                f"static 80 greedy top: {top_programs(static80, family)}."
            )

    if results:
        lines.extend(["", "## Per-Condition Details", ""])
        for path in eval_files:
            result = results[path.name]
            lines.extend(
                [
                    f"### {path.stem}",
                    "",
                    f"- Adapter: `{result.get('adapter')}`.",
                    f"- Data: `{result['data']}`.",
                    f"- Prompt mode: `{result['prompt_mode']}`.",
                    f"- Samples: {result['num_samples']}.",
                    f"- Greedy hidden all-pass: {pct(result['summary']['overall']['greedy_hidden_all'])}.",
                    f"- Rerank hidden all-pass: {pct(result['summary']['overall']['rerank_hidden_all'])}.",
                    "",
                    "| Family | Greedy Hidden | Rerank Hidden | Greedy Visible | Rerank Visible |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for family, summary in result["summary"]["by_family"].items():
                lines.append(
                    "| "
                    + family
                    + " | "
                    + pct(summary["greedy_hidden_all"])
                    + " | "
                    + pct(summary["rerank_hidden_all"])
                    + " | "
                    + pct(summary["greedy_visible_all"])
                    + " | "
                    + pct(summary["rerank_visible_all"])
                    + " |"
                )
            lines.append("")

    out = REPORTS / "qwen35_4b_static_bridge_ceiling_breaker_report.md"
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

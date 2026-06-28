#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = Path(__file__).resolve().parents[3]

SPLIT_ORDER = ["singleton_iid", "composite_iid", "recombination_holdout"]
CORE_ORDER = [
    "frozen_trace",
    "singletons_trace",
    "composites_trace",
    "mixed_trace",
    "mixed_no_trace",
    "mixed_shuffled_trace",
]
ABLATION_ORDER = [
    "mixed_trace",
    "mixed_trace_no_trace_prompt",
    "mixed_trace_shuffled_trace_prompt",
]

DISPLAY_NAMES = {
    "singleton_iid": "Singleton IID",
    "composite_iid": "Composite IID",
    "recombination_holdout": "Recombination Holdout",
    "frozen_trace": "Frozen trace",
    "singletons_trace": "Singleton factors, trace",
    "composites_trace": "Composite factors, trace",
    "mixed_trace": "Mixed factors, trace",
    "mixed_no_trace": "Mixed factors, no trace train/eval",
    "mixed_shuffled_trace": "Mixed factors, shuffled trace train",
    "mixed_trace_no_trace_prompt": "Mixed trace adapter, no trace prompt",
    "mixed_trace_shuffled_trace_prompt": "Mixed trace adapter, shuffled trace prompt",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(results_dir: Path) -> list[dict[str, Any]]:
    manifest_path = results_dir / "final_evaluation_jobs.json"
    if manifest_path.exists():
        return load_json(manifest_path).get("jobs", [])

    jobs = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "final_evaluation_jobs.json":
            continue
        parts = path.stem.split("__")
        if len(parts) != 3:
            continue
        suite, name, split = parts
        jobs.append({"suite": suite, "name": name, "split": split, "output": str(path)})
    return jobs


def result_rows(results_dir: Path, allow_missing: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = load_manifest(results_dir)
    rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    missing = []
    for job in jobs:
        output = Path(str(job["output"]))
        if not output.exists():
            missing.append(output)
            continue
        payload = load_json(output)
        summary = payload["summary"]
        row = {
            "suite": job.get("suite"),
            "name": job.get("name"),
            "split": job.get("split"),
            "condition": summary.get("condition"),
            "adapter": summary.get("adapter"),
            "records": summary.get("records", 0),
            "successes": summary.get("successes", 0),
            "failures": summary.get("failures", 0),
            "repair_at_1": summary.get("repair@1", 0.0),
            "visible_pass_rate": summary.get("visible_pass_rate", 0.0),
            "hidden_pass_rate": summary.get("hidden_pass_rate", 0.0),
            "patch_apply_rate": summary.get("patch_apply_rate", 0.0),
            "syntax_valid_rate": summary.get("syntax_valid_rate", 0.0),
            "target_added_line_match_rate": summary.get("target_added_line_match_rate", 0.0),
            "target_marker_presence_rate": summary.get("target_marker_presence_rate", 0.0),
            "visible_input_literal_rate": summary.get("visible_input_literal_rate", 0.0),
            "median_generation_seconds": summary.get("median_generation_seconds", 0.0),
            "output": str(output),
        }
        rows.append(row)
        for family, family_summary in summary.get("by_family", {}).items():
            family_rows.append(
                {
                    "suite": job.get("suite"),
                    "name": job.get("name"),
                    "split": job.get("split"),
                    "bug_family": family,
                    "records": family_summary.get("records", 0),
                    "successes": family_summary.get("successes", 0),
                    "repair_at_1": family_summary.get("repair@1", 0.0),
                    "visible_pass_rate": family_summary.get("visible_pass_rate", 0.0),
                    "hidden_pass_rate": family_summary.get("hidden_pass_rate", 0.0),
                    "patch_apply_rate": family_summary.get("patch_apply_rate", 0.0),
                    "target_marker_presence_rate": family_summary.get("target_marker_presence_rate", 0.0),
                }
            )
        for factor, factor_summary in summary.get("by_factor", {}).items():
            factor_rows.append(
                {
                    "suite": job.get("suite"),
                    "name": job.get("name"),
                    "split": job.get("split"),
                    "factor": factor,
                    "records": factor_summary.get("records", 0),
                    "successes": factor_summary.get("successes", 0),
                    "repair_at_1": factor_summary.get("repair@1", 0.0),
                    "visible_pass_rate": factor_summary.get("visible_pass_rate", 0.0),
                    "hidden_pass_rate": factor_summary.get("hidden_pass_rate", 0.0),
                    "patch_apply_rate": factor_summary.get("patch_apply_rate", 0.0),
                    "target_marker_presence_rate": factor_summary.get("target_marker_presence_rate", 0.0),
                }
            )
    if missing and not allow_missing:
        missing_list = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing final result files:\n{missing_list}")
    return rows, family_rows, factor_rows


def rate(row: dict[str, Any] | None, key: str = "repair_at_1") -> float:
    if not row:
        return 0.0
    return float(row.get(key) or 0.0)


def pct(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def success_text(row: dict[str, Any] | None) -> str:
    if not row:
        return "missing"
    return f"{int(row.get('successes', 0))}/{int(row.get('records', 0))}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, int, str]:
        name = str(row.get("name"))
        split = str(row.get("split"))
        if name in CORE_ORDER:
            condition_index = CORE_ORDER.index(name)
        elif name in ABLATION_ORDER:
            condition_index = 100 + ABLATION_ORDER.index(name)
        else:
            condition_index = 999
        split_index = SPLIT_ORDER.index(split) if split in SPLIT_ORDER else 99
        return condition_index, split_index, name

    return sorted(rows, key=key)


def table_for(rows: list[dict[str, Any]], names: list[str]) -> str:
    lookup = {(row["name"], row["split"]): row for row in rows}
    header = ["Condition", *[DISPLAY_NAMES[split] for split in SPLIT_ORDER]]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for name in names:
        cells = [DISPLAY_NAMES.get(name, name)]
        for split in SPLIT_ORDER:
            row = lookup.get((name, split))
            if not row:
                cells.append("missing")
            else:
                cells.append(f"{pct(rate(row))} ({success_text(row)})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def recombination_family_table(family_rows: list[dict[str, Any]], names: list[str]) -> str:
    families = sorted(
        {
            row["bug_family"]
            for row in family_rows
            if row.get("split") == "recombination_holdout" and row.get("name") in names
        }
    )
    header = ["Family", *[DISPLAY_NAMES.get(name, name) for name in names]]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lookup = {(row["bug_family"], row["name"]): row for row in family_rows if row.get("split") == "recombination_holdout"}
    for family in families:
        cells = [family]
        for name in names:
            row = lookup.get((family, name))
            cells.append(f"{pct(rate(row))} ({success_text(row)})" if row else "missing")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mixed_factor_table(factor_rows: list[dict[str, Any]]) -> str:
    rows = [
        row
        for row in factor_rows
        if row.get("name") == "mixed_trace" and row.get("split") == "recombination_holdout"
    ]
    rows = sorted(rows, key=lambda row: str(row.get("factor")))
    lines = ["| Factor | repair@1 |", "| --- | --- |"]
    for row in rows:
        lines.append(f"| {row['factor']} | {pct(rate(row))} ({success_text(row)}) |")
    return "\n".join(lines)


def maybe_plot(rows: list[dict[str, Any]], family_rows: list[dict[str, Any]], figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        note = figures_dir / "plotting_skipped.txt"
        note.write_text("matplotlib is not installed; figures were not generated.\n", encoding="utf-8")
        return [str(note)]

    outputs: list[str] = []
    colors = ["#3B6EA8", "#D85C33", "#2E8B57", "#7A4EA3", "#A65E2E", "#4C8C8A"]

    core_rows = [row for row in rows if row.get("suite") == "core" and row.get("name") in CORE_ORDER]
    lookup = {(row["name"], row["split"]): row for row in core_rows}
    x = list(range(len(SPLIT_ORDER)))
    width = 0.12
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for index, name in enumerate(CORE_ORDER):
        offsets = [value + (index - (len(CORE_ORDER) - 1) / 2) * width for value in x]
        values = [rate(lookup.get((name, split))) for split in SPLIT_ORDER]
        ax.bar(offsets, values, width=width, label=DISPLAY_NAMES[name], color=colors[index % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[split] for split in SPLIT_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("repair@1")
    ax.legend(fontsize=8, ncols=2)
    ax.set_title("Final repair rates by condition and split")
    fig.tight_layout()
    path = figures_dir / "final_repair_by_condition_split.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path))

    target_rows = [
        row
        for row in family_rows
        if row.get("split") == "recombination_holdout" and row.get("name") in {"singletons_trace", "composites_trace", "mixed_trace"}
    ]
    families = sorted({row["bug_family"] for row in target_rows})
    names = ["singletons_trace", "composites_trace", "mixed_trace"]
    if families:
        fig, ax = plt.subplots(figsize=(11.0, 4.8))
        width = 0.24
        x = list(range(len(families)))
        lookup_family = {(row["bug_family"], row["name"]): row for row in target_rows}
        for index, name in enumerate(names):
            offsets = [value + (index - 1) * width for value in x]
            values = [rate(lookup_family.get((family, name))) for family in families]
            ax.bar(offsets, values, width=width, label=DISPLAY_NAMES[name], color=colors[index])
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("repair@1")
        ax.legend(fontsize=8)
        ax.set_title("Recombination holdout by family")
        fig.tight_layout()
        path = figures_dir / "recombination_holdout_by_family.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(str(path))

    return outputs


def bytes_text(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except FileNotFoundError:
                pass
    return total


def refresh_large_artifacts_manifest() -> None:
    base = WORKSPACE_DIR / "large_artifacts" / "feature_factorized_rule_diversity"
    model_dir = base / "models"
    lines = [
        "# Large Artifacts Manifest",
        "",
        "Large files for this experiment are intentionally stored outside the downloadable experiment package.",
        "",
        "| Path | Contents | Size |",
        "| --- | --- | --- |",
    ]
    for path in sorted(model_dir.glob("*_lora")):
        lines.append(
            f"| `{path.relative_to(WORKSPACE_DIR)}` | LoRA adapter, checkpoints, optimizer state, tokenizer files | {bytes_text(directory_size(path))} |"
        )
    lines.extend(
        [
            "",
            "Download `experiments/feature_factorized_rule_diversity/` for compact experiment artifacts.",
            "Download directories under `large_artifacts/feature_factorized_rule_diversity/models/` only when adapter weights or checkpoints are needed.",
            "",
        ]
    )
    (EXPERIMENT_DIR / "large_artifacts_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(
    report_path: Path,
    rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    factor_rows: list[dict[str, Any]],
    figures: list[str],
) -> None:
    lookup = {(row["name"], row["split"]): row for row in rows}
    holdout_rows = [
        row
        for row in rows
        if row.get("suite") == "core" and row.get("split") == "recombination_holdout" and row.get("name") in CORE_ORDER
    ]
    best_holdout = max(holdout_rows, key=lambda row: (rate(row), int(row.get("successes", 0))), default=None)
    mixed = lookup.get(("mixed_trace", "recombination_holdout"))
    no_trace = lookup.get(("mixed_no_trace", "recombination_holdout"))
    shuffled = lookup.get(("mixed_shuffled_trace", "recombination_holdout"))

    lines = [
        "# Feature-Factorized Rule Diversity",
        "",
        "## Question",
        "",
        "Does held-out rule repair improve more when training examples cover isolated primitive factors, analogous multi-factor compositions, or a fixed-budget mixture of both?",
        "",
        "## Design",
        "",
        "- Three trace-trained adapters use the same 240-record budget: singleton factors only, composite factors only, and a mixed singleton/composite allocation.",
        "- Two controls use the mixed allocation with traces removed or trace outputs shuffled during training.",
        "- Final evaluation uses singleton IID, composite IID, and recombination holdout splits.",
        "- Checkpoints and adapter weights are stored outside this experiment directory under `large_artifacts/feature_factorized_rule_diversity/models/`.",
        "",
        "## Overall Results",
        "",
        table_for(rows, CORE_ORDER),
        "",
        "## Prompt Ablations",
        "",
        table_for(rows, ABLATION_ORDER),
        "",
        "## Recombination Holdout By Family",
        "",
        recombination_family_table(family_rows, ["singletons_trace", "composites_trace", "mixed_trace"]),
        "",
        "## Mixed Trace Holdout By Factor",
        "",
        mixed_factor_table(factor_rows),
        "",
        "## Readout",
        "",
    ]
    if best_holdout:
        lines.append(
            f"- Best core recombination result: {DISPLAY_NAMES.get(str(best_holdout['name']), best_holdout['name'])} at "
            f"{pct(rate(best_holdout))} ({success_text(best_holdout)})."
        )
    if mixed and no_trace and shuffled:
        lines.append(
            f"- Mixed trace vs controls on recombination: trace {pct(rate(mixed))}, "
            f"no trace {pct(rate(no_trace))}, shuffled-trace train {pct(rate(shuffled))}."
        )
    if figures:
        lines.extend(["", "## Figures", ""])
        for path in figures:
            lines.append(f"- `{Path(path).relative_to(WORKSPACE_DIR)}`")
    lines.extend(["", "## Artifact Layout", ""])
    lines.extend(
        [
            "- Compact artifacts: `experiments/feature_factorized_rule_diversity/`.",
            "- Large artifacts: `large_artifacts/feature_factorized_rule_diversity/`.",
            "- Dataset manifest: `experiments/feature_factorized_rule_diversity/data/dataset_manifest.json`.",
            "- Evaluation manifest: `experiments/feature_factorized_rule_diversity/reports/final/final_evaluation_jobs.json`.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT_DIR / "reports" / "final")
    parser.add_argument("--report-dir", type=Path, default=EXPERIMENT_DIR / "reports")
    parser.add_argument("--figures-dir", type=Path, default=EXPERIMENT_DIR / "figures")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    rows, family_rows, factor_rows = result_rows(args.results_dir, args.allow_missing)
    rows = sort_rows(rows)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / "final_results.csv", list(rows[0]) if rows else [], rows)
    write_csv(args.report_dir / "final_results_by_family.csv", list(family_rows[0]) if family_rows else [], family_rows)
    write_csv(args.report_dir / "final_results_by_factor.csv", list(factor_rows[0]) if factor_rows else [], factor_rows)
    figures = maybe_plot(rows, family_rows, args.figures_dir)
    refresh_large_artifacts_manifest()
    report_path = args.report_dir / "feature_factorized_rule_diversity_report.md"
    write_report(report_path, rows, family_rows, factor_rows, figures)
    print(str(report_path))


if __name__ == "__main__":
    main()

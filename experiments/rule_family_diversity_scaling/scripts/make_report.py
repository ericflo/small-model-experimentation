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

SPLIT_ORDER = ["base_iid", "format_holdout", "rule_holdout"]
CORE_ORDER = [
    "frozen_trace",
    "scale3_trace",
    "scale6_trace",
    "scale12_trace",
    "scale12_no_trace",
    "scale12_shuffled_trace",
]
SCALE_ORDER = ["scale3_trace", "scale6_trace", "scale12_trace"]
ABLATION_ORDER = [
    "scale12_trace",
    "scale12_trace_no_trace_prompt",
    "scale12_trace_shuffled_trace_prompt",
]

DISPLAY_NAMES = {
    "base_iid": "Base IID",
    "format_holdout": "Format Holdout",
    "rule_holdout": "Rule Holdout",
    "frozen_trace": "Frozen trace",
    "scale3_trace": "3 families, trace",
    "scale6_trace": "6 families, trace",
    "scale12_trace": "12 families, trace",
    "scale12_no_trace": "12 families, no trace train/eval",
    "scale12_shuffled_trace": "12 families, shuffled trace train",
    "scale12_trace_no_trace_prompt": "12-family trace adapter, no trace prompt",
    "scale12_trace_shuffled_trace_prompt": "12-family trace adapter, shuffled trace prompt",
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


def result_rows(results_dir: Path, allow_missing: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = load_manifest(results_dir)
    rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
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
    if missing and not allow_missing:
        missing_list = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing final result files:\n{missing_list}")
    return rows, family_rows


def rate(row: dict[str, Any], key: str = "repair_at_1") -> float:
    return float(row.get(key) or 0.0)


def pct(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def success_text(row: dict[str, Any]) -> str:
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
        condition_index = CORE_ORDER.index(name) if name in CORE_ORDER else 100 + ABLATION_ORDER.index(name)
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


def best_scale_row(rows: list[dict[str, Any]], split: str) -> dict[str, Any] | None:
    scale_rows = [row for row in rows if row.get("suite") == "core" and row.get("name") in SCALE_ORDER and row.get("split") == split]
    return max(scale_rows, key=lambda row: (rate(row), int(row.get("successes", 0))), default=None)


def control_comparison(rows: list[dict[str, Any]], split: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    lookup = {(row["name"], row["split"]): row for row in rows}
    return (
        lookup.get(("scale12_trace", split)),
        lookup.get(("scale12_no_trace", split)),
        lookup.get(("scale12_shuffled_trace", split)),
    )


def write_pilot_csv(report_dir: Path) -> None:
    pilot_path = EXPERIMENT_DIR / "reports" / "frozen_trace_base_iid_pilot6.json"
    if not pilot_path.exists():
        return
    summary = load_json(pilot_path)["summary"]
    row = {
        "name": "frozen_trace_base_iid_pilot6",
        "records": summary.get("records", 0),
        "successes": summary.get("successes", 0),
        "repair_at_1": summary.get("repair@1", 0.0),
        "visible_pass_rate": summary.get("visible_pass_rate", 0.0),
        "hidden_pass_rate": summary.get("hidden_pass_rate", 0.0),
        "patch_apply_rate": summary.get("patch_apply_rate", 0.0),
    }
    write_csv(report_dir / "pilot_results.csv", list(row), [row])


def maybe_plot(rows: list[dict[str, Any]], figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        note = figures_dir / "plotting_skipped.txt"
        note.write_text("matplotlib is not installed; figures were not generated.\n", encoding="utf-8")
        return [str(note)]

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    outputs: list[str] = []

    core_rows = [row for row in rows if row.get("suite") == "core" and row.get("name") in CORE_ORDER]
    lookup = {(row["name"], row["split"]): row for row in core_rows}
    x = list(range(len(SPLIT_ORDER)))
    width = 0.12
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for index, name in enumerate(CORE_ORDER):
        offsets = [value + (index - (len(CORE_ORDER) - 1) / 2) * width for value in x]
        values = [rate(lookup.get((name, split), {})) for split in SPLIT_ORDER]
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

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    scale_x = [3, 6, 12]
    for split, color in zip(SPLIT_ORDER, ["#4C78A8", "#F58518", "#54A24B"]):
        values = [rate(lookup.get((name, split), {})) for name in SCALE_ORDER]
        ax.plot(scale_x, values, marker="o", label=DISPLAY_NAMES[split], color=color)
    ax.set_xticks(scale_x)
    ax.set_xlabel("Training rule families")
    ax.set_ylim(0, 1)
    ax.set_ylabel("repair@1")
    ax.legend()
    ax.set_title("Diversity scale curve")
    fig.tight_layout()
    path = figures_dir / "diversity_scale_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path))

    ablation_rows = [row for row in rows if row.get("name") in ABLATION_ORDER]
    ablation_lookup = {(row["name"], row["split"]): row for row in ablation_rows}
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    width = 0.22
    for index, name in enumerate(ABLATION_ORDER):
        offsets = [value + (index - 1) * width for value in x]
        values = [rate(ablation_lookup.get((name, split), {})) for split in SPLIT_ORDER]
        ax.bar(offsets, values, width=width, label=DISPLAY_NAMES[name], color=colors[index % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[split] for split in SPLIT_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("repair@1")
    ax.legend(fontsize=8)
    ax.set_title("Scale12 trace adapter prompt ablations")
    fig.tight_layout()
    path = figures_dir / "scale12_trace_ablation.png"
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
    base = WORKSPACE_DIR / "large_artifacts" / "rule_family_diversity_scaling"
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
        lines.append(f"| `{path.relative_to(WORKSPACE_DIR)}` | LoRA adapter, checkpoints, optimizer state, tokenizer files | {bytes_text(directory_size(path))} |")
    lines.extend(
        [
            "",
            "Download `experiments/rule_family_diversity_scaling/` for compact experiment artifacts.",
            "Download directories under `large_artifacts/rule_family_diversity_scaling/models/` only when adapter weights or checkpoints are needed.",
            "",
        ]
    )
    (EXPERIMENT_DIR / "large_artifacts_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def write_markdown_reports(rows: list[dict[str, Any]], family_rows: list[dict[str, Any]], report_dir: Path, figure_paths: list[str]) -> None:
    manifest = load_json(EXPERIMENT_DIR / "data" / "dataset_manifest.json")
    core_rows = [row for row in rows if row.get("suite") == "core"]
    ablation_rows = [row for row in rows if row.get("suite") == "ablation"]
    all_ablation_rows = core_rows + ablation_rows

    best_rule = best_scale_row(core_rows, "rule_holdout")
    best_format = best_scale_row(core_rows, "format_holdout")
    best_base = best_scale_row(core_rows, "base_iid")
    trace_rule, no_trace_rule, shuffled_rule = control_comparison(core_rows, "rule_holdout")

    conclusion_lines = []
    if best_rule:
        conclusion_lines.append(
            f"- Best held-out rule-family repair among the diversity-scale adapters was `{best_rule['name']}` at {pct(rate(best_rule))} ({success_text(best_rule)})."
        )
    if best_format:
        conclusion_lines.append(
            f"- Best format-holdout repair among the diversity-scale adapters was `{best_format['name']}` at {pct(rate(best_format))} ({success_text(best_format)})."
        )
    if best_base:
        conclusion_lines.append(
            f"- Best base-IID repair among the diversity-scale adapters was `{best_base['name']}` at {pct(rate(best_base))} ({success_text(best_base)})."
        )
    if trace_rule and no_trace_rule and shuffled_rule:
        conclusion_lines.append(
            f"- On held-out rule families, the 12-family trace adapter scored {pct(rate(trace_rule))}, while the no-trace control scored {pct(rate(no_trace_rule))} and the shuffled-trace-trained control scored {pct(rate(shuffled_rule))}."
        )
    if not conclusion_lines:
        conclusion_lines.append("- Final conclusions are unavailable because one or more result files are missing.")

    figure_lines = [f"- `{Path(path).relative_to(WORKSPACE_DIR)}`" for path in figure_paths]
    if not figure_lines:
        figure_lines = ["- No figures were generated."]

    family_focus = [
        row
        for row in family_rows
        if row.get("suite") == "core" and row.get("name") in SCALE_ORDER and row.get("split") == "rule_holdout"
    ]
    family_focus = sorted(
        family_focus,
        key=lambda row: (
            SCALE_ORDER.index(str(row["name"])) if row.get("name") in SCALE_ORDER else 99,
            str(row.get("bug_family")),
        ),
    )
    family_table_lines = ["| Condition | Family | repair@1 | Successes |", "| --- | --- | --- | --- |"]
    for row in family_focus:
        family_table_lines.append(
            f"| {DISPLAY_NAMES.get(str(row['name']), row['name'])} | `{row['bug_family']}` | {pct(rate(row))} | {success_text(row)} |"
        )

    paper = f"""# Rule-Family Diversity Scaling

Date: 2026-06-21

## Abstract

This experiment tests whether increasing rule-family diversity in trace-conditioned repair training improves transfer to unseen rule structures when the total number of training records is fixed. Three trace-conditioned LoRA adapters were trained on 240 examples each, using 3, 6, or 12 rule families. The evaluation separates base-IID repair, format holdout, and fully held-out rule-family transfer. Controls test frozen-model behavior, no-trace training, shuffled-trace training, and prompt-time trace ablations.

## Design

- Base model: `Qwen/Qwen2.5-Coder-3B-Instruct`.
- Revision: `488639f1ff808d1d3d0ba301aef8c11461451ec5`.
- Training method: QLoRA, rank 32, alpha 64, dropout 0.05.
- Training budget: 240 records per diversity scale, 3 epochs.
- Evaluation metric: `repair@1`, requiring both visible and hidden tests to pass.
- Validation splits: 36 base-IID records, 36 format-holdout records, and 48 held-out rule-family records.

Training family counts:

- 3-family scale: 80 records per family.
- 6-family scale: 40 records per family.
- 12-family scale: 20 records per family.

Held-out rule families:

{chr(10).join(f"- `{family}`" for family in manifest["holdout_families"])}

## Core Results

{table_for(core_rows, CORE_ORDER)}

## Diversity Scale Results

{table_for(core_rows, SCALE_ORDER)}

## Trace Control And Ablation Results

{table_for(all_ablation_rows, ABLATION_ORDER)}

## Held-Out Rule Results By Family

{chr(10).join(family_table_lines)}

## Interpretation

{chr(10).join(conclusion_lines)}

The central comparison is the diversity-scale curve on the rule-holdout split. A useful positive result is not simply high base-IID repair; it is improved rule-holdout repair under the same 240-record training budget. The control rows indicate whether any held-out repair depends on valid trace evidence or can be explained by patch priors learned from the training distribution.

## Figures

{chr(10).join(figure_lines)}

## Artifacts

- Dataset manifest: `experiments/rule_family_diversity_scaling/data/dataset_manifest.json`.
- Final JSON results: `experiments/rule_family_diversity_scaling/reports/final/`.
- CSV summaries: `experiments/rule_family_diversity_scaling/reports/*.csv`.
- Figures: `experiments/rule_family_diversity_scaling/figures/`.
- Large adapter artifacts: `large_artifacts/rule_family_diversity_scaling/models/`.
"""

    summary = f"""# Rule-Family Diversity Scaling Summary

{chr(10).join(conclusion_lines)}

Core repair@1:

{table_for(core_rows, CORE_ORDER)}

The compact experiment package is `experiments/rule_family_diversity_scaling/`. Adapter weights and checkpoints are split out under `large_artifacts/rule_family_diversity_scaling/models/`.
"""

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "rule_family_diversity_scaling_paper.md").write_text(paper, encoding="utf-8")
    (report_dir / "rule_family_diversity_scaling_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT_DIR / "reports" / "final")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    report_dir = EXPERIMENT_DIR / "reports"
    figures_dir = EXPERIMENT_DIR / "figures"
    rows, family_rows = result_rows(args.results_dir, args.allow_missing)
    rows = sort_rows(rows)

    core_rows = [row for row in rows if row.get("suite") == "core"]
    ablation_rows = [row for row in rows if row.get("suite") == "ablation"]
    scale_rows = [row for row in core_rows if row.get("name") in SCALE_ORDER]

    fields = [
        "suite",
        "name",
        "split",
        "condition",
        "records",
        "successes",
        "failures",
        "repair_at_1",
        "visible_pass_rate",
        "hidden_pass_rate",
        "patch_apply_rate",
        "syntax_valid_rate",
        "target_added_line_match_rate",
        "target_marker_presence_rate",
        "visible_input_literal_rate",
        "median_generation_seconds",
        "adapter",
        "output",
    ]
    write_csv(report_dir / "final_core_results.csv", fields, core_rows)
    write_csv(report_dir / "final_ablation_results.csv", fields, ablation_rows)
    write_csv(report_dir / "final_scale_by_split.csv", fields, scale_rows)
    write_csv(
        report_dir / "final_trace_by_family.csv",
        [
            "suite",
            "name",
            "split",
            "bug_family",
            "records",
            "successes",
            "repair_at_1",
            "visible_pass_rate",
            "hidden_pass_rate",
            "patch_apply_rate",
            "target_marker_presence_rate",
        ],
        family_rows,
    )
    write_pilot_csv(report_dir)
    figure_paths = maybe_plot(rows, figures_dir)
    refresh_large_artifacts_manifest()
    write_markdown_reports(rows, family_rows, report_dir, figure_paths)

    print(f"Wrote reports to {report_dir}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = Path(__file__).resolve().parents[3]
LARGE_DIR = WORKSPACE_DIR / "large_artifacts" / "factor_recombination_ladder"

SPLIT_ORDER = ["seen_iid", "format_shift", "recombination_holdout"]
CORE_ORDER = [
    "frozen_trace",
    "ladder_trace",
    "ladder_no_trace",
    "ladder_shuffled_trace",
    "labelled_trace",
]
ABLATION_ORDER = [
    "ladder_trace",
    "ladder_trace_no_trace_prompt",
    "ladder_trace_shuffled_trace_prompt",
    "labelled_trace",
    "labelled_trace_labels_removed",
    "labelled_trace_no_trace_prompt",
    "labelled_trace_shuffled_trace_prompt",
]

DISPLAY_NAMES = {
    "seen_iid": "Seen-Combination IID",
    "format_shift": "Format Shift",
    "recombination_holdout": "Recombination Holdout",
    "frozen_trace": "Frozen trace",
    "ladder_trace": "Trace ladder",
    "ladder_no_trace": "No-trace ladder",
    "ladder_shuffled_trace": "Shuffled-trace ladder",
    "labelled_trace": "Factor-labelled trace ladder",
    "ladder_trace_no_trace_prompt": "Trace ladder, no trace prompt",
    "ladder_trace_shuffled_trace_prompt": "Trace ladder, shuffled trace prompt",
    "labelled_trace_labels_removed": "Labelled adapter, labels removed",
    "labelled_trace_no_trace_prompt": "Labelled adapter, no trace prompt",
    "labelled_trace_shuffled_trace_prompt": "Labelled adapter, shuffled trace prompt",
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
        if len(parts) == 3:
            suite, name, split = parts
            jobs.append({"suite": suite, "name": name, "split": split, "output": str(path)})
    return jobs


def result_rows(results_dir: Path, allow_missing: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    missing: list[Path] = []
    for job in load_manifest(results_dir):
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
        raise FileNotFoundError("Missing final result files:\n" + "\n".join(str(path) for path in missing))
    return rows, family_rows, factor_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rate(row: dict[str, Any] | None) -> float:
    return float(row.get("repair_at_1") or 0.0) if row else 0.0


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def success_text(row: dict[str, Any] | None) -> str:
    if not row:
        return "missing"
    return f"{int(row.get('successes', 0))}/{int(row.get('records', 0))}"


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
            cells.append(f"{pct(rate(row))} ({success_text(row)})" if row else "missing")
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
    lookup = {(row["bug_family"], row["name"]): row for row in family_rows if row.get("split") == "recombination_holdout"}
    header = ["Family", *[DISPLAY_NAMES.get(name, name) for name in names]]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for family in families:
        cells = [family]
        for name in names:
            row = lookup.get((family, name))
            cells.append(f"{pct(rate(row))} ({success_text(row)})" if row else "missing")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def recombination_factor_table(factor_rows: list[dict[str, Any]], name: str) -> str:
    rows = [
        row
        for row in factor_rows
        if row.get("name") == name and row.get("split") == "recombination_holdout"
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
    colors = ["#2F6C9F", "#D95F45", "#3F8F5F", "#7B5BA7", "#A96B2D"]
    core_rows = [row for row in rows if row.get("suite") == "core" and row.get("name") in CORE_ORDER]
    lookup = {(row["name"], row["split"]): row for row in core_rows}
    x = list(range(len(SPLIT_ORDER)))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, name in enumerate(CORE_ORDER):
        values = [rate(lookup.get((name, split))) for split in SPLIT_ORDER]
        offsets = [pos + (idx - 2) * width for pos in x]
        ax.bar(offsets, values, width=width, label=DISPLAY_NAMES[name], color=colors[idx % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[split] for split in SPLIT_ORDER], rotation=10, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("repair@1")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Repair Rate by Condition and Split")
    fig.tight_layout()
    path = figures_dir / "final_repair_by_condition_split.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(str(path))

    families = sorted(
        {
            row["bug_family"]
            for row in family_rows
            if row.get("split") == "recombination_holdout" and row.get("name") in ["ladder_trace", "labelled_trace"]
        }
    )
    if families:
        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.32
        x = list(range(len(families)))
        for idx, name in enumerate(["ladder_trace", "labelled_trace"]):
            values = []
            for family in families:
                row = next(
                    (
                        item
                        for item in family_rows
                        if item.get("split") == "recombination_holdout"
                        and item.get("name") == name
                        and item.get("bug_family") == family
                    ),
                    None,
                )
                values.append(rate(row))
            offsets = [pos + (idx - 0.5) * width for pos in x]
            ax.bar(offsets, values, width=width, label=DISPLAY_NAMES[name], color=colors[idx])
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("repair@1")
        ax.set_title("Recombination Holdout by Family")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        path = figures_dir / "recombination_holdout_by_family.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def dir_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    if total >= 1024**3:
        return f"{total / 1024**3:.1f} GB"
    if total >= 1024**2:
        return f"{total / 1024**2:.1f} MB"
    return f"{total / 1024:.1f} KB"


def refresh_large_manifest() -> None:
    model_dir = LARGE_DIR / "models"
    rows = []
    if model_dir.exists():
        for path in sorted(item for item in model_dir.iterdir() if item.is_dir()):
            rows.append((path.relative_to(WORKSPACE_DIR), dir_size(path)))
    lines = [
        "# Large Artifacts Manifest",
        "",
        "Large files for this experiment are intentionally stored outside the downloadable experiment package.",
        "",
        "| Path | Contents | Size |",
        "| --- | --- | --- |",
    ]
    for path, size in rows:
        lines.append(f"| `{path}` | LoRA adapter, checkpoints, optimizer state, tokenizer files | {size} |")
    lines.extend(
        [
            "",
            "Download `experiments/factor_recombination_ladder/` for compact experiment artifacts.",
            "Download directories under `large_artifacts/factor_recombination_ladder/models/` only when adapter weights or checkpoints are needed.",
            "",
        ]
    )
    (EXPERIMENT_DIR / "large_artifacts_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT_DIR / "reports" / "final")
    parser.add_argument("--report-dir", type=Path, default=EXPERIMENT_DIR / "reports")
    parser.add_argument("--figures-dir", type=Path, default=EXPERIMENT_DIR / "figures")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    rows, family_rows, factor_rows = result_rows(args.results_dir, args.allow_missing)
    rows = sort_rows(rows)
    write_csv(args.report_dir / "final_results.csv", list(rows[0]) if rows else [], rows)
    write_csv(args.report_dir / "final_results_by_family.csv", list(family_rows[0]) if family_rows else [], family_rows)
    write_csv(args.report_dir / "final_results_by_factor.csv", list(factor_rows[0]) if factor_rows else [], factor_rows)
    figures = maybe_plot(rows, family_rows, args.figures_dir)
    refresh_large_manifest()

    core_lookup = {(row["name"], row["split"]): row for row in rows if row.get("suite") == "core"}
    best_holdout = max(
        [row for row in rows if row.get("suite") == "core" and row.get("split") == "recombination_holdout"],
        key=lambda row: rate(row),
        default=None,
    )
    trace_holdout = core_lookup.get(("ladder_trace", "recombination_holdout"))
    labelled_holdout = core_lookup.get(("labelled_trace", "recombination_holdout"))

    report = [
        "# Factor Recombination Ladder",
        "",
        "## Question",
        "",
        "Can trace-conditioned repair learn reusable factor recombination when specific factor-pair cells are held out from training?",
        "",
        "## Design",
        "",
        "- The training set contains 12 seen rule families with 240 total records per condition.",
        "- The recombination split contains five held-out factor-pair cells absent from training.",
        "- Core conditions compare frozen trace prompting, aligned trace training, no-trace training, shuffled-trace training, and factor-labelled trace training.",
        "- Prompt ablations test whether trained trace adapters depend on trace content and factor labels at inference time.",
        "- Checkpoints and adapter weights are stored outside this experiment directory under `large_artifacts/factor_recombination_ladder/models/`.",
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
        recombination_family_table(family_rows, ["ladder_trace", "labelled_trace", "ladder_no_trace", "ladder_shuffled_trace"]),
        "",
        "## Trace Ladder Holdout By Factor",
        "",
        recombination_factor_table(factor_rows, "ladder_trace"),
        "",
        "## Readout",
        "",
    ]
    if best_holdout:
        report.append(
            f"- Best core recombination result: {DISPLAY_NAMES.get(best_holdout['name'], best_holdout['name'])} at "
            f"{pct(rate(best_holdout))} ({success_text(best_holdout)})."
        )
    if trace_holdout and labelled_holdout:
        report.append(
            f"- Factor labels delta on recombination: trace {pct(rate(trace_holdout))} ({success_text(trace_holdout)}) "
            f"vs labelled trace {pct(rate(labelled_holdout))} ({success_text(labelled_holdout)})."
        )
    report.extend(
        [
            "",
            "## Figures",
            "",
            *[f"- `{Path(path).relative_to(WORKSPACE_DIR)}`" for path in figures],
            "",
            "## Artifact Layout",
            "",
            "- Compact artifacts: `experiments/factor_recombination_ladder/`.",
            "- Large artifacts: `large_artifacts/factor_recombination_ladder/`.",
            "- Dataset manifest: `experiments/factor_recombination_ladder/data/dataset_manifest.json`.",
            "- Evaluation manifest: `experiments/factor_recombination_ladder/reports/final/final_evaluation_jobs.json`.",
            "",
        ]
    )
    report_path = args.report_dir / "factor_recombination_ladder_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()

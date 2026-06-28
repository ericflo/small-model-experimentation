#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
LARGE = ROOT / "large_artifacts" / "trace_keyed_symbol_repair"
REPORTS = EXP / "reports"
FIGURES = EXP / "figures"
DATA = EXP / "data"

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"

CORE_SPECS = [
    ("IID", "Frozen base + trace", "final_frozen_trace_iid.json"),
    ("Format holdout", "Frozen base + trace", "final_frozen_trace_format_holdout.json"),
    ("IID", "Final-patch SFT + final patch", "final_final_patch_lora_final_patch_iid.json"),
    (
        "Format holdout",
        "Final-patch SFT + final patch",
        "final_final_patch_lora_final_patch_format_holdout.json",
    ),
    ("IID", "No-trace SFT + no trace", "final_no_trace_lora_no_trace_iid.json"),
    (
        "Format holdout",
        "No-trace SFT + no trace",
        "final_no_trace_lora_no_trace_format_holdout.json",
    ),
    ("IID", "Shuffled-trace SFT + trace", "final_shuffled_trace_lora_trace_iid.json"),
    (
        "Format holdout",
        "Shuffled-trace SFT + trace",
        "final_shuffled_trace_lora_trace_format_holdout.json",
    ),
    ("IID", "Trace SFT + trace", "final_trace_lora_trace_iid.json"),
    ("Format holdout", "Trace SFT + trace", "final_trace_lora_trace_format_holdout.json"),
]

ABLATION_SPECS = [
    ("IID", "Trace SFT + no trace", "final_trace_lora_no_trace_iid.json"),
    ("Format holdout", "Trace SFT + no trace", "final_trace_lora_no_trace_format_holdout.json"),
    ("IID", "Trace SFT + shuffled trace", "final_trace_lora_shuffled_trace_iid.json"),
    (
        "Format holdout",
        "Trace SFT + shuffled trace",
        "final_trace_lora_shuffled_trace_format_holdout.json",
    ),
]

PILOT_SPECS = [
    ("Frozen base + trace, 10 IID", "frozen_trace_iid_pilot10.json"),
    ("Pilot trace adapter + trace, 20 IID", "pilot_trace_iid20.json"),
    ("Pilot trace adapter + no trace, 20 IID", "pilot_trace_adapter_no_trace_iid20.json"),
    (
        "Pilot trace adapter + shuffled trace, 20 IID",
        "pilot_trace_adapter_shuffled_trace_iid20.json",
    ),
    ("Pilot trace adapter + trace, 5 IID, 64 tokens", "pilot_trace_iid5_max64.json"),
    ("Pilot trace adapter + trace, 5 IID, 128 tokens", "pilot_trace_iid5_max128.json"),
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_result_rows(specs: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    rows = []
    for split, label, filename in specs:
        payload = read_json(REPORTS / filename)
        summary = payload.get("summary", {}) if payload else {}
        rows.append(
            {
                **summary,
                "split": split,
                "condition": label,
                "eval_condition": summary.get("condition"),
                "file": filename,
                "status": "ok" if payload else "missing",
            }
        )
    return rows


def load_pilot_rows() -> list[dict[str, Any]]:
    rows = []
    for label, filename in PILOT_SPECS:
        payload = read_json(REPORTS / filename)
        summary = payload.get("summary", {}) if payload else {}
        rows.append(
            {
                **summary,
                "condition": label,
                "eval_condition": summary.get("condition"),
                "file": filename,
                "status": "ok" if payload else "missing",
            }
        )
    return rows


def metric(row: dict[str, Any], key: str) -> Any:
    return row.get(key)


def fmt_rate(value: Any) -> str:
    if value is None:
        return "missing"
    return f"{100 * float(value):.1f}%"


def fmt_count(row: dict[str, Any]) -> str:
    successes = row.get("successes")
    records = row.get("records")
    if successes is None or records is None:
        return "missing"
    return f"{successes}/{records}"


def esc(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(esc(item) for item in row) + " |")
    return "\n".join(lines)


def result_table(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        [
            "Split",
            "Condition",
            "Repair@1",
            "Visible pass",
            "Patch apply",
            "Expected-token copy",
            "Wrong-token removed",
            "Successes",
        ],
        [
            [
                row.get("split", ""),
                row["condition"],
                fmt_rate(metric(row, "repair@1")),
                fmt_rate(metric(row, "visible_pass_rate")),
                fmt_rate(metric(row, "patch_apply_rate")),
                fmt_rate(metric(row, "expected_token_copy_rate")),
                fmt_rate(metric(row, "wrong_token_removed_rate")),
                fmt_count(row),
            ]
            for row in rows
        ],
    )


def pilot_table(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        [
            "Condition",
            "Repair@1",
            "Patch apply",
            "Expected-token copy",
            "Max new tokens",
            "Successes",
        ],
        [
            [
                row["condition"],
                fmt_rate(metric(row, "repair@1")),
                fmt_rate(metric(row, "patch_apply_rate")),
                fmt_rate(metric(row, "expected_token_copy_rate")),
                row.get("max_new_tokens", "missing"),
                fmt_count(row),
            ]
            for row in rows
        ],
    )


def manifest_summary(manifest: dict[str, Any]) -> str:
    records = manifest.get("records", {})
    return "\n".join(
        [
            f"- Train records: `{records.get('train', 'missing')}`.",
            f"- IID validation records: `{records.get('val_iid', 'missing')}`.",
            f"- Format-holdout validation records: `{records.get('val_format_holdout', 'missing')}`.",
            f"- Train token styles: `{', '.join(manifest.get('train_token_styles', []))}`.",
            f"- Holdout token styles: `{', '.join(manifest.get('holdout_token_styles', []))}`.",
            f"- Dataset seed: `{manifest.get('seed', 'missing')}`.",
            f"- Invariant: {manifest.get('invariant', 'missing')}.",
        ]
    )


def lora_metadata_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((LARGE / "models").glob("*/experiment_metadata.json")):
        data = read_json(path) or {}
        rows.append(
            {
                "adapter": path.parent.name,
                "mode": data.get("mode"),
                "shuffle_traces": data.get("shuffle_traces"),
                "rank": data.get("lora_rank"),
                "alpha": data.get("lora_alpha"),
                "dropout": data.get("lora_dropout"),
                "epochs": data.get("epochs"),
                "lr": data.get("learning_rate"),
                "max_length": data.get("max_length"),
                "train_records": data.get("train_records"),
            }
        )
    return rows


def lora_table(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        ["Adapter", "Mode", "Shuffled", "Rank", "Alpha", "Dropout", "Epochs", "LR", "Max length", "Train records"],
        [
            [
                row["adapter"],
                row.get("mode", "missing"),
                row.get("shuffle_traces", "missing"),
                row.get("rank", "missing"),
                row.get("alpha", "missing"),
                row.get("dropout", "missing"),
                row.get("epochs", "missing"),
                row.get("lr", "missing"),
                row.get("max_length", "missing"),
                row.get("train_records", "missing"),
            ]
            for row in rows
        ],
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    lines = [",".join(keys)]
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key, "")
            text = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            cells.append('"' + text.replace('"', '""') + '"')
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact(text: str, max_lines: int = 22) -> str:
    lines = text.strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return "\n".join(lines)


def trace_key_lines(text: str) -> str:
    lines = [
        line
        for line in text.splitlines()
        if "TRACE_KEY" in line or "assert actual" in line or "expected_token" in line
    ]
    return compact("\n".join(lines), max_lines=8)


def records_by_episode(path: Path) -> dict[str, dict[str, Any]]:
    return {row["episode_id"]: row for row in read_jsonl(path)}


def results_by_episode(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path) or {}
    return {row["episode_id"]: row for row in payload.get("records", [])}


def choose_result(path: Path, desired: bool) -> dict[str, Any] | None:
    payload = read_json(path) or {}
    for row in payload.get("records", []):
        if bool(row.get("hidden_passed")) is desired:
            return row
    rows = payload.get("records", [])
    return rows[0] if rows else None


def example_section(
    title: str,
    record: dict[str, Any],
    result: dict[str, Any],
    trace_label: str = "Trace evidence:",
) -> list[str]:
    metadata = record.get("metadata", {})
    lines = [
        f"### {title}",
        "",
        f"- Episode: `{record.get('episode_id')}`.",
        f"- Token style: `{metadata.get('token_style')}`.",
        f"- Expected token: `{metadata.get('expected_token')}`.",
        f"- Wrong token: `{metadata.get('wrong_token')}`.",
        f"- Outcome: patch_applied=`{result.get('patch_applied')}`, visible_passed=`{result.get('visible_passed')}`, hidden_passed=`{result.get('hidden_passed')}`.",
        "",
        trace_label,
        "",
        "```text",
        trace_key_lines(record.get("test_output_after_wrong_patch", "")),
        "```",
        "",
        "Generated diff:",
        "",
        "```diff",
        compact(result.get("extracted_patch", ""), max_lines=18),
        "```",
        "",
    ]
    return lines


def build_examples() -> list[str]:
    iid_records = records_by_episode(DATA / "repair_val_iid.jsonl")
    holdout_records = records_by_episode(DATA / "repair_val_format_holdout.jsonl")
    sections: list[str] = []

    success = choose_result(REPORTS / "final_trace_lora_trace_iid.json", desired=True)
    if success and success["episode_id"] in iid_records:
        sections.extend(example_section("Trace-conditioned success", iid_records[success["episode_id"]], success))

    holdout_success = choose_result(REPORTS / "final_trace_lora_trace_format_holdout.json", desired=True)
    if holdout_success and holdout_success["episode_id"] in holdout_records:
        sections.extend(
            example_section(
                "Format-holdout trace-conditioned success",
                holdout_records[holdout_success["episode_id"]],
                holdout_success,
            )
        )

    no_trace = choose_result(REPORTS / "final_trace_lora_no_trace_iid.json", desired=False)
    if not no_trace:
        no_trace = choose_result(REPORTS / "pilot_trace_adapter_no_trace_iid20.json", desired=False)
    if no_trace and no_trace["episode_id"] in iid_records:
        sections.extend(
            example_section(
                "No-trace failure",
                iid_records[no_trace["episode_id"]],
                no_trace,
                "Correct trace withheld from the prompt, shown here for reference:",
            )
        )

    shuffled = choose_result(REPORTS / "final_trace_lora_shuffled_trace_iid.json", desired=False)
    if not shuffled:
        shuffled = choose_result(REPORTS / "pilot_trace_adapter_shuffled_trace_iid20.json", desired=False)
    if shuffled and shuffled["episode_id"] in iid_records:
        sections.extend(
            example_section(
                "Shuffled-trace failure",
                iid_records[shuffled["episode_id"]],
                shuffled,
                "Record's correct trace was replaced by another record's trace; correct trace shown here for reference:",
            )
        )

    return sections


def plot_rates(rows: list[dict[str, Any]], title: str, output: Path, metric_name: str = "repair@1") -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    filtered = [row for row in rows if row.get(metric_name) is not None]
    if not filtered:
        return
    labels = [f"{row.get('split', '')}\n{row['condition']}" for row in filtered]
    values = [float(row[metric_name]) for row in filtered]
    colors = ["#2f6f4e" if "Trace SFT + trace" in row["condition"] else "#6b7c93" for row in filtered]
    height = max(4, 0.36 * len(filtered))
    plt.figure(figsize=(9, height))
    plt.barh(range(len(filtered)), values, color=colors)
    plt.yticks(range(len(filtered)), labels, fontsize=8)
    plt.xlim(0, 1)
    plt.xlabel(metric_name)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def write_artifact_manifest() -> None:
    manifest = [
        "# Trace-Keyed Symbol Repair Large Artifacts",
        "",
        "The downloadable experiment directory intentionally excludes model adapters and checkpoints.",
        "",
        f"- Small experiment directory: `{EXP}`.",
        f"- Large artifact directory: `{LARGE}`.",
        "",
        "Large artifacts:",
    ]
    for path in sorted((LARGE / "models").glob("*")):
        if path.is_dir():
            manifest.append(f"- `{path}`")
    manifest.extend(
        [
            "",
            "To reproduce evaluations, keep the large artifact directory at the path above or update adapter paths in `scripts/run_final_evaluations.py`.",
        ]
    )
    (EXP / "large_artifacts_manifest.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def write_report() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    manifest = read_json(DATA / "dataset_manifest.json") or {}
    core_rows = load_result_rows(CORE_SPECS)
    ablation_rows = load_result_rows(ABLATION_SPECS)
    pilot_rows = load_pilot_rows()
    lora_rows = lora_metadata_rows()

    write_csv(REPORTS / "final_core_results.csv", core_rows)
    write_csv(REPORTS / "final_ablation_results.csv", ablation_rows)
    write_csv(REPORTS / "pilot_results.csv", pilot_rows)

    plot_rates(core_rows, "Trace-Keyed Symbol Repair Final Results", FIGURES / "core_repair_rates.png")
    plot_rates(ablation_rows, "Trace Adapter Input Ablation", FIGURES / "trace_ablation_repair_rates.png")
    plot_rates(core_rows, "Expected-Token Copy Rate", FIGURES / "expected_token_copy_rates.png", metric_name="expected_token_copy_rate")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    paper_lines = [
        "# Trace-Keyed Symbol Repair",
        "",
        f"Generated: `{generated}`.",
        "",
        "## Abstract",
        "",
        (
            "This experiment tests whether a repair model can use a failed execution trace as a data-bearing input, "
            "not only as a generic failure signal. Each synthetic task requires replacing a wrong canonical token with "
            "an expected token that is absent from the issue text and repository files but present in the pytest failure output. "
            "The primary comparison is a trace-conditioned LoRA against frozen, no-trace, shuffled-trace, and final-patch controls."
        ),
        "",
        "## Artifact Layout",
        "",
        f"- Small, download-friendly experiment package: `{EXP}`.",
        f"- Large adapters and checkpoints: `{LARGE}`.",
        "- The small package contains configs, data JSONL files, reports, figures, scripts, and logs.",
        "- The large artifact directory contains model adapters and is excluded from the small package.",
        "",
        "## Dataset",
        "",
        manifest_summary(manifest),
        "",
        "Each record contains a wrong-patched `src/repair_target.py`, visible and hidden pytest tests, a failing trace from the wrong-patched state, and a target corrective diff. The builder validates that the expected token is absent from `current_files`, present in the failing trace, and that the target diff passes visible and hidden tests.",
        "",
        "## Model and Training",
        "",
        f"- Base model: `{MODEL_ID}`.",
        f"- Revision: `{REVISION}`.",
        "- Training method: one-epoch QLoRA adapters.",
        "- Decoding: deterministic generation with `max_new_tokens=128` for final evaluations.",
        "",
        lora_table(lora_rows) if lora_rows else "No adapter metadata was found.",
        "",
        "## Metrics",
        "",
        "- `Repair@1`: the generated diff applies and the repaired files pass hidden tests.",
        "- `Visible pass`: the generated diff applies and the repaired files pass visible tests.",
        "- `Patch apply`: the generated unified diff applies to the intended file state.",
        "- `Expected-token copy`: the generated diff contains the record-specific expected token.",
        "- `Wrong-token removed`: the generated diff removes the wrong token without reintroducing it.",
        "",
        "## Pilot Results",
        "",
        pilot_table(pilot_rows),
        "",
        "The pilot established that the task is learnable from traces and that removing or shuffling trace evidence breaks repair even when the adapter can still emit syntactically valid diffs.",
        "",
        "## Final Results",
        "",
        result_table(core_rows),
        "",
        "## Trace Adapter Ablations",
        "",
        result_table(ablation_rows),
        "",
        "## Figures",
        "",
        "- `figures/core_repair_rates.png`",
        "- `figures/expected_token_copy_rates.png`",
        "- `figures/trace_ablation_repair_rates.png`",
        "",
        "## Qualitative Examples",
        "",
        *build_examples(),
        "## Discussion",
        "",
        (
            "The controlled task isolates whether the trace supplies information needed for repair. "
            "A successful trace-conditioned adapter must both localize the wrong constant and copy a token that is not available in the repository context. "
            "The no-trace and shuffled-trace controls test whether performance can be explained by format memorization or generic patch syntax alone."
        ),
        "",
        "## Limitations",
        "",
        "- The task family is synthetic and intentionally narrow.",
        "- Results measure controlled trace-conditioned token recovery, not broad real-world software maintenance ability.",
        "- All final evaluations use greedy decoding; sampling-based pass rates were not measured.",
        "- The format-holdout split changes token surface form but not program structure.",
        "",
        "## Reproducibility",
        "",
        "Dataset build:",
        "",
        "```bash",
        "python experiments/trace_keyed_symbol_repair/scripts/build_trace_keyed_dataset.py --output-dir experiments/trace_keyed_symbol_repair/data --train 240 --iid 60 --holdout 60 --seed 20260620",
        "```",
        "",
        "Final evaluations:",
        "",
        "```bash",
        "python experiments/trace_keyed_symbol_repair/scripts/run_final_evaluations.py --suite core --max-new-tokens 128",
        "python experiments/trace_keyed_symbol_repair/scripts/run_final_evaluations.py --suite ablation --max-new-tokens 128",
        "```",
        "",
        "Report generation:",
        "",
        "```bash",
        "python experiments/trace_keyed_symbol_repair/scripts/make_report.py",
        "```",
        "",
    ]
    (REPORTS / "trace_keyed_symbol_repair_paper.md").write_text("\n".join(paper_lines), encoding="utf-8")

    summary_lines = [
        "# Trace-Keyed Symbol Repair Summary",
        "",
        f"Generated: `{generated}`.",
        "",
        "## Core Results",
        "",
        result_table(core_rows),
        "",
        "## Ablations",
        "",
        result_table(ablation_rows),
        "",
        "## Artifact Split",
        "",
        f"- Downloadable directory: `{EXP}`.",
        f"- Large adapters/checkpoints: `{LARGE}`.",
        "",
    ]
    (REPORTS / "trace_keyed_symbol_repair_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    write_artifact_manifest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    write_report()
    print(f"wrote {REPORTS / 'trace_keyed_symbol_repair_paper.md'}")


if __name__ == "__main__":
    main()

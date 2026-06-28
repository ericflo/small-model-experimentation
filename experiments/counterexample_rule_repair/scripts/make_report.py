#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
LARGE = ROOT / "large_artifacts" / "counterexample_rule_repair"
REPORTS = EXP / "reports"
FIGURES = EXP / "figures"
DATA = EXP / "data"

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"

SPLIT_LABELS = {
    "iid": "IID",
    "format_holdout": "Format holdout",
    "rule_holdout": "Rule-family holdout",
}

CORE_CONDITIONS = [
    ("Frozen base + trace", "final_frozen_trace_{split}.json"),
    ("Final-patch SFT + final patch", "final_final_patch_lora_final_patch_{split}.json"),
    ("No-trace SFT + no trace", "final_no_trace_lora_no_trace_{split}.json"),
    ("Shuffled-trace SFT + trace", "final_shuffled_trace_lora_trace_{split}.json"),
    ("Trace SFT + trace", "final_trace_lora_trace_{split}.json"),
]

ABLATION_CONDITIONS = [
    ("Trace SFT + no trace", "final_trace_lora_no_trace_{split}.json"),
    ("Trace SFT + shuffled trace", "final_trace_lora_shuffled_trace_{split}.json"),
]

PILOT_SPECS = [
    ("Frozen base + trace, 6 IID", "frozen_trace_iid_pilot6.json"),
    ("Pilot trace adapter + trace, 20 IID", "pilot_trace_iid20.json"),
    ("Pilot trace adapter + no trace, 20 IID", "pilot_trace_adapter_no_trace_iid20.json"),
    ("Pilot trace adapter + shuffled trace, 20 IID", "pilot_trace_adapter_shuffled_trace_iid20.json"),
    ("Full trace adapter + trace, 20 IID check", "full_trace_iid20_check.json"),
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    lines = [",".join(keys)]
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key, "")
            text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
            cells.append('"' + text.replace('"', '""') + '"')
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_specs(conditions: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    specs = []
    for split in SPLIT_LABELS:
        for label, template in conditions:
            specs.append((SPLIT_LABELS[split], label, template.format(split=split)))
    return specs


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


def result_table(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        [
            "Split",
            "Condition",
            "Repair@1",
            "Visible pass",
            "Hidden pass",
            "Patch apply",
            "Marker match",
            "Input literal",
            "Successes",
        ],
        [
            [
                row.get("split", ""),
                row["condition"],
                fmt_rate(row.get("repair@1")),
                fmt_rate(row.get("visible_pass_rate")),
                fmt_rate(row.get("hidden_pass_rate")),
                fmt_rate(row.get("patch_apply_rate")),
                fmt_rate(row.get("target_marker_presence_rate")),
                fmt_rate(row.get("visible_input_literal_rate")),
                fmt_count(row),
            ]
            for row in rows
        ],
    )


def pilot_table(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        ["Condition", "Repair@1", "Visible pass", "Hidden pass", "Patch apply", "Successes"],
        [
            [
                row["condition"],
                fmt_rate(row.get("repair@1")),
                fmt_rate(row.get("visible_pass_rate")),
                fmt_rate(row.get("hidden_pass_rate")),
                fmt_rate(row.get("patch_apply_rate")),
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
            f"- Rule-family-holdout validation records: `{records.get('val_rule_holdout', 'missing')}`.",
            f"- Train families: `{', '.join(manifest.get('train_families', []))}`.",
            f"- Withheld rule families: `{', '.join(manifest.get('rule_holdout_families', []))}`.",
            f"- Dataset seed: `{manifest.get('seed', 'missing')}`.",
            "- Invariants: " + "; ".join(manifest.get("invariants", [])) + ".",
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
                "eval_records": data.get("eval_records"),
            }
        )
    return rows


def lora_table(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        [
            "Adapter",
            "Mode",
            "Shuffled",
            "Rank",
            "Alpha",
            "Dropout",
            "Epochs",
            "LR",
            "Max length",
            "Train records",
            "Eval records",
        ],
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
                row.get("eval_records", "missing"),
            ]
            for row in rows
        ],
    )


def by_family_rows(filename: str, label: str) -> list[dict[str, Any]]:
    payload = read_json(REPORTS / filename) or {}
    summary = payload.get("summary", {})
    rows = []
    for family, family_summary in sorted(summary.get("by_family", {}).items()):
        rows.append(
            {
                **family_summary,
                "condition": label,
                "family": family,
                "file": filename,
            }
        )
    return rows


def family_table(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        ["Condition", "Family", "Repair@1", "Visible pass", "Hidden pass", "Successes"],
        [
            [
                row["condition"],
                row["family"],
                fmt_rate(row.get("repair@1")),
                fmt_rate(row.get("visible_pass_rate")),
                fmt_rate(row.get("hidden_pass_rate")),
                fmt_count(row),
            ]
            for row in rows
        ],
    )


def records_by_episode(path: Path) -> dict[str, dict[str, Any]]:
    return {row["episode_id"]: row for row in read_jsonl(path)}


def choose_result(path: Path, desired: bool) -> dict[str, Any] | None:
    payload = read_json(path) or {}
    for row in payload.get("records", []):
        if bool(row.get("all_tests_passed")) is desired:
            return row
    rows = payload.get("records", [])
    return rows[0] if rows else None


def compact(text: str, max_lines: int = 18) -> str:
    lines = text.strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return "\n".join(lines)


def counterexample_lines(text: str) -> str:
    lines = [line for line in text.splitlines() if "COUNTEREXAMPLE" in line]
    return compact("\n".join(lines), max_lines=10)


def example_section(title: str, record: dict[str, Any], result: dict[str, Any]) -> list[str]:
    metadata = record.get("metadata", {})
    lines = [
        f"### {title}",
        "",
        f"- Episode: `{record.get('episode_id')}`.",
        f"- Family: `{metadata.get('bug_family')}`.",
        f"- Outcome: patch_applied=`{result.get('patch_applied')}`, visible_passed=`{result.get('visible_passed')}`, hidden_passed=`{result.get('hidden_passed')}`.",
        "",
        "Visible counterexamples:",
        "",
        "```text",
        counterexample_lines(record.get("test_output_after_wrong_patch", "")),
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
    format_records = records_by_episode(DATA / "repair_val_format_holdout.jsonl")
    rule_records = records_by_episode(DATA / "repair_val_rule_holdout.jsonl")
    sections: list[str] = []

    success = choose_result(REPORTS / "final_trace_lora_trace_iid.json", desired=True)
    if success and success["episode_id"] in iid_records:
        sections.extend(example_section("Trace-conditioned IID success", iid_records[success["episode_id"]], success))

    format_success = choose_result(REPORTS / "final_trace_lora_trace_format_holdout.json", desired=True)
    if format_success and format_success["episode_id"] in format_records:
        sections.extend(
            example_section(
                "Trace-conditioned format-holdout success",
                format_records[format_success["episode_id"]],
                format_success,
            )
        )

    rule_result = choose_result(REPORTS / "final_trace_lora_trace_rule_holdout.json", desired=True)
    if not rule_result:
        rule_result = choose_result(REPORTS / "final_trace_lora_trace_rule_holdout.json", desired=False)
    if rule_result and rule_result["episode_id"] in rule_records:
        sections.extend(
            example_section(
                "Withheld-rule-family trace-conditioned example",
                rule_records[rule_result["episode_id"]],
                rule_result,
            )
        )

    no_trace = choose_result(REPORTS / "final_trace_lora_no_trace_iid.json", desired=False)
    if no_trace and no_trace["episode_id"] in iid_records:
        sections.extend(example_section("Trace adapter with trace removed", iid_records[no_trace["episode_id"]], no_trace))

    shuffled = choose_result(REPORTS / "final_trace_lora_shuffled_trace_iid.json", desired=False)
    if shuffled and shuffled["episode_id"] in iid_records:
        sections.extend(
            example_section("Trace adapter with shuffled trace", iid_records[shuffled["episode_id"]], shuffled)
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
    colors = ["#2f6f4e" if row["condition"] == "Trace SFT + trace" else "#6b7c93" for row in filtered]
    height = max(4, 0.34 * len(filtered))
    plt.figure(figsize=(10, height))
    plt.barh(range(len(filtered)), values, color=colors)
    plt.yticks(range(len(filtered)), labels, fontsize=8)
    plt.xlim(0, 1)
    plt.xlabel(metric_name)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def write_artifact_manifest() -> None:
    lines = [
        "# Counterexample Rule Repair Large Artifacts",
        "",
        "The downloadable experiment directory intentionally excludes model adapters and checkpoints.",
        "",
        f"- Small experiment directory: `{EXP}`.",
        f"- Large artifact directory: `{LARGE}`.",
        "",
        "Large artifacts:",
    ]
    model_dir = LARGE / "models"
    if model_dir.exists():
        for path in sorted(model_dir.glob("*")):
            if path.is_dir():
                lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "To reproduce adapter-backed evaluations, keep the large artifact directory at the path above or update adapter paths in `scripts/run_final_evaluations.py`.",
        ]
    )
    (EXP / "large_artifacts_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    manifest = read_json(DATA / "dataset_manifest.json") or {}
    core_rows = load_result_rows(load_specs(CORE_CONDITIONS))
    ablation_rows = load_result_rows(load_specs(ABLATION_CONDITIONS))
    pilot_rows = load_pilot_rows()
    lora_rows = lora_metadata_rows()
    family_rows = []
    for split in SPLIT_LABELS:
        family_rows.extend(by_family_rows(f"final_trace_lora_trace_{split}.json", f"Trace SFT + trace / {SPLIT_LABELS[split]}"))

    write_csv(REPORTS / "final_core_results.csv", core_rows)
    write_csv(REPORTS / "final_ablation_results.csv", ablation_rows)
    write_csv(REPORTS / "final_trace_by_family.csv", family_rows)
    write_csv(REPORTS / "pilot_results.csv", pilot_rows)

    plot_rates(core_rows, "Counterexample Rule Repair Final Results", FIGURES / "core_repair_rates.png")
    plot_rates(ablation_rows, "Trace Adapter Input Ablation", FIGURES / "trace_ablation_repair_rates.png")
    plot_rates(core_rows, "Visible Test Pass Rates", FIGURES / "visible_pass_rates.png", metric_name="visible_pass_rate")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    paper_lines = [
        "# Counterexample Rule Repair",
        "",
        f"Generated: `{generated}`.",
        "",
        "## Abstract",
        "",
        (
            "This experiment tests whether a code-repair model can use failed-test counterexamples as evidence for a compact behavioral rule. "
            "Each task presents a wrong-patched implementation and visible pytest failures that print concrete input, expected output, and actual output triples. "
            "The desired rule is not stated in the issue text. A correct patch must infer the rule from visible counterexamples and pass hidden tests on unseen inputs."
        ),
        "",
        "## Artifact Layout",
        "",
        f"- Small, download-friendly experiment package: `{EXP}`.",
        f"- Large adapters and checkpoints: `{LARGE}`.",
        "- The small package contains configs, data JSONL files, reports, figures, scripts, and logs.",
        "- The large artifact directory contains LoRA adapters and is excluded from the small package.",
        "",
        "## Dataset",
        "",
        manifest_summary(manifest),
        "",
        (
            "Each record contains `src/repair_target.py`, visible tests, hidden tests, the failed trace from the wrong-patched implementation, and the target corrective diff. "
            "The visible trace emits `COUNTEREXAMPLE input=... expected=... actual=...` lines. Hidden cases are disjoint from visible inputs, so copying only the visible cases is insufficient."
        ),
        "",
        "## Model and Training",
        "",
        f"- Base model: `{MODEL_ID}`.",
        f"- Revision: `{REVISION}`.",
        "- Training method: QLoRA adapters.",
        "- Final training recipe: 3 epochs, rank 32, alpha 64, dropout 0.05, learning rate 1.5e-4, max length 3072.",
        "- Decoding: deterministic generation with `max_new_tokens=256` for final evaluations.",
        "",
        lora_table(lora_rows) if lora_rows else "No adapter metadata was found.",
        "",
        "## Conditions",
        "",
        "- `Frozen base + trace`: base model with the wrong-patched file and failed trace, no fine-tuning.",
        "- `Trace SFT + trace`: adapter trained and evaluated with failed counterexample traces.",
        "- `No-trace SFT + no trace`: adapter trained and evaluated without failed trace text.",
        "- `Shuffled-trace SFT + trace`: adapter trained on mismatched trace evidence, evaluated with the real trace.",
        "- `Final-patch SFT + final patch`: adapter trained to reproduce final diffs from the original buggy state rather than repair from the wrong-patched state.",
        "- `Trace SFT + no trace` and `Trace SFT + shuffled trace`: input ablations for the trace adapter.",
        "",
        "## Metrics",
        "",
        "- `Repair@1`: the generated diff applies and repaired files pass both visible and hidden tests.",
        "- `Visible pass`: repaired files pass the visible counterexample tests.",
        "- `Hidden pass`: repaired files pass hidden tests on unseen inputs.",
        "- `Patch apply`: the generated unified diff applies to the intended file state.",
        "- `Marker match`: the diff contains all target rule markers recorded by the dataset builder.",
        "- `Input literal`: the diff contains at least one visible input literal, a diagnostic for hardcoding visible cases.",
        "",
        "## Iteration Log Summary",
        "",
        pilot_table(pilot_rows),
        "",
        (
            "The initial frozen pilot showed near-zero patch application and no successful repairs. "
            "A small trace adapter made the task learnable, while no-trace and shuffled-trace prompts remained at zero repair. "
            "The full trace adapter then improved the 20-record IID check enough to justify training the full control adapters."
        ),
        "",
        "## Final Results",
        "",
        result_table(core_rows),
        "",
        "## Trace Adapter Ablations",
        "",
        result_table(ablation_rows),
        "",
        "## Trace Adapter Family Breakdown",
        "",
        family_table(family_rows) if family_rows else "Trace-family results are not available yet.",
        "",
        "## Figures",
        "",
        "- `figures/core_repair_rates.png`",
        "- `figures/trace_ablation_repair_rates.png`",
        "- `figures/visible_pass_rates.png`",
        "",
        "## Qualitative Examples",
        "",
        *build_examples(),
        "## Discussion",
        "",
        (
            "The core contrast is whether the model can transform failed counterexample traces into a general rule rather than merely producing syntactically plausible diffs. "
            "The visible and hidden pass split is important: a patch can sometimes satisfy hidden cases while violating visible counterexamples, so the primary metric requires both. "
            "The rule-family holdout is a harder extrapolation test because the parity-offset structure is absent from training."
        ),
        "",
        "## Limitations",
        "",
        "- The tasks are synthetic and intentionally focused on one-file rule repair.",
        "- The experiment measures greedy single-sample repair, not sampling-based pass rates.",
        "- Hidden tests are generated from known templates, so they are controlled probes rather than open-ended software behavior.",
        "- The withheld rule family tests structural transfer to one unseen family only.",
        "",
        "## Reproducibility",
        "",
        "Dataset build:",
        "",
        "```bash",
        "python experiments/counterexample_rule_repair/scripts/build_counterexample_dataset.py --output-dir experiments/counterexample_rule_repair/data --train-per-family 80 --iid-per-family 15 --format-per-family 15 --rule-holdout 45 --seed 20260620",
        "```",
        "",
        "Final evaluations:",
        "",
        "```bash",
        "python experiments/counterexample_rule_repair/scripts/run_final_evaluations.py --suite all --max-new-tokens 256",
        "```",
        "",
        "Report generation:",
        "",
        "```bash",
        "python experiments/counterexample_rule_repair/scripts/make_report.py",
        "```",
        "",
    ]
    (REPORTS / "counterexample_rule_repair_paper.md").write_text("\n".join(paper_lines), encoding="utf-8")

    summary_lines = [
        "# Counterexample Rule Repair Summary",
        "",
        f"Generated: `{generated}`.",
        "",
        "## Core Results",
        "",
        result_table(core_rows),
        "",
        "## Trace Adapter Ablations",
        "",
        result_table(ablation_rows),
        "",
        "## Trace Adapter Family Breakdown",
        "",
        family_table(family_rows) if family_rows else "Trace-family results are not available yet.",
        "",
        "## Artifact Split",
        "",
        f"- Downloadable directory: `{EXP}`.",
        f"- Large adapters/checkpoints: `{LARGE}`.",
        "",
    ]
    (REPORTS / "counterexample_rule_repair_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    write_artifact_manifest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    write_report()
    print(f"wrote {REPORTS / 'counterexample_rule_repair_paper.md'}")


if __name__ == "__main__":
    main()

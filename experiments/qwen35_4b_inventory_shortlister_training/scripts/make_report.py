#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "reports" / "shortlister_results.json"
LOSS_PATH = ROOT / "reports" / "training_losses.json"
REPORT_PATH = ROOT / "reports" / "qwen35_4b_inventory_shortlister_training_report.md"
FIG_DIR = ROOT / "reports" / "figures"


def pct(metric: dict[str, Any]) -> float:
    return 100.0 * float(metric["rate"])


def flatten(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, dict) and {"successes", "records", "rate"} <= set(value):
                flat[f"{key}_successes"] = value["successes"]
                flat[f"{key}_records"] = value["records"]
                flat[f"{key}_pct"] = round(100.0 * value["rate"], 3)
            else:
                flat[key] = value
        out.append(flat)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_losses(losses: list[dict[str, Any]]) -> None:
    if not losses:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot([row["step"] for row in losses], [row["loss"] for row in losses], marker=".", linewidth=1)
    plt.xlabel("optimizer step")
    plt.ylabel("training loss")
    plt.title("QLoRA Shortlister Training Loss")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_loss.png", dpi=180)
    plt.close()


def plot_budget_summary(result: dict[str, Any]) -> None:
    rows = result["prediction_summary_by_control"]
    budgets = [1024, 4096, 16384]
    controls = [row["control"] for row in rows]
    width = 0.24
    xs = range(len(controls))
    plt.figure(figsize=(9, 5))
    for offset, budget in enumerate(budgets):
        ys = [pct(row[f"pair_budget{budget}_hit"]) for row in rows]
        positions = [x + (offset - 1) * width for x in xs]
        plt.bar(positions, ys, width=width, label=f"budget {budget}")
    plt.xticks(list(xs), controls, rotation=15, ha="right")
    plt.ylim(0, 100)
    plt.ylabel("target pair in budget (%)")
    plt.title("Fixed-Budget Pair Coverage")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fixed_budget_pair_coverage.png", dpi=180)
    plt.close()


def plot_template_summary(result: dict[str, Any]) -> None:
    rows = result["prediction_summary_by_control_template"]
    controls = sorted({row["control"] for row in rows})
    templates = sorted({row["template"] for row in rows})
    plt.figure(figsize=(9, 5))
    width = 0.22
    xs = range(len(templates))
    for offset, control in enumerate(controls):
        subset = {row["template"]: row for row in rows if row["control"] == control}
        ys = [pct(subset[template]["pair_budget16384_hit"]) for template in templates]
        positions = [x + (offset - (len(controls) - 1) / 2) * width for x in xs]
        plt.bar(positions, ys, width=width, label=control)
    plt.xticks(list(xs), templates, rotation=0)
    plt.ylim(0, 100)
    plt.ylabel("pair in 16384-candidate budget (%)")
    plt.title("Template Split At Largest Budget")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "template_budget_split.png", dpi=180)
    plt.close()


def plot_probe(result: dict[str, Any]) -> None:
    probe = result["probe_summary"]
    labels = ["random six", "designed six"]
    counts = [probe["random_visible_consistent_count"], probe["designed_visible_consistent_count"]]
    selected = [pct(probe["random_selected_hidden_all"]), pct(probe["designed_selected_hidden_all"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, counts, color=["#5b7c99", "#d1863a"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("visible-consistent candidates")
    axes[0].set_title("Probe Candidate Reduction")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(labels, selected, color=["#5b7c99", "#d1863a"])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("selected hidden-all (%)")
    axes[1].set_title("Probe Selection Lift")
    axes[1].grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "probe_design_diagnostic.png", dpi=180)
    plt.close()


def row_for_control(result: dict[str, Any], control: str) -> dict[str, Any]:
    for row in result["prediction_summary_by_control"]:
        if row["control"] == control:
            return row
    raise KeyError(control)


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _key in columns) + " |"
    sep = "| " + " | ".join("---" for _label, _key in columns) + " |"
    body = []
    for row in rows:
        vals = []
        for _label, key in columns:
            value = row[key]
            if isinstance(value, float):
                vals.append(f"{value:.1f}")
            else:
                vals.append(str(value))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def report_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in result["prediction_summary_by_control"]:
        out.append(
            {
                "control": row["control"],
                "records": row["records"],
                "top1": round(pct(row["pair_top1_hit"]), 1),
                "b1024": round(pct(row["pair_budget1024_hit"]), 1),
                "b4096": round(pct(row["pair_budget4096_hit"]), 1),
                "b16384": round(pct(row["pair_budget16384_hit"]), 1),
            }
        )
    return out


def template_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in result["prediction_summary_by_control_template"]:
        out.append(
            {
                "control": row["control"],
                "template": row["template"],
                "records": row["records"],
                "b1024": round(pct(row["pair_budget1024_hit"]), 1),
                "b16384": round(pct(row["pair_budget16384_hit"]), 1),
            }
        )
    return out


def report_text(result: dict[str, Any], losses: list[dict[str, Any]]) -> str:
    rows = report_rows(result)
    probe = result["probe_summary"]
    trained = row_for_control(result, "trained_model")
    shuffled = row_for_control(result, "trained_model_shuffled_inventory")
    last_loss = losses[-1]["loss"] if losses else None
    loss_sentence = f" Final logged loss was `{last_loss:.4f}`." if last_loss is not None else ""
    beam_width = int(result.get("beam_width", 0))
    exact_budget = beam_width * beam_width
    budget_scope = (
        f"The constrained decoder used beam width `{beam_width}`, so the exact fixed-budget measurement is `{exact_budget}` "
        "candidate pairs. Larger budget columns are lower bounds from the same beams, not full top-64/top-128 evaluations."
    )
    return f"""# Qwen3.5-4B Inventory Shortlister Training Report

## Summary

This standalone experiment trains a QLoRA adapter on Qwen3.5-4B to shortlist operators for two-hole 512-operator programs. The model predicts LEFT and RIGHT aliases independently; candidate budgets are formed as top-32 x top-32, top-64 x top-64, and top-128 x top-128.

Training completed and produced a LoRA adapter outside the experiment directory.{loss_sentence}

{budget_scope}

The trained model's exact `{exact_budget}`-candidate pair coverage is `{pct(trained['pair_budget1024_hit']):.1f}%`. The shuffled-inventory control is `{pct(shuffled['pair_budget1024_hit']):.1f}%`, testing whether gains depend on the alias-description mapping.

## Fixed-Budget Pair Coverage

{table(rows, [
    ('control', 'control'),
    ('records', 'records'),
    ('top1 %', 'top1'),
    ('1024 exact %', 'b1024'),
    ('4096 lower-bound %', 'b4096'),
    ('16384 lower-bound %', 'b16384'),
])}

![Fixed budget pair coverage](figures/fixed_budget_pair_coverage.png)

## Template Split

{table(template_rows(result), [
    ('control', 'control'),
    ('template', 'template'),
    ('records', 'records'),
    ('1024 %', 'b1024'),
    ('16384 %', 'b16384'),
])}

![Template budget split](figures/template_budget_split.png)

## Observation Design Diagnostic

For low-information comparison records, a max-split six-query design was compared against the random six visible cases. This diagnostic uses the executable task generator to measure how much better observations can reduce ambiguity before selection.

- Random six visible cases left `{probe['random_visible_consistent_count']}` candidates on average.
- Designed six cases left `{probe['designed_visible_consistent_count']}` candidates on average.
- Random selected-hidden-all was `{pct(probe['random_selected_hidden_all']):.1f}%`.
- Designed selected-hidden-all was `{pct(probe['designed_selected_hidden_all']):.1f}%`.

![Probe design diagnostic](figures/probe_design_diagnostic.png)

## Training Loss

![Training loss](figures/training_loss.png)

## Decision

This run directly tests whether Qwen3.5-4B can turn inventory-conditioned examples into a useful two-hole shortlist. At this pilot scale, it does not: trained, base, and shuffled controls all miss the exact 1024-candidate budget on the evaluated subset. The next lever should be structured semantic search or a different supervision/evaluation interface, not simply larger blind beam search.

The observation diagnostic is more promising: designed probes reduce the low-information ambiguity substantially, though not enough to solve selection by themselves.

## Artifacts

- Dataset manifest: `data/dataset_manifest.json`
- Train slots: `data/train_slots.jsonl`
- Eval records: `data/eval_records.jsonl`
- Results: `reports/shortlister_results.json`
- Training losses: `reports/training_losses.json`
- Large artifacts: `/workspace/large_artifacts/qwen35_4b_inventory_shortlister_training`
"""


def main() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    losses = json.loads(LOSS_PATH.read_text(encoding="utf-8")) if LOSS_PATH.exists() else []
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(ROOT / "reports" / "prediction_summary.csv", flatten(result["prediction_summary_by_control"]))
    write_csv(ROOT / "reports" / "prediction_template_summary.csv", flatten(result["prediction_summary_by_control_template"]))
    write_csv(ROOT / "reports" / "probe_rows.csv", flatten(result["probe_rows"]))
    plot_losses(losses)
    plot_budget_summary(result)
    plot_template_summary(result)
    plot_probe(result)
    REPORT_PATH.write_text(report_text(result, losses), encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()

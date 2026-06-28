#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, color: str = "#4c78a8") -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=color)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, min(0.98, v + 0.025), pct(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    gate = load_json(REPORTS / "gate1_summary.json")
    sweep = load_json(REPORTS / "gate1_split_sweep.json")
    calibration = gate["calibration"]["overall"]
    heldout = gate["heldout"]["overall"]
    train = gate["train"]["overall"]

    bar(
        FIGURES / "gate1_coverage_calibration_vs_heldout.png",
        ["calibration", "heldout", "train"],
        [calibration["oracle_coverage"], heldout["oracle_coverage"], train["oracle_coverage"]],
        "Frozen ABI Oracle Coverage",
        "Coverage",
    )

    seed_labels = [f"seed {item['seed']}" for item in sweep["results"]]
    seed_values = [item["overall"]["oracle_coverage"] for item in sweep["results"]]
    bar(
        FIGURES / "gate1_heldout_seed_sweep.png",
        seed_labels,
        seed_values,
        "Held-Out Coverage Across Random Test-Suffix Samples",
        "Coverage",
        color="#f58518",
    )

    slices = sorted(gate["heldout"]["by_slice"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(slices))
    cal_values = [gate["calibration"]["by_slice"].get(s, {"oracle_coverage": 0})["oracle_coverage"] for s in slices]
    held_values = [gate["heldout"]["by_slice"][s]["oracle_coverage"] for s in slices]
    width = 0.38
    ax.bar([i - width / 2 for i in x], cal_values, width=width, label="calibration", color="#4c78a8")
    ax.bar([i + width / 2 for i in x], held_values, width=width, label="heldout", color="#e45756")
    ax.set_xticks(list(x))
    ax.set_xticklabels(slices, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage by Task Slice")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "gate1_coverage_by_slice.png", dpi=160)
    plt.close(fig)

    depth_counts = gate["depth_counts"]
    all_depths = sorted(set(depth_counts["calibration"]) | set(depth_counts["heldout"]) | set(depth_counts["train"]), key=int)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(all_depths))
    width = 0.25
    for offset, key, color in [(-width, "calibration", "#4c78a8"), (0, "heldout", "#e45756"), (width, "train", "#54a24b")]:
        ax.bar([i + offset for i in x], [depth_counts[key].get(d, 0) for d in all_depths], width=width, label=key, color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(all_depths)
    ax.set_xlabel("Winning ABI depth")
    ax.set_ylabel("Covered tasks")
    ax.set_title("Covered Tasks by Winning Program Depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "gate1_depth_counts.png", dpi=160)
    plt.close(fig)

    table = f"""| split | n | oracle-covered | oracle coverage | first-visible correct | visible-any tasks | task false-pass rate | candidate hidden-wrong rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| calibration | {calibration['n']} | {calibration['oracle_covered']} | {pct(calibration['oracle_coverage'])} | {calibration['first_visible_full_pass']} | {calibration['visible_any']} | {pct(calibration['visible_false_pass_rate_among_visible'])} | {pct(calibration['visible_hidden_wrong_rate_among_candidates'])} |
| heldout | {heldout['n']} | {heldout['oracle_covered']} | {pct(heldout['oracle_coverage'])} | {heldout['first_visible_full_pass']} | {heldout['visible_any']} | {pct(heldout['visible_false_pass_rate_among_visible'])} | {pct(heldout['visible_hidden_wrong_rate_among_candidates'])} |
| train | {train['n']} | {train['oracle_covered']} | {pct(train['oracle_coverage'])} | {train['first_visible_full_pass']} | {train['visible_any']} | {pct(train['visible_false_pass_rate_among_visible'])} | {pct(train['visible_hidden_wrong_rate_among_candidates'])} |"""

    sweep_rows = "\n".join(
        f"| {item['seed']} | {item['overall']['n']} | {item['overall']['oracle_covered']} | {pct(item['overall']['oracle_coverage'])} | "
        f"{item['overall']['first_visible_full_pass']} | {pct(item['overall']['visible_false_pass_rate_among_visible'])} |"
        for item in sweep["results"]
    )

    slice_rows = "\n".join(
        f"| {s} | {gate['calibration']['by_slice'].get(s, {'oracle_covered': 0, 'n': 0, 'oracle_coverage': 0})['oracle_covered']}/"
        f"{gate['calibration']['by_slice'].get(s, {'n': 0})['n']} ({pct(gate['calibration']['by_slice'].get(s, {'oracle_coverage': 0})['oracle_coverage'])}) | "
        f"{gate['heldout']['by_slice'][s]['oracle_covered']}/{gate['heldout']['by_slice'][s]['n']} ({pct(gate['heldout']['by_slice'][s]['oracle_coverage'])}) |"
        for s in slices
    )

    report = f"""# Frozen Code ABI Held-Out Primitive Pilot

## Purpose

This standalone experiment tests whether a frozen code-primitive ABI remains reusable on held-out MBPP tasks before training a Qwen3.5-4B compiler to emit ABI programs.

The package uses a fixed ABI implementation and does not add kernels after seeing the held-out tasks. The first gate is oracle coverage: if the ABI cannot express held-out tasks, compiler training would not test reusable compilation.

## Gate 1 Result

Gate 1 failed. Frozen-ABI oracle coverage dropped from {calibration['oracle_covered']}/{calibration['n']} ({pct(calibration['oracle_coverage'])}) on the calibration slice to {heldout['oracle_covered']}/{heldout['n']} ({pct(heldout['oracle_coverage'])}) on the held-out slice, a drop of {pct(gate['coverage_drop'])}.

The train split also has low coverage: {train['oracle_covered']}/{train['n']} ({pct(train['oracle_coverage'])}), leaving only {gate['target_counts']['train']} compiler-training targets after the deterministic validation split. That is not enough to make a QLoRA compiler result meaningful.

![Gate 1 coverage](figures/gate1_coverage_calibration_vs_heldout.png)

{table}

## Split-Sweep Check

To check whether the fixed held-out slice was unlucky, the experiment sampled three random 160-task subsets from the test suffix excluded from the calibration slice. Coverage remained low: mean {pct(sweep['mean_coverage'])}, range {pct(sweep['min_coverage'])}-{pct(sweep['max_coverage'])}.

| seed | n | oracle-covered | oracle coverage | first-visible correct | task false-pass rate |
|---:|---:|---:|---:|---:|---:|
{sweep_rows}

![Held-out seed sweep](figures/gate1_heldout_seed_sweep.png)

## Slice Diagnostics

| slice | calibration coverage | heldout coverage |
|---|---:|---:|
{slice_rows}

![Coverage by slice](figures/gate1_coverage_by_slice.png)

## Depth Diagnostics

Covered held-out tasks are mostly depth-1 programs: {gate['depth_counts']['heldout']}. There are too few held-out depth-2/3 targets to support a composition-training claim.

![Depth counts](figures/gate1_depth_counts.png)

## Decision

The compiler-training arm was intentionally not run. The precondition for interpreting it did not hold: the frozen ABI was not broadly reusable on held-out tasks. Training Qwen to emit this ABI would mostly measure a small, contaminated target set rather than reusable code compilation.

The next productive step is library curation under a strict protocol: define primitives from a source independent of the evaluation tasks, freeze them, then repeat this held-out coverage gate. Adding kernels after inspecting held-out misses would invalidate the purpose of the gate.

## Files

- `src/abi_oracle.py`: frozen ABI oracle implementation.
- `scripts/build_targets.py`: calibration, held-out, and train target builder.
- `scripts/gate1_split_sweep.py`: random held-out split sweep.
- `reports/gate1_summary.json`: primary Gate 1 results.
- `reports/gate1_split_sweep.json`: multi-split confirmation.
- `reports/figures/`: generated charts.
"""

    (REPORTS / "final_report.md").write_text(report, encoding="utf-8")
    print(REPORTS / "final_report.md")


if __name__ == "__main__":
    main()

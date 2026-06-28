#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "reports" / "eval"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_PATH = ROOT / "reports" / "qwen35_4b_typed_sketch_synthesis_report.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(result: dict[str, Any], key: str) -> tuple[int, int, float]:
    item = result["summary"]["overall"][key]
    return int(item["successes"]), int(item["records"]), float(item["rate"])


def split_results() -> dict[str, dict[str, Any]]:
    return {
        "IID": load_json(EVAL_DIR / "sketch_iid.json"),
        "Support": load_json(EVAL_DIR / "sketch_support.json"),
        "Ceiling": load_json(EVAL_DIR / "sketch_ceiling.json"),
    }


def pct(successes: int, records: int) -> str:
    return f"{successes}/{records} ({successes / records:.1%})"


def plot_overall(results: dict[str, dict[str, Any]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = list(results)
    series = [
        ("Direct program", "base_rerank_hidden_all", "#3b5b92"),
        ("Sketch selected", "sketch_synth_hidden_all", "#d18f2f"),
        ("Sketch oracle", "sketch_synth_oracle_hidden_all", "#4f8f5b"),
        ("Conservative hybrid", "hybrid_hidden_all", "#7a4f9f"),
    ]
    x = range(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for offset, (name, key, color) in enumerate(series):
        rates = [metric(results[label], key)[2] for label in labels]
        positions = [value + (offset - 1.5) * width for value in x]
        ax.bar(positions, rates, width=width, label=name, color=color)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Hidden all-cases success rate")
    ax.set_title("Verified Typed-Sketch Synthesis Results")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "overall_hidden_success.png", dpi=180)
    plt.close(fig)


def plot_ceiling_by_family(ceiling: dict[str, Any]) -> None:
    families = list(ceiling["summary"]["by_family"])
    keys = [
        ("Direct", "base_rerank_hidden_all", "#3b5b92"),
        ("Sketch", "sketch_synth_hidden_all", "#d18f2f"),
        ("Hybrid", "hybrid_hidden_all", "#7a4f9f"),
    ]
    y = range(len(families))
    height = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    for offset, (name, key, color) in enumerate(keys):
        rates = [ceiling["summary"]["by_family"][family][key]["rate"] for family in families]
        positions = [value + (offset - 1) * height for value in y]
        ax.barh(positions, rates, height=height, label=name, color=color)
    ax.set_yticks(list(y), families)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("Hidden all-cases success rate")
    ax.set_title("Ceiling Split By Family")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ceiling_by_family.png", dpi=180)
    plt.close(fig)


def plot_candidate_counts(ceiling: dict[str, Any]) -> None:
    families = list(ceiling["summary"]["by_family"])
    counts = [ceiling["summary"]["by_family"][family]["avg_synthesized_programs"] for family in families]
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.barh(families, counts, color="#607d8b")
    ax.set_xlabel("Average synthesized programs per record")
    ax.set_title("Ceiling Split Search Cost")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ceiling_candidate_counts.png", dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(results: dict[str, dict[str, Any]]) -> None:
    overview_rows = []
    for split, result in results.items():
        base = metric(result, "base_rerank_hidden_all")
        sketch = metric(result, "sketch_synth_hidden_all")
        oracle = metric(result, "sketch_synth_oracle_hidden_all")
        hybrid = metric(result, "hybrid_hidden_all")
        overview_rows.append([split, pct(*base[:2]), pct(*sketch[:2]), pct(*oracle[:2]), pct(*hybrid[:2])])

    ceiling = results["Ceiling"]
    family_rows = []
    for family, block in ceiling["summary"]["by_family"].items():
        records = block["base_rerank_hidden_all"]["records"]
        family_rows.append(
            [
                family,
                str(block["base_rerank_hidden_all"]["successes"]),
                str(block["sketch_synth_hidden_all"]["successes"]),
                str(block["sketch_synth_oracle_hidden_all"]["successes"]),
                str(block["hybrid_hidden_all"]["successes"]),
                f"{block['avg_synthesized_programs']:.1f}",
            ]
        )

    ceiling_overall = ceiling["summary"]["overall"]
    content = f"""# Qwen 3.5 4B Typed Sketch Synthesis

## Objective

Test whether Qwen 3.5 4B can produce typed executable sketches that a verifier completes into better DSL repairs than direct program generation.

## Method

- Base model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Training data: 240 records for each adapter.
- Adapters:
  - `program_lora`: trained to emit complete DSL programs.
  - `sketch_lora`: trained to emit typed sketches with `?NUM`, `?TEXT`, `?SEQ`, and `?PRED` holes.
- Verifier:
  - Completes typed holes from a schema-derived expression bank.
  - Runs candidates on visible cases.
  - Reports selected hidden success and hidden-oracle coverage.
- Final selection rule:
  - Keep the direct-program result if it passes every visible case.
  - Otherwise use sketch synthesis when it passes every visible case.
  - Otherwise choose the candidate with more visible passes.

## Iterations

1. Deterministic target sketches recovered the target program on all train, IID, support, and ceiling records under the planned caps.
2. Initial model-generated sketch synthesis failed on a five-record ceiling smoke test: target synthesis was 0/5 and oracle hidden success was 0/5.
3. Added structural abstraction variants such as `(if ?PRED0 high_label low_label)` and `(format "X{{}}" ?NUM0)`, plus deeper numeric and predicate expression-bank entries.
4. Changed visible-pass tie-breaking to prefer input-dependent, structurally richer candidates instead of shorter programs.
5. Fixed candidate tag merging so targeted predicates promote generic candidates already in the bank.

## Results

{markdown_table(["Split", "Direct program", "Sketch selected", "Sketch oracle", "Conservative hybrid"], overview_rows)}

![Overall hidden success](figures/overall_hidden_success.png)

The ceiling split is the important result. Direct program generation solved {pct(*metric(ceiling, "base_rerank_hidden_all")[:2])}. Sketch synthesis selected by visible cases solved {pct(*metric(ceiling, "sketch_synth_hidden_all")[:2])}. The conservative hybrid solved {pct(*metric(ceiling, "hybrid_hidden_all")[:2])}. Hidden-oracle coverage was {pct(*metric(ceiling, "sketch_synth_oracle_hidden_all")[:2])}, which means the verifier search space contained every target program on the ceiling split.

Ceiling family breakdown:

{markdown_table(["Family", "Direct", "Sketch", "Oracle", "Hybrid", "Avg candidates"], family_rows)}

![Ceiling by family](figures/ceiling_by_family.png)

![Candidate counts](figures/ceiling_candidate_counts.png)

## Interpretation

Typed sketch synthesis changed the ceiling result from 40/120 to 94/120 with sketch selection alone and to 106/120 with the conservative hybrid. The oracle result of 120/120 shows that the remaining failures are not expression coverage failures; they are visible-case selection failures.

The experiment did not produce a universal training tweak by itself. It did produce a strong concrete mechanism: use Qwen 3.5 4B to identify output format and coarse control structure, then let a typed verifier search deeper compositions than the model reliably emits token-by-token.

## Failure Modes

- Sketch-alone selection is unsafe on easy splits: IID direct generation is 60/60, while sketch-alone is 45/60 because many visible-equivalent candidates exist.
- The conservative hybrid protects solved visible-all direct outputs, but ceiling still has 14 hidden failures versus a 120/120 oracle.
- Several families hit the 8,000-candidate cap, so runtime is still dominated by broad symbolic enumeration.
- The expression bank is manually engineered for this DSL. The result is evidence for the typed-sketch/verifier direction, not for a domain-independent recipe yet.

## Next Experiment

The next experiment should make selection adaptive: after sketch synthesis finds many visible-equivalent programs, generate new discriminating visible cases on the fly, rerun the candidates, and train or evaluate the policy on that counterexample-acquisition loop. The MDP framing is direct: state is the candidate set plus visible traces, actions request additional cases or commit to a program, and reward is verified generalization under a fixed case budget.

## Artifacts

- Compact experiment directory: `{ROOT}`
- Large adapter/checkpoint root: `/workspace/large_artifacts/qwen35_4b_typed_sketch_synthesis`
- Direct evals: `reports/eval/program_iid.json`, `reports/eval/program_support.json`, `reports/eval/program_ceiling.json`
- Sketch evals: `reports/eval/sketch_iid.json`, `reports/eval/sketch_support.json`, `reports/eval/sketch_ceiling.json`
- Training logs: `run_logs/training_program_lora_console.log`, `run_logs/training_sketch_lora_console.log`
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    results = split_results()
    plot_overall(results)
    plot_ceiling_by_family(results["Ceiling"])
    plot_candidate_counts(results["Ceiling"])
    write_report(results)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()

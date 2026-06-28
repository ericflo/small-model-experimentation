#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DATA = ROOT / "data"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def selector_table(summary: dict) -> str:
    names = [
        "first_visible_independent",
        "consensus_independent_min2",
        "consensus_independent_min3",
        "first_visible_same",
        "consensus_same_min2",
        "consensus_same_min3",
        "oracle_independent",
        "oracle_same",
        "oracle_union",
    ]
    labels = {
        "first_visible_independent": "first visible independent",
        "consensus_independent_min2": "independent consensus min-2",
        "consensus_independent_min3": "independent consensus min-3",
        "first_visible_same": "first visible same-neighborhood",
        "consensus_same_min2": "same-neighborhood consensus min-2",
        "consensus_same_min3": "same-neighborhood consensus min-3",
        "oracle_independent": "oracle independent",
        "oracle_same": "oracle same-neighborhood",
        "oracle_union": "oracle union",
    }
    lines = [
        "| selector | commits | hidden-correct commits | hidden-wrong commits | false-pass rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        item = summary["selectors"][name]
        lines.append(
            "| "
            + " | ".join(
                [
                    labels[name],
                    str(item["commit_count"]),
                    str(item["selected_hidden_correct"]),
                    str(item["selected_visible_hidden_wrong"]),
                    pct(item["selected_false_pass_rate"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def pool_table(consensus: dict, direct_manifest: dict) -> str:
    rows = [
        (
            "independent retrieval-adapt top-6",
            consensus["independent_pool"]["coverage"],
            consensus["independent_pool"]["visible_hidden_wrong_rate"],
            consensus["independent_pool"]["candidate_count_mean"],
            consensus["token_usage"]["independent"]["forward_tokens"],
        ),
        (
            "same-neighborhood retrieval-adapt top-6",
            consensus["same_neighborhood_pool"]["coverage"],
            consensus["same_neighborhood_pool"]["visible_hidden_wrong_rate"],
            consensus["same_neighborhood_pool"]["candidate_count_mean"],
            consensus["token_usage"]["same_neighborhood"]["forward_tokens"],
        ),
        (
            "direct sample-more K12",
            direct_manifest["summary"]["coverage"],
            0.0,
            direct_manifest["summary"]["candidate_count_mean"],
            direct_manifest["token_usage"]["forward_tokens"],
        ),
    ]
    lines = [
        "| pool | coverage | visible-pass hidden-wrong rate | candidates/task | forward tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, coverage, false_rate, candidates, tokens in rows:
        false_text = pct(false_rate) if name != "direct sample-more K12" else "not selector pool"
        lines.append(
            f"| {name} | {pct(coverage)} | {false_text} | {candidates:.2f} | {tokens:,} |"
        )
    return "\n".join(lines)


def probe_diagnostic(rows: list[dict]) -> dict:
    out = {}
    for key in ["independent", "same_neighborhood"]:
        counts = [len(row["selected_probe_indices"][key]) for row in rows]
        out[key] = {
            "tasks_with_disagreement": sum(count > 0 for count in counts),
            "mean_selected_probes": sum(counts) / len(counts) if counts else 0.0,
            "counts": counts,
        }
    return out


def plot_independence(plan: dict) -> None:
    labels = ["same", "independent"]
    code = [
        plan["same_neighborhood"]["mean_pairwise_code_distance"],
        plan["independent"]["mean_pairwise_code_distance"],
    ]
    task = [
        plan["same_neighborhood"]["mean_pairwise_task_distance"],
        plan["independent"]["mean_pairwise_task_distance"],
    ]
    x = range(len(labels))
    width = 0.36
    plt.figure(figsize=(7, 4))
    plt.bar([i - width / 2 for i in x], code, width=width, label="code distance")
    plt.bar([i + width / 2 for i in x], task, width=width, label="task-token distance")
    plt.xticks(list(x), labels)
    plt.ylim(0, 1.0)
    plt.ylabel("mean pairwise distance")
    plt.title("Retrieval independence gate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "retrieval_independence.png", dpi=180)
    plt.close()


def plot_pool_coverage(consensus: dict, direct_manifest: dict) -> None:
    labels = ["independent", "same", "direct K12"]
    coverage = [
        consensus["independent_pool"]["coverage"] * 24,
        consensus["same_neighborhood_pool"]["coverage"] * 24,
        direct_manifest["summary"]["coverage"] * 24,
    ]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, coverage, color=["#4c78a8", "#f58518", "#54a24b"])
    plt.ylim(0, 12)
    plt.ylabel("covered tasks out of 24")
    plt.title("Pool coverage before selection")
    plt.tight_layout()
    plt.savefig(FIGURES / "pool_coverage.png", dpi=180)
    plt.close()


def plot_selectors(consensus: dict) -> None:
    names = [
        "first_visible_independent",
        "consensus_independent_min2",
        "first_visible_same",
        "consensus_same_min2",
        "oracle_union",
    ]
    labels = ["first ind", "cons ind", "first same", "cons same", "oracle"]
    correct = [consensus["selectors"][name]["selected_hidden_correct"] for name in names]
    wrong = [consensus["selectors"][name]["selected_visible_hidden_wrong"] for name in names]
    x = range(len(labels))
    plt.figure(figsize=(8, 4))
    plt.bar(x, correct, label="hidden-correct")
    plt.bar(x, wrong, bottom=correct, label="hidden-wrong")
    plt.xticks(list(x), labels)
    plt.ylim(0, 16)
    plt.ylabel("selected tasks")
    plt.title("Selector outcomes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "selector_outcomes.png", dpi=180)
    plt.close()


def plot_tokens(consensus: dict, direct_manifest: dict) -> None:
    labels = ["independent", "same", "direct K12"]
    tokens = [
        consensus["token_usage"]["independent"]["forward_tokens"],
        consensus["token_usage"]["same_neighborhood"]["forward_tokens"],
        direct_manifest["token_usage"]["forward_tokens"],
    ]
    coverage = [
        consensus["independent_pool"]["coverage"] * 24,
        consensus["same_neighborhood_pool"]["coverage"] * 24,
        direct_manifest["summary"]["coverage"] * 24,
    ]
    plt.figure(figsize=(7, 4))
    plt.scatter(tokens, coverage, s=90)
    for label, x, y in zip(labels, tokens, coverage):
        plt.text(x + 800, y, label, va="center")
    plt.xlabel("forward tokens")
    plt.ylabel("covered tasks out of 24")
    plt.title("Coverage versus model cost")
    plt.tight_layout()
    plt.savefig(FIGURES / "tokens_vs_coverage.png", dpi=180)
    plt.close()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plan = read_json(REPORTS / "retrieval_plan_summary.json")
    consensus = read_json(REPORTS / "consensus_summary.json")
    direct_manifest = read_json(DATA / "direct_sample_more_k12_records.manifest.json")
    rows = read_jsonl(DATA / "consensus_selector_records.jsonl")
    probes = probe_diagnostic(rows)

    plot_independence(plan)
    plot_pool_coverage(consensus, direct_manifest)
    plot_selectors(consensus)
    plot_tokens(consensus, direct_manifest)

    report = f"""# qwen35_4b_independent_retrieval_consensus

Date: 2026-06-26

## Decision

Independent-retrieval consensus does **not** pass the deployable gate in this run. The retrieval planner did create more independent sources, but the generated independent adaptations did not converge on disagreement probes. With the mechanism-faithful rule that requires at least one generated disagreement probe, independent consensus committed on **0/24** tasks. First-visible on the same independent pool selected **6/24** hidden-correct tasks, with 8 hidden-wrong visible-pass selections.

The stronger pool was the non-independent same-neighborhood control: it had **9/24** oracle coverage versus **7/24** for independent retrieval and **7/24** for direct K12 sampling. But same-neighborhood consensus still failed selection: min-2 consensus selected **2/24** hidden-correct and 3 hidden-wrong tasks.

## Question

Can independently retrieved verified algorithms supply the missing behavioral evidence for selecting correct retrieval-adapt candidates without hidden-test labels?

## Setup

- Residual tasks: 24 MBPP heldout tasks missed by the base direct K=4 pool.
- Library: 364 verified algorithms.
- Retrieval arms: top-6 same-neighborhood semantic retrieval and top-6 MMR-diversified independent retrieval.
- Generation: one Qwen3.5-4B adaptation per retrieved algorithm, T=0.2, top-p 0.95.
- Consensus: generate up to 64 input probes from public test perturbations, choose up to 8 calls that split visible-passing candidates, and commit only when at least 2 or 3 distinct retrieved sources agree on the same output signature.
- Baseline: direct K12 sample-more on the same 24 residual tasks.

## Independence Gate

![Retrieval independence](figures/retrieval_independence.png)

| retrieval set | code distance | task-token distance | mean retrieval score |
|---|---:|---:|---:|
| same-neighborhood | {plan['same_neighborhood']['mean_pairwise_code_distance']:.3f} | {plan['same_neighborhood']['mean_pairwise_task_distance']:.3f} | {plan['same_neighborhood']['mean_retrieval_score']:.3f} |
| independent MMR | {plan['independent']['mean_pairwise_code_distance']:.3f} | {plan['independent']['mean_pairwise_task_distance']:.3f} | {plan['independent']['mean_retrieval_score']:.3f} |

The build gate passed: independent retrieval increased source-code and task-token distance while retaining similar semantic score.

## Pool Coverage

![Pool coverage](figures/pool_coverage.png)

{pool_table(consensus, direct_manifest)}

Independence did not improve coverage. It lowered coverage relative to the same-neighborhood control and matched direct K12.

## Consensus Selection

![Selector outcomes](figures/selector_outcomes.png)

{selector_table(consensus)}

Strict independent consensus had no deployable commits because no covered task had cross-source agreement on actual disagreement probes. Same-neighborhood consensus committed sometimes, but its false-pass rate remained high.

## Probe Diagnostics

| pool | tasks with generated disagreement probes | mean selected probes/task |
|---|---:|---:|
| independent | {probes['independent']['tasks_with_disagreement']}/24 | {probes['independent']['mean_selected_probes']:.2f} |
| same-neighborhood | {probes['same_neighborhood']['tasks_with_disagreement']}/24 | {probes['same_neighborhood']['mean_selected_probes']:.2f} |

Covered independent tasks were {[row['task_id'] for row in rows if row['independent_hidden_correct_count'] > 0]}. Covered same-neighborhood tasks were {[row['task_id'] for row in rows if row['same_hidden_correct_count'] > 0]}.

The key failure mode is not just a conservative threshold. Independent adaptations often produced no source-agreement cluster after target-independent disagreement probes. When agreement existed in same-neighborhood candidates, it was often agreement on the wrong behavior.

## Cost

![Tokens vs coverage](figures/tokens_vs_coverage.png)

- Independent retrieval-adapt: {consensus['token_usage']['independent']['forward_tokens']:,} forward tokens.
- Same-neighborhood retrieval-adapt: {consensus['token_usage']['same_neighborhood']['forward_tokens']:,} forward tokens.
- Direct K12 sample-more: {direct_manifest['token_usage']['forward_tokens']:,} forward tokens.

The strongest coverage/cost point here is same-neighborhood retrieval-adapt, not independent consensus. It still does not solve deployable selection.

## Interpretation

The hypothesis was plausible: independent derivations agreeing should provide evidence unavailable to any single-candidate judge. The implementation successfully increased retrieval independence, but that independence did not translate into useful agreement. Instead, it mostly reduced relevance and coverage. The non-independent control remained more coverage-effective, and consensus over it did not suppress false-pass enough to beat first-visible.

This is a negative for independent-retrieval consensus as implemented here. The result narrows the next direction: the missing evidence probably needs either stronger, task-targeted counterexample generation with a real oracle, or a larger/higher-quality verified library where multiple genuinely relevant independent algorithms exist for the same residual task. Merely diversifying among the current 364 MBPP-derived algorithms did not produce independent correct convergence.

## Artifacts

- Data: `data/`
- Run logs: `run_logs/`
- Summaries and figures: `reports/`
- Experiment log: `logs/experiment_log.md`
- Large artifact manifest: `large_artifacts_manifest.md`
"""
    (REPORTS / "final_report.md").write_text(report)


if __name__ == "__main__":
    main()

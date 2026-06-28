#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl, write_json  # noqa: E402


EXP = "qwen35_4b_retrieval_adapt_verify_scale"
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA = ROOT_DIR / "data"
REPORTS = ROOT_DIR / "reports"
FIGS = REPORTS / "figures"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_coverage(path: Path) -> tuple[int, list[int]]:
    rows = load_jsonl(path)
    tasks = [int(row["task_id"]) for row in rows if any(candidate.get("full_pass") for candidate in row.get("candidates", []))]
    return len(tasks), tasks


def visible_wrong(path: Path) -> tuple[int, int, float]:
    rows = load_jsonl(path)
    visible = 0
    wrong = 0
    for row in rows:
        for candidate in row.get("candidates", []):
            if candidate.get("visible_all_pass"):
                visible += 1
                if not candidate.get("full_pass"):
                    wrong += 1
    return visible, wrong, wrong / visible if visible else 0.0


def selector_counts(summary: dict[str, Any], name: str) -> tuple[int, int, int]:
    item = summary["selectors"][name]
    good = int(item["selected_hidden_correct"])
    bad = int(item["selected_visible_hidden_wrong"])
    no_commit = int(item["records"]) - int(item["commit_count"])
    return good, bad, no_commit


def qwen_counts(summary: dict[str, Any]) -> tuple[int, int, int]:
    good = int(summary["selected_hidden_correct"])
    bad = int(summary["selected_visible_hidden_wrong"])
    no_commit = int(summary["records"]) - int(summary["commit_count"])
    return good, bad, no_commit


def make_figures(summary: dict[str, Any]) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    arms = ["copy", "semantic", "random", "shuffled"]
    values = [
        summary["pool_coverage_counts"]["copy"],
        summary["pool_coverage_counts"]["semantic"],
        summary["pool_coverage_counts"]["random"],
        summary["pool_coverage_counts"]["shuffled"],
    ]
    colors = ["#999999", "#2f6f9f", "#b2772c", "#8b5fbf"]
    plt.figure(figsize=(7, 4))
    plt.bar(arms, values, color=colors)
    plt.axhline(summary["base_miss_count"], color="#444444", linewidth=1, linestyle=":")
    plt.ylabel("Recovered residual tasks / 24")
    plt.title("Residual candidate-pool coverage")
    for idx, value in enumerate(values):
        plt.text(idx, value + 0.15, str(value), ha="center")
    plt.tight_layout()
    plt.savefig(FIGS / "pool_coverage_by_arm.png", dpi=160)
    plt.close()

    labels = ["first", "consensus", "shortest", "qwen", "oracle"]
    triples = [
        summary["selector_counts"]["first_visible"],
        summary["selector_counts"]["consensus_visible"],
        summary["selector_counts"]["shortest_visible"],
        summary["selector_counts"]["frozen_qwen_visible_rerank"],
        summary["selector_counts"]["oracle_hidden"],
    ]
    good = [row["correct"] for row in triples]
    bad = [row["wrong_visible"] for row in triples]
    no_commit = [row["no_commit"] for row in triples]
    x = range(len(labels))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x, good, label="hidden-correct commit", color="#2f8f5b")
    plt.bar(x, bad, bottom=good, label="visible-pass hidden-wrong commit", color="#c75f46")
    plt.bar(x, no_commit, bottom=[a + b for a, b in zip(good, bad)], label="no commit", color="#cfcfcf")
    plt.xticks(list(x), labels)
    plt.ylabel("Residual tasks")
    plt.title("Deployable selector outcomes")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGS / "selector_outcomes.png", dpi=160)
    plt.close()

    fp_labels = ["copy+semantic", "semantic", "random", "shuffled"]
    fp_values = [
        summary["false_pass_rates"]["copy_semantic"],
        summary["false_pass_rates"]["semantic"],
        summary["false_pass_rates"]["random"],
        summary["false_pass_rates"]["shuffled"],
    ]
    plt.figure(figsize=(7, 4))
    plt.bar(fp_labels, [100 * value for value in fp_values], color=["#2f6f9f", "#3b8eb8", "#b2772c", "#8b5fbf"])
    plt.ylabel("Visible-pass hidden-wrong rate (%)")
    plt.title("False-pass risk among visible-pass candidates")
    plt.ylim(0, 100)
    for idx, value in enumerate(fp_values):
        plt.text(idx, 100 * value + 2, f"{100 * value:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGS / "false_pass_rates.png", dpi=160)
    plt.close()

    tokens = [
        summary["forward_tokens"]["semantic"],
        summary["forward_tokens"]["random"],
        summary["forward_tokens"]["shuffled"],
    ]
    rec = [
        summary["pool_coverage_counts"]["semantic"],
        summary["pool_coverage_counts"]["random"],
        summary["pool_coverage_counts"]["shuffled"],
    ]
    names = ["semantic", "random", "shuffled"]
    plt.figure(figsize=(6, 4))
    plt.scatter(tokens, rec, s=90, color=["#2f6f9f", "#b2772c", "#8b5fbf"])
    for token, value, name in zip(tokens, rec, names):
        plt.text(token + 120, value + 0.05, name, fontsize=9)
    plt.xlabel("Forward tokens")
    plt.ylabel("Recovered residual tasks")
    plt.title("Adaptation cost vs pool recovery")
    plt.tight_layout()
    plt.savefig(FIGS / "tokens_vs_recovery.png", dpi=160)
    plt.close()


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    base_manifest = load_json(DATA / "base_direct_k4_records.manifest.json")
    library = load_json(REPORTS / "library_summary.json")
    plan = load_json(REPORTS / "retrieval_plan_summary.json")
    selector_main = load_json(REPORTS / "selector_copy_semantic_summary.json")
    selector_semantic = load_json(REPORTS / "selector_semantic_summary.json")
    selector_random = load_json(REPORTS / "selector_random_summary.json")
    selector_shuffled = load_json(REPORTS / "selector_shuffled_summary.json")
    qwen = load_json(REPORTS / "qwen_rerank_copy_semantic_summary.json")

    arm_paths = {
        "copy": DATA / "retrieval_copy_rename_top3_records.jsonl",
        "semantic": DATA / "retrieval_adapt_semantic_top3_records.jsonl",
        "random": DATA / "retrieval_adapt_random_top3_records.jsonl",
        "shuffled": DATA / "retrieval_adapt_shuffled_top3_records.jsonl",
    }
    pool_counts: dict[str, int] = {}
    pool_tasks: dict[str, list[int]] = {}
    false_pass: dict[str, float] = {}
    false_pass_counts: dict[str, dict[str, int]] = {}
    forward_tokens: dict[str, int] = {"copy": 0}
    for name, path in arm_paths.items():
        count, tasks = count_coverage(path)
        pool_counts[name] = count
        pool_tasks[name] = tasks
        visible, wrong, rate = visible_wrong(path)
        false_pass[name] = rate
        false_pass_counts[name] = {"visible_pass": visible, "hidden_wrong": wrong}
        if name != "copy":
            manifest = load_json(path.with_suffix(".manifest.json"))
            forward_tokens[name] = int(manifest["token_usage"]["forward_tokens"])

    copy_sem_visible = int(selector_main["residual_pool"]["visible_pass_count"])
    copy_sem_wrong = int(selector_main["residual_pool"]["visible_hidden_wrong_count"])
    false_pass["copy_semantic"] = copy_sem_wrong / copy_sem_visible if copy_sem_visible else 0.0
    false_pass_counts["copy_semantic"] = {"visible_pass": copy_sem_visible, "hidden_wrong": copy_sem_wrong}

    control_union = set(pool_tasks["copy"]) | set(pool_tasks["random"]) | set(pool_tasks["shuffled"])
    semantic_unique = sorted(set(pool_tasks["semantic"]) - control_union)
    semantic_plus_copy = sorted(set(pool_tasks["semantic"]) | set(pool_tasks["copy"]))

    selector_summary = {
        "first_visible": dict(zip(["correct", "wrong_visible", "no_commit"], selector_counts(selector_main, "first_visible"))),
        "consensus_visible": dict(zip(["correct", "wrong_visible", "no_commit"], selector_counts(selector_main, "consensus_visible"))),
        "shortest_visible": dict(zip(["correct", "wrong_visible", "no_commit"], selector_counts(selector_main, "shortest_visible"))),
        "oracle_hidden": dict(zip(["correct", "wrong_visible", "no_commit"], selector_counts(selector_main, "oracle_hidden"))),
        "frozen_qwen_visible_rerank": dict(zip(["correct", "wrong_visible", "no_commit"], qwen_counts(qwen))),
    }

    summary: dict[str, Any] = {
        "experiment": EXP,
        "base_records": int(base_manifest["records"]["records"]),
        "base_covered": int(round(base_manifest["records"]["coverage"] * base_manifest["records"]["records"])),
        "base_coverage": base_manifest["records"]["coverage"],
        "base_miss_count": int(base_manifest["miss_count"]),
        "base_miss_tasks": base_manifest["miss_tasks"],
        "library_entries": int(library["library_entries"]),
        "retrieval_plan_records": int(plan["eval_records"]),
        "pool_coverage_counts": pool_counts,
        "pool_coverage_tasks": pool_tasks,
        "semantic_unique_vs_copy_random_shuffled": semantic_unique,
        "semantic_plus_copy_tasks": semantic_plus_copy,
        "combined_oracle_coverage_all_tasks": (int(round(base_manifest["records"]["coverage"] * base_manifest["records"]["records"])) + len(set(pool_tasks["semantic"]))) / int(base_manifest["records"]["records"]),
        "false_pass_rates": false_pass,
        "false_pass_counts": false_pass_counts,
        "selector_counts": selector_summary,
        "selector_selected_recovery_rates": {
            name: value["correct"] / int(base_manifest["miss_count"]) for name, value in selector_summary.items()
        },
        "forward_tokens": forward_tokens,
        "qwen_rerank_forward_tokens": int(qwen["token_usage"]["forward_tokens"]),
        "primary_read": "semantic retrieval+adaptation improves residual pool coverage, but deployable reranking still fails to remove false visible-pass candidates",
    }
    make_figures(summary)
    write_json(REPORTS / "report_summary.json", summary)

    report = f"""# {EXP}

## Motivation

Direct Qwen3.5-4B sampling covers many MBPP held-out tasks, but leaves a residual set with no hidden-correct candidate in the sample pool. This experiment tests an external-memory route around that gap: retrieve verified train-library algorithms, ask Qwen to adapt them to each residual task, then measure both hidden-test pool coverage and deployable selection.

Hidden tests are used only for evaluation and oracle ceilings. Public tests, candidate code, and target-independent agreement probes are the only deployable evidence used by selectors.

## Setup

- Base pool: {summary['base_records']} MBPP held-out tasks, K=4 direct samples per task.
- Base coverage: {summary['base_covered']}/{summary['base_records']} ({100 * summary['base_coverage']:.1f}%).
- Residual tasks: {summary['base_miss_count']} direct-sampling misses.
- Verified algorithm library: {summary['library_entries']} MBPP train references.
- Retrieval: TF-IDF top-3 semantic retrieval, plus random and shuffled-query controls.
- Adaptation: one Qwen3.5-4B completion per retrieved algorithm.
- Selector pool: copy/rename top-3 plus semantic adaptations.

## Candidate-Pool Coverage

| arm | residual recovered | rate | recovered tasks | forward tokens |
|---|---:|---:|---|---:|
| copy/rename top-3 | {pool_counts['copy']}/24 | {100 * pool_counts['copy'] / 24:.1f}% | {pool_tasks['copy']} | 0 |
| semantic adapt top-3 | {pool_counts['semantic']}/24 | {100 * pool_counts['semantic'] / 24:.1f}% | {pool_tasks['semantic']} | {forward_tokens['semantic']} |
| random adapt top-3 | {pool_counts['random']}/24 | {100 * pool_counts['random'] / 24:.1f}% | {pool_tasks['random']} | {forward_tokens['random']} |
| shuffled-query adapt top-3 | {pool_counts['shuffled']}/24 | {100 * pool_counts['shuffled'] / 24:.1f}% | {pool_tasks['shuffled']} | {forward_tokens['shuffled']} |

Semantic retrieval is the strongest pool-coverage arm: 8/24 residual recoveries versus 4/24 random and 3/24 shuffled. The semantic-only recoveries beyond copy/random/shuffled are {semantic_unique}. If those hidden-correct candidates were selectable perfectly, all-task coverage would rise from 56/80 to 64/80 ({100 * summary['combined_oracle_coverage_all_tasks']:.1f}%).

![Pool coverage](figures/pool_coverage_by_arm.png)

## Selection and False Passes

The caveat is still visible-pass hidden-wrong noise. In the main copy+semantic pool, {copy_sem_wrong}/{copy_sem_visible} visible-pass candidates fail hidden tests ({100 * false_pass['copy_semantic']:.1f}%).

| selector | correct residual commits | wrong visible-pass commits | no commit | selected recovery |
|---|---:|---:|---:|---:|
| first visible | {selector_summary['first_visible']['correct']} | {selector_summary['first_visible']['wrong_visible']} | {selector_summary['first_visible']['no_commit']} | {100 * summary['selector_selected_recovery_rates']['first_visible']:.1f}% |
| consensus visible | {selector_summary['consensus_visible']['correct']} | {selector_summary['consensus_visible']['wrong_visible']} | {selector_summary['consensus_visible']['no_commit']} | {100 * summary['selector_selected_recovery_rates']['consensus_visible']:.1f}% |
| shortest visible | {selector_summary['shortest_visible']['correct']} | {selector_summary['shortest_visible']['wrong_visible']} | {selector_summary['shortest_visible']['no_commit']} | {100 * summary['selector_selected_recovery_rates']['shortest_visible']:.1f}% |
| frozen-Qwen rerank | {selector_summary['frozen_qwen_visible_rerank']['correct']} | {selector_summary['frozen_qwen_visible_rerank']['wrong_visible']} | {selector_summary['frozen_qwen_visible_rerank']['no_commit']} | {100 * summary['selector_selected_recovery_rates']['frozen_qwen_visible_rerank']:.1f}% |
| hidden oracle | {selector_summary['oracle_hidden']['correct']} | {selector_summary['oracle_hidden']['wrong_visible']} | {selector_summary['oracle_hidden']['no_commit']} | {100 * summary['selector_selected_recovery_rates']['oracle_hidden']:.1f}% |

The simplest deployable selector, first-visible, captures 7/8 oracle recoveries but also commits 7 hidden-wrong visible passers. Target-independent agreement probes do not help here, and the frozen-Qwen reranker is worse than first-visible.

![Selector outcomes](figures/selector_outcomes.png)

![False-pass rates](figures/false_pass_rates.png)

![Tokens vs recovery](figures/tokens_vs_recovery.png)

## Interpretation

This is a positive coverage result and a negative selector result.

The positive part is that semantic retrieval plus Qwen adaptation works on this 24-task residual scale: it recovers a third of the direct-sampling residual, doubles random-retrieval coverage, beats shuffled-query retrieval, and adds four control-clean residual tasks. This supports the external algorithmic-memory direction: some misses are not beyond adaptation; they are missing the right algorithmic hint.

The negative part is deployable selection. Public tests are too thin: most visible-pass candidates in the main pool are hidden-wrong, and neither agreement probes nor a frozen-Qwen reranker reduce that risk. The main bottleneck after retrieval is not generating a candidate; it is obtaining enough trustworthy evidence to commit it.

## Next Direction

The next high-value run should keep semantic retrieval+adaptation, but replace weak public-test selection with stronger deployable evidence:

1. generate or mine counterexample tests with output agreement, not expected answers;
2. require candidates to survive multiple independently retrieved/adapted implementations by consensus;
3. use a verifier only after the evidence set is enlarged, because code-only reranking did not separate correct from hidden-wrong candidates here.

## Artifacts

- `data/base_direct_k4_records.jsonl`
- `data/retrieval_plan.jsonl`
- `data/retrieval_adapt_semantic_top3_records.jsonl`
- `data/retrieval_adapt_random_top3_records.jsonl`
- `data/retrieval_adapt_shuffled_top3_records.jsonl`
- `data/selector_copy_semantic_records.jsonl`
- `data/qwen_rerank_copy_semantic_records.jsonl`
- `reports/report_summary.json`
"""
    (REPORTS / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

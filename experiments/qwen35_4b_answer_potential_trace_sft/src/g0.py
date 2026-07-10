"""Preregistered G0 scorer validation and prefix-checkpoint construction."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

from model_ops import answer_mention
from stats import kendall_tau_b, mean, paired_bootstrap, roc_auc


def attach_scores_and_rollouts(
    traces: Sequence[Mapping[str, Any]],
    potentials: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    score_by_trace = {str(row["trace_id"]): row for row in potentials}
    rollout_by_trace = {str(row["trace_id"]): row for row in rollouts}
    joined: list[dict[str, Any]] = []
    for trace in traces:
        trace_id = str(trace["trace_id"])
        if trace_id not in score_by_trace:
            continue
        if trace_id not in rollout_by_trace:
            raise ValueError(f"missing rollout row for scored trace {trace_id}")
        joined.append({**dict(trace), **dict(score_by_trace[trace_id]), **dict(rollout_by_trace[trace_id])})
    return joined


def selected_top_by_gain(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    return {
        task: max(group, key=lambda row: (float(row["gain_sum"]), -int(row["n_tokens"]), str(row["trace_id"])))
        for task, group in by_task.items()
    }


def make_premember_checkpoints(
    selected: Mapping[str, Mapping[str, Any]],
    item_by_id: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Make the latest natural token boundary before a verbatim answer mention."""
    controls: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    for task_id, row in selected.items():
        item = item_by_id[task_id]
        answer = str(item["canonical_answer"])
        text = str(row.get("text", ""))
        mention = answer_mention(text, answer)
        diagnostics[task_id] = {
            "trace_id": row["trace_id"],
            "answer": answer,
            "mention_char": mention,
            "no_answer_mention": mention is None,
        }
        if mention is None:
            continue
        token_ids = [int(value) for value in row["token_ids"]]
        candidates: list[tuple[int, str]] = []
        # Only selected traces are scanned, so exact prefix decoding is cheap
        # and avoids approximate character/token alignment.
        for index in range(1, len(token_ids) + 1):
            prefix_text = tokenizer.decode(token_ids[:index], skip_special_tokens=False)
            if len(prefix_text) > mention:
                break
            if re.search(r"(?:[.!?;:]|\n)\s*$", prefix_text):
                candidates.append((index, prefix_text))
        if not candidates:
            diagnostics[task_id]["checkpoint"] = None
            continue
        index, prefix_text = candidates[-1]
        checkpoint = {
            **dict(row),
            "trace_id": f"{row['trace_id']}::premember",
            "source_trace_id": row["trace_id"],
            "condition": "premember_checkpoint",
            "token_ids": token_ids[:index],
            "text": prefix_text,
            "n_tokens": index,
        }
        controls.append(checkpoint)
        diagnostics[task_id]["checkpoint"] = {
            "trace_id": checkpoint["trace_id"],
            "n_tokens": index,
            "text_chars": len(prefix_text),
        }
    return controls, diagnostics


def _task_aurocs(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    output: dict[str, float] = {}
    for task_id, task_rows in by_task.items():
        labels: list[bool] = []
        scores: list[float] = []
        for row in task_rows:
            value = row.get(field)
            if value is None:
                continue
            for outcome in row["outcomes"]:
                labels.append(bool(outcome["correct"]))
                scores.append(float(value))
        auc = roc_auc(labels, scores)
        if auc is not None:
            output[task_id] = auc
    return output


def _paired_control_summary(
    originals: Sequence[Mapping[str, Any]], controls: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
) -> dict[str, Any]:
    original = {str(row["trace_id"]): row for row in originals}
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in controls:
        source_id = str(row["source_trace_id"])
        if source_id not in original:
            continue
        by_task[str(row["task_id"])].append(
            float(original[source_id]["gain_sum"]) - float(row["gain_sum"])
        )
    task_delta = {task: mean(values) for task, values in by_task.items()}
    pairs = {task: (delta, 0.0) for task, delta in task_delta.items()}
    return {
        "task_macro_mean_gain_advantage": mean(list(task_delta.values())),
        "bootstrap": paired_bootstrap(pairs, resamples=resamples, seed=seed),
    }


def evaluate_g0(
    *,
    traces: Sequence[Mapping[str, Any]],
    canonical_scores: Sequence[Mapping[str, Any]],
    format_scores: Sequence[Mapping[str, Any]],
    shuffled_scores: Sequence[Mapping[str, Any]],
    foreign_scores: Sequence[Mapping[str, Any]],
    rollout_rows: Sequence[Mapping[str, Any]],
    premention_scores: Sequence[Mapping[str, Any]],
    premention_diagnostics: Mapping[str, Mapping[str, Any]],
    auroc_min: float,
    uplift_min: float,
    kendall_min: float,
    premention_fraction_min: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    random_seed: int,
) -> dict[str, Any]:
    rows = attach_scores_and_rollouts(traces, canonical_scores, rollout_rows)
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)

    gain_auc_by_task = _task_aurocs(rows, "gain_sum")
    length_auc_by_task = _task_aurocs(
        [{**row, "negative_length": -int(row["n_tokens"])} for row in rows],
        "negative_length",
    )
    prior_auc_by_task = _task_aurocs(rows, "prior_logprob_mean")

    selected = selected_top_by_gain(rows)
    random_picker = random.Random(random_seed)
    top_values: dict[str, float] = {}
    random_values: dict[str, float] = {}
    shortest_values: dict[str, float] = {}
    for task_id in sorted(by_task):
        group = by_task[task_id]
        top_values[task_id] = float(selected[task_id]["success_fraction"])
        random_values[task_id] = float(
            group[random_picker.randrange(len(group))]["success_fraction"]
        )
        shortest = min(
            group,
            key=lambda row: (int(row["n_tokens"]), str(row["trace_id"])),
        )
        shortest_values[task_id] = float(shortest["success_fraction"])
    top_random_pairs = {
        task: (top_values[task], random_values[task]) for task in sorted(top_values)
    }
    top_short_pairs = {
        task: (top_values[task], shortest_values[task]) for task in sorted(top_values)
    }
    top_random_bootstrap = paired_bootstrap(
        top_random_pairs, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    top_short_bootstrap = paired_bootstrap(
        top_short_pairs, resamples=bootstrap_resamples, seed=bootstrap_seed + 1
    )

    canonical_by_trace = {str(row["trace_id"]): row for row in canonical_scores}
    format_by_trace = {str(row["trace_id"]): row for row in format_scores}
    tau_by_task: dict[str, float] = {}
    for task_id, group in by_task.items():
        trace_ids = [str(row["trace_id"]) for row in group]
        left = [float(canonical_by_trace[trace_id]["gain_sum"]) for trace_id in trace_ids]
        right = [float(format_by_trace[trace_id]["gain_sum"]) for trace_id in trace_ids]
        tau = kendall_tau_b(left, right)
        if tau is not None:
            tau_by_task[task_id] = tau

    premention_by_source = {
        str(row["source_trace_id"]): row for row in premention_scores
    }
    premention_outcomes: dict[str, bool] = {}
    for task_id, selected_row in selected.items():
        diagnostic = premention_diagnostics[task_id]
        if diagnostic["no_answer_mention"]:
            premention_outcomes[task_id] = True
            continue
        checkpoint = diagnostic.get("checkpoint")
        if checkpoint is None:
            premention_outcomes[task_id] = False
            continue
        source_id = str(selected_row["trace_id"])
        checkpoint_score = premention_by_source.get(source_id)
        premention_outcomes[task_id] = bool(
            checkpoint_score is not None and float(checkpoint_score["gain_sum"]) > 0.0
        )
    premention_fraction = mean([float(value) for value in premention_outcomes.values()])

    shuffled = _paired_control_summary(
        canonical_scores,
        shuffled_scores,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 2,
    )
    foreign = _paired_control_summary(
        canonical_scores,
        foreign_scores,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 3,
    )

    gain_auc_macro = mean(list(gain_auc_by_task.values()))
    length_auc_macro = mean(list(length_auc_by_task.values()))
    prior_auc_macro = (
        mean(list(prior_auc_by_task.values())) if prior_auc_by_task else None
    )
    metrics = {
        "n_tasks": len(by_task),
        "n_traces": len(rows),
        "within_task_auroc": {
            "gain_task_macro": gain_auc_macro,
            "negative_length_task_macro": length_auc_macro,
            "prior_mean_logprob_task_macro": prior_auc_macro,
            "prior_mean_logprob_available": bool(prior_auc_by_task),
            "mixed_tasks_gain": len(gain_auc_by_task),
            "by_task": {
                "gain": gain_auc_by_task,
                "negative_length": length_auc_by_task,
                "prior_mean_logprob": prior_auc_by_task,
            },
        },
        "top_one": {
            "gain": mean(list(top_values.values())),
            "seeded_random": mean(list(random_values.values())),
            "shortest": mean(list(shortest_values.values())),
            "gain_minus_random": top_random_bootstrap,
            "gain_minus_shortest": top_short_bootstrap,
        },
        "controls": {"token_shuffled": shuffled, "foreign": foreign},
        "format_kendall_tau": {
            "task_macro": mean(list(tau_by_task.values())),
            "n_tasks": len(tau_by_task),
            "by_task": tau_by_task,
        },
        "premember": {
            "fraction": premention_fraction,
            "n_tasks": len(premention_outcomes),
            "passed_by_task": premention_outcomes,
        },
    }
    criteria = {
        "auroc": gain_auc_macro >= auroc_min,
        "top1_random_uplift": (
            top_random_bootstrap["mean_delta"] >= uplift_min
            and top_random_bootstrap["ci95_low"] > 0.0
        ),
        "top1_shortest_uplift": (
            top_short_bootstrap["mean_delta"] >= uplift_min
            and top_short_bootstrap["ci95_low"] > 0.0
        ),
        "beats_length_and_prior_auroc": (
            prior_auc_macro is not None
            and gain_auc_macro > length_auc_macro
            and gain_auc_macro > prior_auc_macro
        ),
        "beats_token_shuffled": (
            shuffled["task_macro_mean_gain_advantage"] > 0.0
            and shuffled["bootstrap"]["ci95_low"] > 0.0
        ),
        "beats_foreign": (
            foreign["task_macro_mean_gain_advantage"] > 0.0
            and foreign["bootstrap"]["ci95_low"] > 0.0
        ),
        "format_rank_stability": metrics["format_kendall_tau"]["task_macro"] >= kendall_min,
        "premember": premention_fraction >= premention_fraction_min,
    }
    return {
        "schema_version": 1,
        "gate": "G0",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "thresholds": {
            "within_task_auroc_min": auroc_min,
            "top1_uplift_min": uplift_min,
            "bootstrap_lower_strictly_positive": True,
            "kendall_tau_min": kendall_min,
            "premember_fraction_min": premention_fraction_min,
        },
        "metrics": metrics,
        "selected_trace_ids": {
            task: row["trace_id"] for task, row in selected.items()
        },
    }

"""Metrics: unbiased pass@k plus deployable-vs-oracle selection over sampled candidates."""
from __future__ import annotations

from statistics import mean


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021). n samples, c correct."""
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    prob_all_fail = 1.0
    for i in range(k):
        prob_all_fail *= (n - c - i) / (n - i)
    return 1.0 - prob_all_fail


def summarize_condition(greedy_pass: list[bool],
                        sampled_pass: list[list[bool]],
                        sampled_visible: list[list[bool]],
                        n_think: list[float],
                        n_gen: list[float],
                        forced_frac: float,
                        k: int) -> dict:
    """Aggregate one condition across tasks.

    greedy_pass[t]            : did the single greedy sample pass full tests
    sampled_pass[t][s]        : full-test pass for each of the k sampled candidates
    sampled_visible[t][s]     : visible-test (first assert) pass for each candidate
    """
    n_tasks = len(sampled_pass)
    # deployable: single greedy sample
    greedy_at1 = mean(greedy_pass) if greedy_pass else float("nan")
    # sampled mean pass@1 and oracle pass@k
    p1 = mean(pass_at_k(len(p), sum(p), 1) for p in sampled_pass) if n_tasks else float("nan")
    pk = mean(pass_at_k(len(p), sum(p), k) for p in sampled_pass) if n_tasks else float("nan")
    # deployable selector: among candidates, pick first that passes the VISIBLE test; verify full.
    sel = []
    for full, vis in zip(sampled_pass, sampled_visible):
        chosen = next((i for i, v in enumerate(vis) if v), None)
        sel.append(full[chosen] if chosen is not None else False)
    selector_at1 = mean(sel) if sel else float("nan")
    return {
        "n_tasks": n_tasks,
        "greedy_pass@1": round(greedy_at1, 4),
        "sampled_pass@1": round(p1, 4),
        f"pass@{k}_oracle": round(pk, 4),
        "visible_selector@1": round(selector_at1, 4),
        "oracle_minus_deployable": round(pk - selector_at1, 4),
        "mean_think_tokens": round(mean(n_think), 1) if n_think else 0,
        "mean_total_tokens": round(mean(n_gen), 1) if n_gen else 0,
        "forced_close_frac": round(forced_frac, 3),
    }

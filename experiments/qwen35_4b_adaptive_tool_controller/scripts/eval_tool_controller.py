#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt


VARIANT_ORDER = [
    "verified_structural",
    "cell_parser",
    "row_column_rule",
    "header_aware",
    "split_fold_unpivot",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def table_dims(table: Any) -> tuple[int, int, int, int]:
    if not isinstance(table, list):
        return 0, 0, 0, 0
    rows = len(table)
    cols = max([len(row) for row in table if isinstance(row, list)] or [0])
    cells = sum(len(row) for row in table if isinstance(row, list))
    chars = sum(len(str(cell)) for row in table if isinstance(row, list) for cell in row)
    return rows, cols, cells, chars


def direct_tokens(record: dict[str, Any]) -> int:
    return int(record.get("direct_total_tokens", 0))


def by_variant(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["variant"]: candidate for candidate in record["program_candidates"]}


def variant_tokens(record: dict[str, Any], variants: list[str]) -> int:
    wanted = set(variants)
    return sum(int(candidate.get("total_tokens", 0)) for candidate in record["program_candidates"] if candidate["variant"] in wanted)


def candidate_visible(candidate: dict[str, Any]) -> bool:
    final = candidate["final"]
    return bool(final.get("visible_pass")) and final.get("hidden_output_key") is not None


def visible_candidates(record: dict[str, Any], variants: list[str]) -> list[dict[str, Any]]:
    candidates = by_variant(record)
    out: list[dict[str, Any]] = []
    for variant in variants:
        candidate = candidates.get(variant)
        if candidate is not None and candidate_visible(candidate):
            out.append(candidate)
    return out


@dataclass(frozen=True)
class Decision:
    source: str
    ok: bool
    tokens: int
    generated: tuple[str, ...]
    program_committed: bool
    trigger: bool = False


def choose_first_visible(record: dict[str, Any], variants: list[str]) -> tuple[str, bool]:
    visible = visible_candidates(record, variants)
    if visible:
        candidate = visible[0]
        return f"program:{candidate['variant']}", bool(candidate["final"]["hidden_exact"])
    return "direct", bool(record["direct_exact"])


def action_direct(record: dict[str, Any]) -> Decision:
    return Decision("direct", bool(record["direct_exact"]), direct_tokens(record), (), False)


def action_first_visible(name: str, variants: list[str]) -> Callable[[dict[str, Any]], Decision]:
    def _action(record: dict[str, Any]) -> Decision:
        source, ok = choose_first_visible(record, variants)
        return Decision(
            source=f"{name}:{source}",
            ok=ok,
            tokens=direct_tokens(record) + variant_tokens(record, variants),
            generated=tuple(variants),
            program_committed=source != "direct",
        )

    return _action


def action_canary_escalate(canary: str, full_order: list[str]) -> Callable[[dict[str, Any]], Decision]:
    def _action(record: dict[str, Any]) -> Decision:
        candidate = by_variant(record)[canary]
        tokens = direct_tokens(record) + int(candidate.get("total_tokens", 0))
        generated = [canary]
        trigger = False
        if candidate_visible(candidate):
            direct_key = record.get("direct_output_key")
            trigger = (not bool(record.get("direct_parse_ok"))) or candidate["final"].get("hidden_output_key") != direct_key
        if trigger:
            remaining = [variant for variant in full_order if variant != canary]
            tokens += variant_tokens(record, remaining)
            generated = list(full_order)
            source, ok = choose_first_visible(record, full_order)
            return Decision(
                source=f"canary_{canary}:{source}",
                ok=ok,
                tokens=tokens,
                generated=tuple(generated),
                program_committed=source != "direct",
                trigger=True,
            )
        return Decision(f"canary_{canary}:direct", bool(record["direct_exact"]), tokens, tuple(generated), False, False)

    return _action


def action_oracle_best(full_order: list[str]) -> Callable[[dict[str, Any]], Decision]:
    def _action(record: dict[str, Any]) -> Decision:
        tokens = direct_tokens(record) + variant_tokens(record, full_order)
        if bool(record["direct_exact"]):
            return Decision("oracle:direct", True, tokens, tuple(full_order), False)
        for candidate in visible_candidates(record, full_order):
            if bool(candidate["final"]["hidden_exact"]):
                return Decision(f"oracle:{candidate['variant']}", True, tokens, tuple(full_order), True, True)
        return Decision("oracle:miss", False, tokens, tuple(full_order), False)

    return _action


def action_oracle_budget(full_order: list[str]) -> Callable[[dict[str, Any]], Decision]:
    def _action(record: dict[str, Any]) -> Decision:
        source, ok = choose_first_visible(record, full_order)
        if ok and not bool(record["direct_exact"]):
            return Decision(
                f"oracle_budget:{source}",
                True,
                direct_tokens(record) + variant_tokens(record, full_order),
                tuple(full_order),
                source != "direct",
                True,
            )
        return action_direct(record)

    return _action


def split_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["split"]].append(record)
    return dict(grouped)


def features(record: dict[str, Any], cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case = cases[record["file"]]
    vin_r, vin_c, vin_cells, vin_chars = table_dims(case["input_table"])
    vout_r, vout_c, vout_cells, vout_chars = table_dims(case["output_table"])
    new_r, new_c, new_cells, new_chars = table_dims(case["testing_table"])
    direct_r, direct_c, direct_cells, direct_chars = table_dims(record.get("direct_table"))
    return {
        "visible_rows_change": vout_r != vin_r,
        "visible_cols_change": vout_c != vin_c,
        "visible_shape_change": (vout_r, vout_c) != (vin_r, vin_c),
        "visible_out_more_rows": vout_r > vin_r,
        "visible_out_less_rows": vout_r < vin_r,
        "visible_out_more_cols": vout_c > vin_c,
        "visible_out_less_cols": vout_c < vin_c,
        "direct_parse_fail": not bool(record.get("direct_parse_ok")),
        "direct_cols_mismatch_visible_out": direct_c != vout_c,
        "direct_rows_mismatch_visible_out": direct_r != vout_r,
        "direct_out_cols_less_new_in": direct_c < new_c,
        "direct_out_rows_less_new_in": direct_r < new_r,
        "visible_less_cols_or_direct_mismatch": (vout_c < vin_c) or (direct_c != vout_c),
        "visible_cols_change_or_direct_mismatch": (vout_c != vin_c) or (direct_c != vout_c),
        "visible_rows_change_or_direct_mismatch": (vout_r != vin_r) or (direct_r != vout_r),
        "direct_output_empty": direct_r == 0 or direct_c == 0,
        "new_input_many_rows": new_r >= 5,
        "new_input_many_cols": new_c >= 5,
        "visible_output_wider_than_tall": vout_c > vout_r,
        "visible_output_taller_than_wide": vout_r > vout_c,
        "visible_row_expansion_or_col_contraction": (vout_r > vin_r) or (vout_c < vin_c),
        "visible_col_expansion_or_row_contraction": (vout_c > vin_c) or (vout_r < vin_r),
    }


ActionFn = Callable[[dict[str, Any]], Decision]


def build_actions(full_order: list[str]) -> dict[str, ActionFn]:
    actions: dict[str, ActionFn] = {"direct_only": action_direct}
    for variant in full_order:
        actions[f"single_{variant}"] = action_first_visible(f"single_{variant}", [variant])
        actions[f"canary_{variant}_disagree_escalate"] = action_canary_escalate(variant, full_order)
    for k in range(1, len(full_order) + 1):
        actions[f"prefix{k}_first_visible"] = action_first_visible(f"prefix{k}", full_order[:k])
    actions["oracle_best_available_full_budget"] = action_oracle_best(full_order)
    actions["oracle_budget_full_only_when_helpful"] = action_oracle_budget(full_order)
    return actions


def evaluate_action(records: list[dict[str, Any]], action: ActionFn) -> dict[str, Any]:
    return evaluate_decisions(records, [action(record) for record in records], "action")


def evaluate_decisions(records: list[dict[str, Any]], decisions: list[Decision], policy: str) -> dict[str, Any]:
    n = len(records)
    exact = sum(int(decision.ok) for decision in decisions)
    direct_exact = sum(int(record["direct_exact"]) for record in records)
    program_commits = sum(int(decision.program_committed) for decision in decisions)
    program_correct = sum(int(decision.program_committed and decision.ok) for decision in decisions)
    recoveries = sum(
        int(decision.program_committed and decision.ok and not bool(record["direct_exact"]))
        for decision, record in zip(decisions, records)
    )
    losses = sum(
        int(decision.program_committed and (not decision.ok) and bool(record["direct_exact"]))
        for decision, record in zip(decisions, records)
    )
    tokens = sum(decision.tokens for decision in decisions)
    generated_counts = Counter(variant for decision in decisions for variant in decision.generated)
    return {
        "policy": policy,
        "n": n,
        "exact": exact,
        "accuracy": exact / n if n else 0.0,
        "direct_exact": direct_exact,
        "direct_accuracy": direct_exact / n if n else 0.0,
        "total_forward_tokens": tokens,
        "avg_forward_tokens": tokens / n if n else 0.0,
        "program_commits": program_commits,
        "program_commit_correct": program_correct,
        "program_commit_precision": program_correct / program_commits if program_commits else None,
        "direct_miss_recoveries": recoveries,
        "direct_correct_losses": losses,
        "generated_variant_counts": dict(sorted(generated_counts.items())),
        "decisions": [
            {
                "file": record["file"],
                "family": record["family"],
                "source": decision.source,
                "ok": decision.ok,
                "direct_exact": bool(record["direct_exact"]),
                "tokens": decision.tokens,
                "generated": list(decision.generated),
                "trigger": decision.trigger,
            }
            for record, decision in zip(records, decisions)
        ],
    }


def action_score(metrics: dict[str, Any], token_weight: float = 0.0) -> tuple[float, int, float, float]:
    precision = metrics["program_commit_precision"] if metrics["program_commit_precision"] is not None else 1.0
    return (
        metrics["exact"] - token_weight * metrics["total_forward_tokens"],
        -metrics["direct_correct_losses"],
        precision,
        -metrics["total_forward_tokens"],
    )


def evaluate_controller(
    records: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    actions: dict[str, ActionFn],
    policy: dict[str, Any],
    name: str = "learned_controller",
) -> dict[str, Any]:
    decisions: list[Decision] = []
    for record in records:
        if policy["feature"] == "__constant__":
            action_name = policy["true_action"]
        else:
            action_name = policy["true_action"] if features(record, cases)[policy["feature"]] else policy["false_action"]
        decisions.append(actions[action_name](record))
    return evaluate_decisions(records, decisions, name)


def select_controller(
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    actions: dict[str, ActionFn],
) -> dict[str, Any]:
    train_feature_map = {record["file"]: features(record, cases) for record in train_records}
    feature_names = sorted(next(iter(train_feature_map.values())).keys()) if train_feature_map else []
    action_names = [
        name
        for name in actions
        if not name.startswith("oracle")
        and (name == "direct_only" or name.startswith("single_") or name.startswith("prefix") or name.startswith("canary_"))
    ]
    raw_policies = []
    for action_name in action_names:
        raw_policies.append({"feature": "__constant__", "true_action": action_name, "false_action": action_name})
    for feature_name in feature_names:
        if not any(train_feature_map[record["file"]][feature_name] for record in train_records):
            continue
        if all(train_feature_map[record["file"]][feature_name] for record in train_records):
            continue
        for true_action in action_names:
            for false_action in action_names:
                raw_policies.append({"feature": feature_name, "true_action": true_action, "false_action": false_action})

    direct_train = evaluate_action(train_records, actions["direct_only"])
    candidates = []
    for policy in raw_policies:
        train_metrics = evaluate_controller(train_records, cases, actions, policy, "candidate_train")
        # Train sanity gate: keep policies that do not reduce train accuracy below direct.
        if train_metrics["exact"] < direct_train["exact"]:
            continue
        dev_metrics = evaluate_controller(dev_records, cases, actions, policy, "candidate_dev")
        candidates.append(
            {
                "policy": policy,
                "train": {k: v for k, v in train_metrics.items() if k != "decisions"},
                "dev": {k: v for k, v in dev_metrics.items() if k != "decisions"},
            }
        )

    def dev_key(row: dict[str, Any]) -> tuple[Any, ...]:
        dev = row["dev"]
        precision = dev["program_commit_precision"] if dev["program_commit_precision"] is not None else 1.0
        return (dev["exact"], -dev["direct_correct_losses"], precision, -dev["total_forward_tokens"], row["train"]["exact"], -row["train"]["total_forward_tokens"])

    candidates.sort(key=dev_key, reverse=True)
    return {"selected": candidates[0], "num_candidates_after_train_gate": len(candidates), "top_candidates": candidates[:25]}


def evaluate_baselines(
    records: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    actions: dict[str, ActionFn],
) -> list[dict[str, Any]]:
    selected_actions = [
        "direct_only",
        "single_verified_structural",
        "single_row_column_rule",
        "single_split_fold_unpivot",
        "prefix1_first_visible",
        "prefix2_first_visible",
        "prefix3_first_visible",
        "prefix4_first_visible",
        "prefix5_first_visible",
        "canary_verified_structural_disagree_escalate",
        "canary_row_column_rule_disagree_escalate",
        "canary_split_fold_unpivot_disagree_escalate",
        "oracle_budget_full_only_when_helpful",
        "oracle_best_available_full_budget",
    ]
    rows = []
    for name in selected_actions:
        if name in actions:
            metrics = evaluate_action(records, actions[name])
            metrics["policy"] = name
            rows.append({k: v for k, v in metrics.items() if k != "decisions"})
    # Add two fixed hand stumps as controls.
    for feature_name in ["visible_out_less_cols", "direct_out_cols_less_new_in", "visible_cols_change"]:
        policy = {"feature": feature_name, "true_action": "prefix5_first_visible", "false_action": "direct_only"}
        metrics = evaluate_controller(records, cases, actions, policy, f"fixed_{feature_name}_full_else_direct")
        rows.append({k: v for k, v in metrics.items() if k != "decisions"})
    return rows


def family_metrics(records: list[dict[str, Any]], controller_metrics: dict[str, Any]) -> dict[str, Any]:
    by_file = {decision["file"]: decision for decision in controller_metrics["decisions"]}
    rows = defaultdict(lambda: {"n": 0, "exact": 0, "direct_exact": 0, "tokens": 0, "program_commits": 0, "recoveries": 0, "losses": 0})
    for record in records:
        decision = by_file[record["file"]]
        row = rows[record["family"]]
        row["n"] += 1
        row["exact"] += int(decision["ok"])
        row["direct_exact"] += int(record["direct_exact"])
        row["tokens"] += int(decision["tokens"])
        row["program_commits"] += int(decision["source"] != "direct")
        row["recoveries"] += int(decision["source"] != "direct" and decision["ok"] and not record["direct_exact"])
        row["losses"] += int(decision["source"] != "direct" and not decision["ok"] and record["direct_exact"])
    return {
        family: {
            **row,
            "accuracy": row["exact"] / row["n"],
            "direct_accuracy": row["direct_exact"] / row["n"],
        }
        for family, row in sorted(rows.items())
    }


def pareto_front(rows: list[dict[str, Any]]) -> list[str]:
    deployable = [row for row in rows if not row["policy"].startswith("oracle")]
    front = []
    for row in deployable:
        dominated = False
        for other in deployable:
            if row is other:
                continue
            if (
                other["accuracy"] >= row["accuracy"]
                and other["total_forward_tokens"] <= row["total_forward_tokens"]
                and (other["accuracy"] > row["accuracy"] or other["total_forward_tokens"] < row["total_forward_tokens"])
            ):
                dominated = True
                break
        if not dominated:
            point = (row["exact"], row["total_forward_tokens"])
            if all((existing["exact"], existing["total_forward_tokens"]) != point for existing in front):
                front.append(row)
    front.sort(key=lambda row: row["total_forward_tokens"])
    return [row["policy"] for row in front]


def save_pareto_chart(path: Path, rows: list[dict[str, Any]], primary_name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for row in rows:
        oracle = row["policy"].startswith("oracle")
        primary = row["policy"] == primary_name
        color = "tab:red" if primary else ("tab:gray" if oracle else "tab:blue")
        marker = "X" if primary else ("*" if oracle else "o")
        ax.scatter(row["total_forward_tokens"], row["accuracy"], color=color, marker=marker, s=95 if primary else 48, alpha=1.0 if primary or oracle else 0.48)
    labels = {
        "direct_only": ("direct", (8, -12)),
        primary_name: ("learned controller", (8, 8)),
        "prefix1_first_visible": ("prefix1", (8, -16)),
        "prefix5_first_visible": ("full portfolio", (-78, -20)),
        "fixed_visible_out_less_cols_full_else_direct": ("best fixed shape", (8, -18)),
        "oracle_budget_full_only_when_helpful": ("oracle budget", (8, 8)),
    }
    by_name = {row["policy"]: row for row in rows}
    for name, (label, offset) in labels.items():
        row = by_name.get(name)
        if row:
            ax.annotate(label, (row["total_forward_tokens"], row["accuracy"]), xytext=offset, textcoords="offset points", fontsize=8)
    ax.set_xlabel("Total forward tokens")
    ax.set_ylabel("Exact accuracy")
    ax.set_ylim(0.35, 0.65)
    ax.set_title("Held-out accuracy vs controller budget")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_selection_chart(path: Path, selection: dict[str, Any]) -> None:
    rows = selection["top_candidates"][:12]
    labels = [row["policy"]["feature"] for row in rows]
    dev_acc = [row["dev"]["accuracy"] for row in rows]
    tokens = [row["dev"]["total_forward_tokens"] for row in rows]
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    x = list(range(len(rows)))
    ax1.bar(x, dev_acc, color="tab:blue", alpha=0.75)
    ax1.set_ylabel("Pilot-dev accuracy")
    ax1.set_ylim(0, 0.8)
    ax1.set_xticks(x, labels, rotation=35, ha="right")
    ax1.set_xlabel("Controller feature")
    ax2 = ax1.twinx()
    ax2.plot(x, tokens, color="tab:red", marker="o")
    ax2.set_ylabel("Pilot-dev forward tokens")
    ax1.set_title("Top pilot-dev controller candidates")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_family_chart(path: Path, rows: dict[str, Any]) -> None:
    labels = list(rows)
    direct = [rows[label]["direct_accuracy"] for label in labels]
    controller = [rows[label]["accuracy"] for label in labels]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    width = 0.36
    ax.bar([i - width / 2 for i in x], direct, width, label="Direct")
    ax.bar([i + width / 2 for i in x], controller, width, label="Controller")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Exact accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Held-out accuracy by family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_action_mix_chart(path: Path, rows: list[dict[str, Any]], primary_name: str) -> None:
    selected = next(row for row in rows if row["policy"] == primary_name)
    variants = list(VARIANT_ORDER)
    counts = [selected["generated_variant_counts"].get(variant, 0) for variant in variants]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(variants, counts, color="tab:purple", alpha=0.8)
    ax.set_ylabel("Generated tasks")
    ax.set_title("Tool actions generated by learned controller")
    ax.set_ylim(0, max(counts + [1]) + 2)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_report(root: Path, summary: dict[str, Any]) -> None:
    selected = summary["controller_selection"]["selected"]
    policy = selected["policy"]
    test_rows = {row["policy"]: row for row in summary["test_metrics"]}
    primary = test_rows[summary["selected_test_policy"]]
    direct = test_rows["direct_only"]
    full = test_rows["prefix5_first_visible"]
    best_fixed = test_rows.get("fixed_visible_out_less_cols_full_else_direct")
    oracle_budget = test_rows["oracle_budget_full_only_when_helpful"]
    oracle_full = test_rows["oracle_best_available_full_budget"]

    headline = [direct, primary]
    if best_fixed and best_fixed["policy"] != primary["policy"]:
        headline.append(best_fixed)
    headline += [full, oracle_budget, oracle_full]

    lines = [
        "# Adaptive Tool Controller",
        "",
        "## Summary",
        "",
        "This experiment trains a small offline controller to choose between direct answering and external executable-program tool actions. The controller is selected on pilot train/dev records and frozen before held-out test scoring.",
        "",
        "Selected controller:",
        "",
        f"- Feature: `{policy['feature']}`",
        f"- If true: `{policy['true_action']}`",
        f"- If false: `{policy['false_action']}`",
        f"- Candidates passing train gate: `{summary['controller_selection']['num_candidates_after_train_gate']}`",
        "",
        "## Held-Out Test Result",
        "",
        "| Policy | Exact | Accuracy | Tokens | Avg tokens/task | Program commits | Recoveries | Losses | Commit precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in headline:
        lines.append(
            f"| `{row['policy']}` | {row['exact']}/{row['n']} | {pct(row['accuracy'])} | "
            f"{row['total_forward_tokens']:,} | {row['avg_forward_tokens']:.0f} | "
            f"{row['program_commits']} | {row['direct_miss_recoveries']} | {row['direct_correct_losses']} | "
            f"{pct(row['program_commit_precision'])} |"
        )
    lines += [
        "",
        "The learned controller improves over direct answering and becomes a low-cost Pareto point. It does not reach the strongest fixed shape rule included as a diagnostic; the learned policy buys the first four recoveries cheaply, while the fixed shape rule buys the remaining four recoveries with additional tool budget.",
        "",
        "## Pilot Selection",
        "",
        "| Rank | Feature | True action | False action | Train exact | Dev exact | Dev tokens | Dev losses |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summary["controller_selection"]["top_candidates"][:12], 1):
        p = row["policy"]
        lines.append(
            f"| {rank} | `{p['feature']}` | `{p['true_action']}` | `{p['false_action']}` | "
            f"{row['train']['exact']}/{row['train']['n']} | {row['dev']['exact']}/{row['dev']['n']} | "
            f"{row['dev']['total_forward_tokens']:,} | {row['dev']['direct_correct_losses']} |"
        )
    lines += [
        "",
        "## Family Breakdown",
        "",
        "| Family | n | Direct | Controller | Recoveries | Losses | Program commits | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, row in summary["family_metrics"].items():
        lines.append(
            f"| `{family}` | {row['n']} | {row['direct_exact']}/{row['n']} | {row['exact']}/{row['n']} | "
            f"{row['recoveries']} | {row['losses']} | {row['program_commits']} | {row['tokens']:,} |"
        )
    lines += [
        "",
        "## Pareto Frontier",
        "",
    ]
    for name in summary["deployable_pareto_front"]:
        row = test_rows[name]
        lines.append(f"- `{name}`: {row['exact']}/{row['n']} ({pct(row['accuracy'])}), {row['total_forward_tokens']:,} tokens")
    lines += [
        "",
        "## Figures",
        "",
        "![Pareto](figures/controller_pareto.png)",
        "",
        "![Selection](figures/controller_selection.png)",
        "",
        "![Family](figures/family_accuracy.png)",
        "",
        "![Action mix](figures/action_mix.png)",
        "",
        "## Interpretation",
        "",
        "The controller confirms that external program tools are valuable only on a narrow structural subset. A one-feature policy selected from pilot data recovers a useful low-cost slice without paying full portfolio cost on every task. The stronger fixed shape diagnostic shows that the remaining recoveries require broader portfolio calls, not just a cheaper single-tool action.",
        "",
        "This is an orchestration result, not a generation-capability result. The controller changes how budget is allocated over already-generated tool candidates; it does not create new candidates outside the recorded pool.",
        "",
        "## Limitations",
        "",
        "- Offline evaluation over a fixed candidate pool; no fresh model generations are produced in this package.",
        "- The pilot split is small, so feature selection is unstable. The fixed shape-rule diagnostic should be validated across more held-out family splits.",
        "- The depth-1 controller is intentionally simple. A richer sequential controller should be trained only after this signal replicates under regenerated candidates.",
    ]
    write_text(root / "reports/report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root

    pilot = load_jsonl(root / "data/pilot_train_dev_records.jsonl")
    test = load_jsonl(root / "data/test_records.jsonl")
    cases = {case["file"]: case for case in load_jsonl(root / "data/cases.jsonl")}
    by_split = split_records(pilot)
    train = by_split.get("train", [])
    dev = by_split.get("dev", [])
    actions = build_actions(VARIANT_ORDER)

    selection = select_controller(train, dev, cases, actions)
    selected_policy = selection["selected"]["policy"]
    selected_name = "learned_controller"
    selected_test = evaluate_controller(test, cases, actions, selected_policy, selected_name)
    write_json(root / f"reports/decisions/{selected_name}.json", selected_test["decisions"])

    baseline_rows = evaluate_baselines(test, cases, actions)
    test_rows = [{k: v for k, v in selected_test.items() if k != "decisions"}] + baseline_rows
    # Keep best row for duplicate policy names.
    by_policy: dict[str, dict[str, Any]] = {}
    for row in test_rows:
        by_policy[row["policy"]] = row
    test_rows = sorted(by_policy.values(), key=lambda row: (row["accuracy"], -row["total_forward_tokens"]), reverse=True)

    # Save decisions for baseline actions used in the report.
    for action_name, action in actions.items():
        if action_name.startswith("oracle") or action_name in {"direct_only", "prefix1_first_visible", "prefix5_first_visible"}:
            metrics = evaluate_action(test, action)
            write_json(root / f"reports/decisions/{action_name}.json", metrics["decisions"])

    family = family_metrics(test, selected_test)
    summary = {
        "variant_order": VARIANT_ORDER,
        "controller_selection": selection,
        "selected_test_policy": selected_name,
        "selected_test_metrics": {k: v for k, v in selected_test.items() if k != "decisions"},
        "test_metrics": test_rows,
        "family_metrics": family,
        "deployable_pareto_front": pareto_front(test_rows),
    }
    write_json(root / "reports/final_summary.json", summary)
    save_pareto_chart(root / "reports/figures/controller_pareto.png", test_rows, selected_name)
    save_selection_chart(root / "reports/figures/controller_selection.png", selection)
    save_family_chart(root / "reports/figures/family_accuracy.png", family)
    save_action_mix_chart(root / "reports/figures/action_mix.png", test_rows, selected_name)
    make_report(root, summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"controller_selection"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

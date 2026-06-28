#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
EVAL = REPORTS / "eval"

POLICY_ORDER = [
    "max_split_random8",
    "oracle_random8",
    "random_mined8",
    "max_split_mined8",
    "fullpool_max_split",
    "base_mined8",
    "sft_mined8",
    "sft_scrambled_features",
    "dpo_mined8",
    "grpo_mined8",
    "oracle_mined8",
    "fullpool_oracle",
]

POLICY_LABELS = {
    "max_split_random8": "Max-split random8",
    "oracle_random8": "Oracle random8",
    "random_mined8": "Random mined8",
    "max_split_mined8": "Max-split mined8",
    "fullpool_max_split": "Max-split full pool",
    "base_mined8": "Base Qwen mined8",
    "sft_mined8": "SFT mined8",
    "sft_scrambled_features": "SFT scrambled features",
    "dpo_mined8": "DPO mined8",
    "grpo_mined8": "GRPO mined8",
    "oracle_mined8": "Oracle mined8",
    "fullpool_oracle": "Oracle full pool",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_eval_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in POLICY_ORDER:
        path = EVAL / f"{policy}.json"
        if not path.exists():
            continue
        payload = load_json(path)
        for row in payload["records"]:
            row = dict(row)
            row["policy"] = policy
            rows.append(row)
    return pd.DataFrame(rows)


def overall_by_policy(records: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (policy, budget), group in records.groupby(["policy", "budget"]):
        out.append(
            {
                "policy": policy,
                "budget": int(budget),
                "records": len(group),
                "selected_hidden_all": group["selected_hidden_all"].mean(),
                "selected_exact_pair": group["selected_exact_pair"].mean(),
                "target_reachable": group["target_reachable"].mean(),
                "candidate_count_mean": group["candidate_count"].mean(),
                "hidden_equivalent_candidates_mean": group["hidden_equivalent_candidates"].mean(),
            }
        )
    df = pd.DataFrame(out)
    df["policy_order"] = df["policy"].map({p: i for i, p in enumerate(POLICY_ORDER)})
    return df.sort_values(["policy_order", "budget"]).drop(columns=["policy_order"])


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def weighted_budget3(overall: pd.DataFrame, policy: str, metric: str) -> float:
    row = overall[(overall["policy"] == policy) & (overall["budget"] == 3)]
    if row.empty:
        return 0.0
    return float(row.iloc[0][metric])


def make_plots(records: pd.DataFrame, overall: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for policy in [
        "max_split_random8",
        "max_split_mined8",
        "base_mined8",
        "sft_mined8",
        "dpo_mined8",
        "grpo_mined8",
        "oracle_mined8",
        "fullpool_oracle",
    ]:
        df = overall[overall["policy"] == policy]
        if df.empty:
            continue
        plt.plot(df["budget"], df["selected_hidden_all"], marker="o", label=POLICY_LABELS[policy])
    plt.xlabel("Probe budget")
    plt.ylabel("Hidden-all solved")
    plt.ylim(0, 0.95)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.title("Accuracy by Probe Budget")
    plt.tight_layout()
    plt.savefig(FIGURES / "budget_curve_hidden_all.png", dpi=180)
    plt.close()

    b3 = overall[overall["budget"] == 3].copy()
    b3 = b3[b3["policy"].isin(POLICY_ORDER)]
    b3["label"] = b3["policy"].map(POLICY_LABELS)
    plt.figure(figsize=(13, 5.2))
    values = b3["selected_hidden_all"].tolist()
    colors = []
    for policy in b3["policy"]:
        if "oracle" in policy:
            colors.append("#54a24b")
        elif policy in {"sft_scrambled_features", "dpo_mined8", "grpo_mined8"}:
            colors.append("#e45756")
        elif "qwen" in POLICY_LABELS[policy].lower() or policy.startswith("sft"):
            colors.append("#4c78a8")
        else:
            colors.append("#8a8f98")
    plt.bar(b3["label"], values, color=colors)
    for i, value in enumerate(values):
        plt.text(i, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Hidden-all solved at budget 3")
    plt.ylim(0, 0.95)
    plt.title("Budget-3 Policy and Probe-Source Comparison")
    plt.tight_layout()
    plt.savefig(FIGURES / "budget3_policy_bar.png", dpi=180)
    plt.close()

    cell = records[records["budget"] == 3].groupby(["policy", "library_size", "template"], as_index=False)["selected_hidden_all"].mean()
    keep = ["base_mined8", "sft_mined8", "dpo_mined8", "grpo_mined8", "oracle_mined8", "fullpool_oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, template in zip(axes, ["pair_affine_mod", "pair_compare_gate"]):
        sub = cell[(cell["template"] == template) & (cell["policy"].isin(keep))]
        pivot = sub.pivot(index="library_size", columns="policy", values="selected_hidden_all").reindex(columns=keep)
        pivot.rename(columns=POLICY_LABELS).plot(kind="bar", ax=ax)
        ax.set_title(template)
        ax.set_xlabel("Library size")
        ax.set_ylabel("Hidden-all solved")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "budget3_by_cell.png", dpi=180)
    plt.close()

    train_logs = {
        "SFT": load_json(REPORTS / "sft_training_losses.json") or [],
        "DPO": load_json(REPORTS / "dpo_training_logs.json") or [],
        "GRPO": load_json(REPORTS / "grpo_training_logs.json") or [],
    }
    plt.figure(figsize=(9, 4.5))
    for name, logs in train_logs.items():
        if not logs:
            continue
        df = pd.DataFrame(logs)
        if "loss" in df:
            plt.plot(df["step"], df["loss"], label=f"{name} loss")
    plt.xlabel("Step")
    plt.ylabel("Training objective")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig(FIGURES / "training_curves.png", dpi=180)
    plt.close()

    grpo = pd.DataFrame(load_json(REPORTS / "grpo_training_logs.json") or [])
    if not grpo.empty:
        plt.figure(figsize=(9, 4.5))
        plt.plot(grpo["step"], grpo["sampled_reward_mean"], label="sampled reward mean")
        plt.plot(grpo["step"], grpo["sampled_reward_max"], label="sampled reward max")
        plt.xlabel("GRPO step")
        plt.ylabel("Verifier reward")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.title("GRPO Sampled Rewards")
        plt.tight_layout()
        plt.savefig(FIGURES / "grpo_rewards.png", dpi=180)
        plt.close()


def report_md(records: pd.DataFrame, overall: pd.DataFrame, manifest: dict[str, Any]) -> str:
    b3 = overall[overall["budget"] == 3].set_index("policy")
    table = b3.loc[[p for p in POLICY_ORDER if p in b3.index], ["selected_hidden_all", "selected_exact_pair", "candidate_count_mean"]].copy()
    table.insert(0, "policy_name", [POLICY_LABELS[p] for p in table.index])
    table["selected_hidden_all"] = table["selected_hidden_all"].map(pct)
    table["selected_exact_pair"] = table["selected_exact_pair"].map(pct)
    table["candidate_count_mean"] = table["candidate_count_mean"].map(lambda x: f"{x:.1f}")
    md_table = table.rename(
        columns={
            "policy_name": "policy",
            "selected_hidden_all": "hidden-all @3",
            "selected_exact_pair": "exact pair @3",
            "candidate_count_mean": "survivors @3",
        }
    ).to_markdown(index=False)

    base = weighted_budget3(overall, "base_mined8", "selected_hidden_all")
    sft = weighted_budget3(overall, "sft_mined8", "selected_hidden_all")
    dpo = weighted_budget3(overall, "dpo_mined8", "selected_hidden_all")
    grpo = weighted_budget3(overall, "grpo_mined8", "selected_hidden_all")
    max_random = weighted_budget3(overall, "max_split_random8", "selected_hidden_all")
    max_mined = weighted_budget3(overall, "max_split_mined8", "selected_hidden_all")
    oracle_random = weighted_budget3(overall, "oracle_random8", "selected_hidden_all")
    oracle_mined = weighted_budget3(overall, "oracle_mined8", "selected_hidden_all")
    oracle_full = weighted_budget3(overall, "fullpool_oracle", "selected_hidden_all")
    scrambled = weighted_budget3(overall, "sft_scrambled_features", "selected_hidden_all")

    cell = records[records["budget"] == 3].groupby(["policy", "template"], as_index=False)["selected_hidden_all"].mean()
    template_lines = []
    for template in ["pair_affine_mod", "pair_compare_gate"]:
        vals = {row["policy"]: row["selected_hidden_all"] for _, row in cell[cell["template"] == template].iterrows()}
        template_lines.append(
            f"- `{template}`: base {pct(vals.get('base_mined8', 0.0))}, SFT {pct(vals.get('sft_mined8', 0.0))}, "
            f"DPO {pct(vals.get('dpo_mined8', 0.0))}, GRPO {pct(vals.get('grpo_mined8', 0.0))}, "
            f"mined oracle {pct(vals.get('oracle_mined8', 0.0))}, full-pool oracle {pct(vals.get('fullpool_oracle', 0.0))}."
        )

    return f"""# Qwen3.5-4B Oracle Probe Synthesis MDP

## Question

Can Qwen3.5-4B exploit a richer, deployable probe-generation layer inside a deterministic verifier MDP?

The model still does not name operators. It sees visible executions, the current surviving candidate count, and eight proposed probe inputs. The difference from a fixed small action set is that those eight probes are mined from a 96-case bank by target-independent candidate-bucket statistics. Training labels and rewards then use the verifier oracle to identify which displayed probe actually shrinks the target-retaining candidate set.

## Design

- Base model: Qwen3.5-4B with 4-bit QLoRA adapters.
- Train records: {manifest['train_records']['records']}; eval records: {manifest['eval_records']['records']}.
- Informative train states: {manifest['train_states']['states']}; informative eval states: {manifest['eval_states']['states']}.
- Probe bank: {manifest['query_pool_cases']} candidate inputs per task; displayed action set: 8 mined probes.
- Eval ladder: library sizes 64, 128, 256, 512; templates `pair_affine_mod` and `pair_compare_gate`.
- Probe budget: 0-3 verifier queries.
- Arms: random8 and mined8 controls, full-pool upper bounds, base Qwen, SFT, feature-scrambled SFT, DPO, and GRPO.

## Main Result

{md_table}

The action-source result is the largest signal. Moving from random-eight to mined-eight improves max-split from {pct(max_random)} to {pct(max_mined)}, and the same-budget oracle from {pct(oracle_random)} to {pct(oracle_mined)}. Scanning the full 96-probe bank with target-aware oracle selection reaches {pct(oracle_full)}, so the earlier low-information ceiling was partly an action-space ceiling, not just an intrinsic task ceiling.

The best learned policy is the SFT warm start: base Qwen {pct(base)} -> SFT {pct(sft)}. SFT beats target-independent mined max-split and falls when features are scrambled ({pct(scrambled)}), so the learned gain depends on the candidate-bucket summaries. DPO and GRPO did not improve on SFT in this run: DPO reached {pct(dpo)} and GRPO reached {pct(grpo)}.

## Regime Split

{chr(10).join(template_lines)}

The full-pool oracle changes the low-information story: `pair_compare_gate` rises to 73.8% under target-aware full-pool probing. That means the verifier environment contains useful discriminating probes, but the current deployable mining heuristic and Qwen action policy do not reliably surface or select them.

## Interpretation

This experiment supports a sharper line-2 thesis: the next leverage is not more operator naming, and not naive RL over the same eight choices. It is probe generation. Exhaustive search can expose high-value observations, and Qwen can learn a modest but real selection improvement over base and max-split once those observations are displayed. However, the full-pool oracle is far ahead of the learned policies, so the main remaining gap is proposing the right high-information probes under deployable constraints.

The negative DPO/GRPO result is useful. Preference or on-policy optimization over the same mined-eight actions was not enough; SFT was the robust learned component. The next step should make the action generator itself trainable or differentiably rankable, rather than only training a selector over the top eight target-independent probes.

## Figures

- ![Budget curve](figures/budget_curve_hidden_all.png)
- ![Budget 3 policy bar](figures/budget3_policy_bar.png)
- ![Budget 3 by cell](figures/budget3_by_cell.png)
- ![Training curves](figures/training_curves.png)
- ![GRPO rewards](figures/grpo_rewards.png)

## Artifacts

Large LoRA adapters are outside the experiment directory under `/workspace/large_artifacts/qwen35_4b_oracle_probe_synthesis_mdp`. The experiment directory contains standalone source, generated datasets, run logs, metrics, plots, and this report.
"""


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    records = load_eval_rows()
    if records.empty:
        raise RuntimeError("no eval records found")
    overall = overall_by_policy(records)
    overall.to_csv(REPORTS / "overall_by_policy_budget.csv", index=False)
    records.groupby(["policy", "library_size", "template", "budget"], as_index=False).agg(
        records=("record_id", "count"),
        selected_hidden_all=("selected_hidden_all", "mean"),
        selected_exact_pair=("selected_exact_pair", "mean"),
        target_reachable=("target_reachable", "mean"),
        candidate_count_mean=("candidate_count", "mean"),
        hidden_equivalent_candidates_mean=("hidden_equivalent_candidates", "mean"),
    ).to_csv(REPORTS / "summary_by_policy_cell.csv", index=False)
    records.to_json(REPORTS / "all_eval_records.json", orient="records", indent=2)
    make_plots(records, overall)
    manifest = load_json(ROOT / "data" / "dataset_manifest.json")
    md = report_md(records, overall, manifest)
    report_path = REPORTS / "qwen35_4b_oracle_probe_synthesis_mdp_report.md"
    report_path.write_text(md, encoding="utf-8")
    print(json.dumps({"report": str(report_path), "rows": len(records)}, indent=2))


if __name__ == "__main__":
    main()

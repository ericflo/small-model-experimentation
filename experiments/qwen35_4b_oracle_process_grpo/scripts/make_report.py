#!/usr/bin/env python
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
EVAL = REPORTS / "eval"


POLICY_ORDER = [
    "random",
    "base",
    "sft",
    "dpo",
    "dpo_shuffled",
    "dpo_scrambled_features",
    "grpo",
    "max_split",
    "oracle",
]
POLICY_LABELS = {
    "random": "Random",
    "base": "Base Qwen",
    "sft": "SFT",
    "dpo": "Process-DPO",
    "dpo_shuffled": "Shuffled-DPO",
    "dpo_scrambled_features": "DPO scrambled features",
    "grpo": "GRPO",
    "max_split": "Max-split",
    "oracle": "Oracle",
}


def load_eval_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for policy in POLICY_ORDER:
        path = EVAL / f"{policy}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload["records"])
        summaries.extend(payload["summary"])
    return pd.DataFrame(records), pd.DataFrame(summaries)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def overall_by_policy(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy, budget), group in records.groupby(["policy", "budget"]):
        rows.append(
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
    out = pd.DataFrame(rows)
    out["policy_order"] = out["policy"].map({p: i for i, p in enumerate(POLICY_ORDER)})
    return out.sort_values(["policy_order", "budget"]).drop(columns=["policy_order"])


def add_labels(ax: Any, values: list[float]) -> None:
    for i, value in enumerate(values):
        ax.text(i, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8, rotation=0)


def make_plots(records: pd.DataFrame, overall: pd.DataFrame, train_logs: dict[str, Any]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for policy in ["random", "base", "sft", "dpo", "dpo_shuffled", "grpo", "max_split", "oracle"]:
        df = overall[overall["policy"] == policy]
        plt.plot(df["budget"], df["selected_hidden_all"], marker="o", label=POLICY_LABELS[policy])
    plt.xlabel("Probe budget")
    plt.ylabel("Hidden-all solved")
    plt.ylim(0, 0.55)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.title("Deployable Accuracy by Probe Budget")
    plt.tight_layout()
    plt.savefig(FIGURES / "budget_curve_hidden_all.png", dpi=180)
    plt.close()

    b3 = overall[overall["budget"] == 3].copy()
    b3 = b3[b3["policy"].isin(["random", "base", "sft", "dpo", "dpo_shuffled", "dpo_scrambled_features", "grpo", "max_split", "oracle"])]
    b3["label"] = b3["policy"].map(POLICY_LABELS)
    plt.figure(figsize=(10, 4.8))
    values = b3["selected_hidden_all"].tolist()
    colors = ["#8a8f98" if p in {"random", "base"} else "#4c78a8" for p in b3["policy"]]
    colors = ["#e45756" if p in {"dpo_shuffled", "dpo_scrambled_features"} else c for p, c in zip(b3["policy"], colors)]
    colors = ["#54a24b" if p in {"oracle", "max_split"} else c for p, c in zip(b3["policy"], colors)]
    plt.bar(b3["label"], values, color=colors)
    add_labels(plt.gca(), values)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Hidden-all solved at budget 3")
    plt.ylim(0, 0.55)
    plt.title("Budget-3 Policy Comparison")
    plt.tight_layout()
    plt.savefig(FIGURES / "budget3_policy_bar.png", dpi=180)
    plt.close()

    cell = records[records["budget"] == 3].groupby(["policy", "library_size", "template"], as_index=False)["selected_hidden_all"].mean()
    keep = ["base", "sft", "dpo", "grpo", "max_split", "oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, template in zip(axes, ["pair_affine_mod", "pair_compare_gate"]):
        sub = cell[(cell["template"] == template) & (cell["policy"].isin(keep))]
        pivot = sub.pivot(index="library_size", columns="policy", values="selected_hidden_all").reindex(columns=keep)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(template)
        ax.set_xlabel("Library size")
        ax.set_ylabel("Hidden-all solved")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "budget3_by_cell.png", dpi=180)
    plt.close()

    if train_logs.get("sft"):
        plt.figure(figsize=(8, 4))
        sft = pd.DataFrame(train_logs["sft"])
        plt.plot(sft["step"], sft["loss"], label="SFT loss")
        if train_logs.get("dpo"):
            dpo = pd.DataFrame(train_logs["dpo"])
            plt.plot(dpo["step"], dpo["loss"], label="DPO loss")
        if train_logs.get("grpo"):
            grpo = pd.DataFrame(train_logs["grpo"])
            plt.plot(grpo["step"], grpo["loss"], label="GRPO loss")
        plt.xlabel("Step")
        plt.ylabel("Training objective")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.title("Training Curves")
        plt.tight_layout()
        plt.savefig(FIGURES / "training_curves.png", dpi=180)
        plt.close()

    train = pd.DataFrame(load_json(REPORTS / "grpo_training_logs.json") or [])
    if not train.empty:
        plt.figure(figsize=(8, 4))
        plt.plot(train["step"], train["sampled_reward_mean"], label="sampled reward mean")
        plt.plot(train["step"], train["sampled_reward_max"], label="sampled reward max")
        plt.xlabel("GRPO step")
        plt.ylabel("Verifier reward")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.title("GRPO Sampled Rewards")
        plt.tight_layout()
        plt.savefig(FIGURES / "grpo_rewards.png", dpi=180)
        plt.close()


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def report_md(records: pd.DataFrame, overall: pd.DataFrame, manifest: dict[str, Any]) -> str:
    b3 = overall[overall["budget"] == 3].set_index("policy")
    base = float(b3.loc["base", "selected_hidden_all"])
    oracle = float(b3.loc["oracle", "selected_hidden_all"])
    dpo = float(b3.loc["dpo", "selected_hidden_all"])
    sft = float(b3.loc["sft", "selected_hidden_all"])
    grpo = float(b3.loc["grpo", "selected_hidden_all"])
    shuffled = float(b3.loc["dpo_shuffled", "selected_hidden_all"])
    scrambled = float(b3.loc["dpo_scrambled_features", "selected_hidden_all"])
    max_split = float(b3.loc["max_split", "selected_hidden_all"])
    headroom = (dpo - base) / max(oracle - base, 1e-9)

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

    cell = records[records["budget"] == 3].groupby(["policy", "template"], as_index=False)["selected_hidden_all"].mean()
    template_lines = []
    for template in ["pair_affine_mod", "pair_compare_gate"]:
        vals = {row["policy"]: row["selected_hidden_all"] for _, row in cell[cell["template"] == template].iterrows()}
        template_lines.append(
            f"- `{template}`: base {pct(vals.get('base', 0.0))}, SFT {pct(vals.get('sft', 0.0))}, "
            f"DPO {pct(vals.get('dpo', 0.0))}, GRPO {pct(vals.get('grpo', 0.0))}, oracle {pct(vals.get('oracle', 0.0))}."
        )

    return f"""# Qwen3.5-4B Oracle Process GRPO Report

## Question

Can Qwen3.5-4B learn to make useful decisions inside a deterministic verifier MDP when the training oracle can score process actions exactly?

The model is not asked to name operators directly. It receives a compact process state containing visible executions, candidate-set size, and eight concrete probe choices with candidate-output bucket summaries. The action is one letter, A-H. The verifier then executes that probe, filters candidates, and repeats for up to three probes. Evaluation is deployable: learned policies do not see hidden labels or the target pair.

## Design

- Base model: Qwen3.5-4B, 4-bit QLoRA adapters.
- Train records: {manifest['train_records']['records']}; eval records: {manifest['eval_records']['records']}.
- Informative train states: {manifest['train_states']['states']}; informative eval states: {manifest['eval_states']['states']}.
- Eval ladder: library sizes 64, 128, 256, 512; two output regimes, `pair_affine_mod` and `pair_compare_gate`.
- Probe budget: 0-3 verifier queries.
- Optimizers: oracle-action SFT, process-DPO, shuffled-reward DPO control, and GRPO.

## Main Result

At budget 3, Qwen posttraining substantially improved the process controller over the base policy:

{md_table}

SFT recovered most of the available same-budget oracle headroom: base {pct(base)} -> SFT {pct(sft)}. Process-DPO added a small further gain to {pct(dpo)}, which is {headroom:.1%} of the base-to-oracle headroom. GRPO matched DPO at {pct(grpo)} but did not clearly exceed it in the short run.

The shuffled-reward and scrambled-feature controls are important. Shuffled-DPO fell to {pct(shuffled)}, and feature-scrambled DPO fell to {pct(scrambled)}. That means the verifier-aligned rewards and displayed candidate-bucket summaries both mattered.

## Regime Split

The learned policy helped mainly where the observations carry enough information:

{chr(10).join(template_lines)}

The low-information comparison regime remains bounded by identifiability. Even the same-budget oracle reaches only {pct(records[(records['policy']=='oracle') & (records['budget']==3) & (records['template']=='pair_compare_gate')]['selected_hidden_all'].mean())} there, because many candidate programs remain hidden-equivalent after three probes.

## Interpretation

This supports the process-control version of the neurosymbolic hypothesis: let exhaustive search and execution make answers reachable, then train Qwen to orchestrate verifier actions. The cleanest signal is not GRPO alone; it is the full stack:

- SFT learns the action interface and makes a large jump over base.
- Process-DPO uses the perfect per-step verifier oracle and gives a smaller additional improvement.
- GRPO is viable but did not beat DPO in this short run.
- Shuffled reward and scrambled feature controls collapse, so the improvement is not just formatting or letter bias.

The hard ceiling is also clear. The controller cannot solve states where three observations do not identify a hidden-correct candidate. That pushes the next step toward joint optimization of policy and observation design, with either larger query budgets or richer probe-generation actions.

## Figures

- ![Budget curve](figures/budget_curve_hidden_all.png)
- ![Budget 3 policy bar](figures/budget3_policy_bar.png)
- ![Budget 3 by cell](figures/budget3_by_cell.png)
- ![Training curves](figures/training_curves.png)
- ![GRPO rewards](figures/grpo_rewards.png)

## Artifacts

Large LoRA adapters are outside the experiment directory under `/workspace/large_artifacts/qwen35_4b_oracle_process_grpo`. This directory contains the standalone source, generated datasets, run logs, metrics, figures, and report.
"""


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    records, summaries = load_eval_rows()
    if records.empty:
        raise RuntimeError("no eval records found")
    overall = overall_by_policy(records)
    overall.to_csv(REPORTS / "overall_by_policy_budget.csv", index=False)
    summaries.to_csv(REPORTS / "summary_by_policy_cell.csv", index=False)
    records.to_json(REPORTS / "all_eval_records.json", orient="records", indent=2)

    train_logs = {
        "sft": load_json(REPORTS / "sft_training_losses.json") or [],
        "dpo": load_json(REPORTS / "dpo_training_logs.json") or [],
        "dpo_shuffled": load_json(REPORTS / "dpo_shuffled_training_logs.json") or [],
        "grpo": load_json(REPORTS / "grpo_training_logs.json") or [],
    }
    make_plots(records, overall, train_logs)
    manifest = load_json(ROOT / "data" / "dataset_manifest.json")
    md = report_md(records, overall, manifest)
    (REPORTS / "qwen35_4b_oracle_process_grpo_report.md").write_text(md, encoding="utf-8")
    print(json.dumps({"report": str(REPORTS / "qwen35_4b_oracle_process_grpo_report.md"), "rows": len(records)}, indent=2))


if __name__ == "__main__":
    main()


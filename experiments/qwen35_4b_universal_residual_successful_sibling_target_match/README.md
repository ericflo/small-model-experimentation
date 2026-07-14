# Residual-Skill Successful-Sibling Universal Curriculum

Test same-parent shortest-success distillation only on procedural skills with a live greedy residual, while exact-exposure replay and an unchanged all-skill gate protect saturated skills.

**Status:** in-progress · since 2026-07-14 · authenticated sibling collection is complete; model-free sibling selection is the only next stage

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: C54 shortest-success compression; terminal clean-oracle restarts; terminal balanced-sibling availability stop.

## Question

Can shortest verifier-correct trajectories from the same deployed parent improve the ten procedural skills with real greedy residuals, while active replay preserves count, route, and select without manufacturing failures for them?

## Hypothesis

A correct sampled sibling supplies a complete reasoning path already inside the parent’s support. Restricting this signal to actual residual skills avoids off-policy oracle language and avoids wasting treatment capacity on saturated skills. Exact-exposure replay and the unchanged all-skill local gate make retention part of the success criterion.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: authenticated explicit composite `replay_after_close`; runtime LoRA is forbidden.
- Inherited collection: immutable 624-task source and 227-failure inventory from `qwen35_4b_universal_successful_sibling_target_match`.
- Residual treatment skills: induct, execute, trace, verify, repair, optimize, abstain, state, order, and probe. These prospectively have at least four hard failures each.
- Retention skills: select, count, and route. They receive no manufactured failure rows; active replay and a fresh all-13-skill gate protect them.
- Collection: all 225 residual hard failures, same parent and pinned vLLM runner, natural thinking, `n=16`, seed 66,117, temperature/top-p/top-k `0.6/0.95/20`, 1,024-token cap.
- Selection: four tasks per residual skill; shortest naturally stopped, closed, canonical, exact-answer sibling within 768 thinking tokens. No oracle fallback or second sampling event.
- Planned active control: independent same-parent replay continuation matched exactly on forward tokens, loss-bearing targets, absolute loss mass, updates, and 200 aligned replay rows.
- Fresh local gate: seed 88,012, two tasks for every one of all 13 skills, with the unchanged strict candidate wins over both parent and replay.
- Hidden boundary: `benchmarks/` remains unread; conditional aggregate seed 78,142 stays sealed until local promotion.
- Claim boundary: independent higher-tier confirmation and matched-compute sample-more remain mandatory.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_residual_successful_sibling_target_match/scripts/run.py --smoke
```

Checkpointed stages:

```bash
.venv/bin/python -B experiments/qwen35_4b_universal_residual_successful_sibling_target_match/scripts/run.py --stage collect-siblings
.venv/bin/python -B experiments/qwen35_4b_universal_residual_successful_sibling_target_match/scripts/run.py --stage select-siblings
```

Every model event requires clean synchronized `main`, its prerequisite committed at `HEAD`, and both required workflows green. Every result gets its own checked, rebased, pushed checkpoint before the next stage.

## Results

Model-free inheritance and residual input construction are complete. The source, failure inventory, greedy receipt, and terminal stop receipt retain their published hashes. The derived input contains exactly 225 hard failures across the ten registered residual skills and no answer, reference thought, audit, or truth field. Input/manifest/design hashes are `dafeb012...1119` / `cee88012...c7e7` / `e1066596...93d7`.

The sole sibling event ran from synchronized published commit `fc5a333b` after Validate Repository `29373498273` and Publish Research Site `29373498296` passed. It produced 3,600/3,600 completions over 225 prompts, sampled 2,337,087 tokens at 739.2 tok/s, and took 3,210.1 wrapper seconds. Raw/metadata/log/receipt hashes are `688c4f7e...c332` / `56951f00...9cdf` / `d0b31be8...f29` / `c3a3a297...f614`. Recovery was unused and generation was not rerun. Sibling correctness grading and selection remain unopened; no training, local evaluation, or benchmark access has occurred.

## Interpretation

There is no capability result yet. The complete same-parent sample bank now permits the frozen model-free availability test, but no sibling has yet been graded or selected. Residual repair remains separated from all-skill retention.

## Knowledgebase Update

- Program evidence updated: pending the frozen successful-sibling availability result.
- Program backlog updated: collection is complete and model-free selection is next.
- Claim ledger updated: no; no capability result exists.

## Artifacts

- `data/inherited_*`: self-contained published lineage artifacts.
- `data/residual_sibling_input_seed66117.jsonl`: oracle-free model input.
- `data/residual_collection_manifest.json` and `data/design_receipt.json`: frozen provenance.
- `runs/sibling_collection/seed66117.*`: complete raw output, metadata, log, and authenticated receipt.
- `reports/preregistration.md` and `reports/design_review.md`: prospective contract and collection authorization.
- `reports/artifact_manifest.yaml`: external parent and conditional model-artifact plan.

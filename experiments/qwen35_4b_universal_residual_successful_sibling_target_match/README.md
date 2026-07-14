# Residual-Skill Successful-Sibling Universal Curriculum

Test same-parent shortest-success distillation only on procedural skills with a live greedy residual, while exact-exposure replay and an unchanged all-skill gate protect saturated skills.

**Status:** finished · 2026-07-14 · terminal successful-sibling availability stop; training, local evaluation, and benchmark access never opened

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

The sole sibling event ran from synchronized published commit `fc5a333b` after Validate Repository `29373498273` and Publish Research Site `29373498296` passed. It produced 3,600/3,600 completions over 225 prompts, sampled 2,337,087 tokens at 739.2 tok/s, and took 3,210.1 wrapper seconds. Raw/metadata/log/receipt hashes are `688c4f7e...c332` / `56951f00...9cdf` / `d0b31be8...f29` / `c3a3a297...f614`. Recovery was unused and generation was not rerun.

The frozen model-free selection ran from the published-green collection checkpoint `915a7c62`. Grading found 855/3,600 qualified siblings overall, but per-task availability was `induct=2`, below the mandatory four tasks per residual skill; every other skill met quota (execute 29, optimize/probe/repair 21, state/trace/verify 12, order 11, abstain 6). The outcome is `STOP_INSUFFICIENT_SUCCESSFUL_SIBLINGS`; inventory/selection-receipt hashes are `60c95b7a...083e` / `d3926daf...ad01`. No training corpus was emitted, training seed 50 and local seed 88,012 were not consumed, and benchmark aggregate seed 78,142 was never opened.

## Interpretation

This is an availability negative for policy-supported successful-sibling distillation, concentrated exactly at the program's known wall. Across 46 induct hard failures and 736 samples, the parent produced a qualified short correct sibling on only two tasks, while nine of ten residual skills supplied quota easily. The signal a same-parent curriculum needs is missing precisely where repair matters most: induction failures are not near-misses the parent can re-sample its way out of (C38/C39). Lowering the quota, dropping induct, raising `n`, or relaxing the 768-token ceiling would each abandon the preregistered design rather than test it. The residual-vs-retention separation itself worked: the stop is now attributable to one skill's policy support, not to saturated skills polluting the quota.

## Knowledgebase Update

- Program evidence updated: terminal availability result recorded.
- Program backlog updated: the successful-sibling line is closed; a designed-curriculum successor is queued.
- Claim ledger updated: no; no capability result exists.

## Artifacts

- `data/inherited_*`: self-contained published lineage artifacts.
- `data/residual_sibling_input_seed66117.jsonl`: oracle-free model input.
- `data/residual_collection_manifest.json` and `data/design_receipt.json`: frozen provenance.
- `runs/sibling_collection/seed66117.*`: complete raw output, metadata, log, and authenticated receipt.
- `data/successful_sibling_inventory_seed66117.json` and `data/successful_sibling_selection_receipt.json`: terminal model-free grading and stop receipt.
- `reports/preregistration.md` and `reports/design_review.md`: prospective contract and collection authorization.
- `reports/artifact_manifest.yaml`: external parent and conditional model-artifact plan.

## Terminal Disposition

No later event is authorized here. Do not lower the four-task quota, drop induct from the treatment set, resample deficient skills at larger `n`, relax the 768-thinking-token ceiling, or add oracle rows. The published 3,600-output bank is immutable and may be reused only by a new experiment with its own intake, prospective policy, receipts, and lifecycle. Together with the balanced predecessor, this closes same-parent successful-sibling mining as a curriculum source: the skills that keep failing greedily are the skills whose successes the parent cannot sample.

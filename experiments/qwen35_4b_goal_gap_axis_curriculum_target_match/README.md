# Goal-Gap Axis Curriculum

Attack the benchmark's four empirically stuck families directly: designed, contamination-free, single-turn atom curricula built only from public axis descriptions — multi-formalism program repair, budgeted unique-shortest route search, instruction hygiene with parseable-wrong decoys, and branch-balanced protocol execution — trained on the surface-general `designed_fresh` parent against a three-axis exact-exposure replay control, with a prospectively achievable two-instrument gate.

**Status:** in-progress · since 2026-07-14 · model-free construction under way; no model event has run

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: goal-gap forensics (the all-families goal passed once in 65 quick events; failures concentrate on four families whose gym analogues near-mastered internally without transferring); the fresh-surface trial's positive surface-generality reading and its structurally unpassable induct floor.

## Question

Does axis-targeted designed-atom content — varied formalisms within each stuck axis — install held-out-surface capability on those axes without regressing the thirteen retained skills, where single-surface harvested episodes did not transfer?

## Hypothesis

Coverage-without-transfer is a surface-binding failure: each gym axis lived on one surface, so the installed skill bound to the skin. Designed atoms that vary formalism and vocabulary within each axis should bind to structure (as the fresh-surface trial proved for the generic dose), and the hygiene block's decoy construction makes injection-compliance a parseable wrong answer, scoring resistance directly. Exact three-axis exposure matching against replay makes any win attributable to content.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: authenticated `designed_fresh` composite (tree `93433aa2...0255`, weights `0a3b89cd...7979`); warm start from its adapter (`36f41095...b442`). Runtime LoRA forbidden.
- Treatment corpus (construction seed 77,117): `sft_axis160.jsonl` — 40 rows each of `u_tracefix` (four invented executable formalisms, unique repair enforced by exhaustive enumeration), `u_explore` (unique shortest route under an exact move budget, compact frontier-notation thinking), `u_hygiene` (embedded directives carry format-matched decoys; 30 injected / 10 clean), `u_protocol` (documented flags plus tally; all three closing branches trained). Banned-vocabulary audit covers benchmark family names, gym families and their flavor nouns, and every predecessor surface pool and attribute set.
- Streams: 1,280 shared position-aligned replay rows + one 240-row variable block per arm (candidate = 160 treatment + 80 fillers; control = 240 replay), EXACT equality on forward tokens, loss-bearing targets, and absolute loss mass (MILP, namespace seed 55,118); 1,520 rows per arm.
- Training: one event per arm, 190 updates, LR 1e-5, rank 32 alpha 64, think/close weights 0.2/0.2, seed 52, zero skips required.
- Local gate (one event, fresh seed 88,014, three merged composites): two instruments — a 40-task axis holdout (10 per axis, unseen seed) and a 104-task retention screen (8 per original skill). Promotion: candidate strictly beats parent AND replay on axis-holdout total and on at least 3 of 4 axis kinds; retention within non-inferiority bands (correct ≥ each control − 5; caps ≤ each control + 3; parsed ≥ each control − 3); route abstentions ≤ 4. No absolute per-kind floors.
- Conditional aggregate: sealed fresh seed 78,144, quick tier, think budget 1,024, four composites (base, parent, replay control, candidate). Gates: candidate aggregate strictly above base, replay, and parent; the every-family-strict-versus-base record is reported as the goal gate with the frozen power statement (a quick-tier pilot failure of that gate is expected even under the hypothesis; family-level confirmation belongs to the medium tier).
- Hidden boundary: `benchmarks/` unread; higher-tier confirmation and matched-compute sample-more remain mandatory before any universal claim.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_goal_gap_axis_curriculum_target_match/scripts/run.py --smoke
```

Checkpointed stages (each requires its prerequisite committed at a clean, pushed, green `main`):

```bash
.venv/bin/python -B experiments/qwen35_4b_goal_gap_axis_curriculum_target_match/scripts/run.py --stage train-control
.venv/bin/python -B experiments/qwen35_4b_goal_gap_axis_curriculum_target_match/scripts/run.py --stage train-candidate
.venv/bin/python -B experiments/qwen35_4b_goal_gap_axis_curriculum_target_match/scripts/run.py --stage merge-arms
.venv/bin/python -B experiments/qwen35_4b_goal_gap_axis_curriculum_target_match/scripts/run.py --stage local
.venv/bin/python -B experiments/qwen35_4b_goal_gap_axis_curriculum_target_match/scripts/run.py --stage benchmark
```

## Results

No model event has run. Model-free construction is in progress.

## Interpretation

None yet.

## Knowledgebase Update

- Program evidence updated: pending the first result.
- Program backlog updated: this trial claims the queued goal-gap successor slot.
- Claim ledger updated: no; no capability result exists.

## Artifacts

- `data/sft_axis160.jsonl`, `data/corpus_manifest.json`: frozen axis corpus.
- `data/sft_blend.jsonl`: frozen replay pool (byte-identical to predecessors).
- `data/stream_manifest.json`, `data/stream_token_receipt.json`: exposure receipts.
- `data/local_tasks_seed88014.jsonl`, `data/local_input_seed88014.jsonl`, `data/local_design_receipt.json`: frozen two-instrument gate.
- `reports/preregistration.md`, `reports/design_review.md`: prospective contract and adversarial authorization.
- `reports/artifact_manifest.yaml`: external parent and conditional model-artifact plan.

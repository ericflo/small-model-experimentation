# Goal-Gap Axis Curriculum

Attack the benchmark's four empirically stuck families directly: designed, contamination-free, single-turn atom curricula built only from public axis descriptions — multi-formalism program repair, budgeted unique-shortest route search, instruction hygiene with parseable-wrong decoys, and branch-balanced protocol execution — trained on the surface-general `designed_fresh` parent against a three-axis exact-exposure replay control, with a prospectively achievable two-instrument gate.

**Status:** finished · 2026-07-15 · local gate PROMOTED (first in the line); aggregate pilot negative — candidate beat base +0.314 with 7 positive and 0 negative families but lost the aggregate to parent and replay; seed 78,144 consumed

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

Both arms trained cleanly (replay 0.3776, axis 0.4884 train loss; 0 skips each) and merged; the frozen 144-task gate event PROMOTED `axis_curriculum` — the first promotion in this program's universal line: axis holdout 28/40 versus parent 22 and replay 18 (kind wins on hygiene 9-vs-5/5, explore 7-vs-6/3, tracefix 4-vs-3/2; protocol tied at the preregistered control ceiling), with retention byte-equal to the parent (71/95/9) while replay drifted (65/89/15).

The conditional aggregate pilot at seed 78,144 (quick, think budget 1,024) then ran all four composites: base 0.1085, axis_curriculum 0.4223, parent 0.4644, replay_repeat 0.5081. The candidate beat base by +0.3138 with SEVEN strictly positive families, three ties, and zero negatives — including flipping warren (0 → 0.125), one of the historically stuck families — but lost the aggregate to both its parent (−0.042) and the replay control (−0.086), so the pilot gate fails. Notably the replay control flipped rites (0 → 0.125) and posted the highest aggregate this line has recorded at any seed, and menders (0 for every arm) and sirens (exactly 0.500 for every arm) did not move for anyone.

## Interpretation

Three separable findings. (1) The axis-atom mechanism installs: first-ever local promotion, +6/+10 on held-out axis tasks with zero retention cost. (2) Local axis wins under-convert at quick tier: hygiene's near-doubling locally left sirens pinned at 0.500, and tracefix's win left menders at zero — though warren did flip, consistent with the explore lesson (or 1/8-granularity noise; single seed). (3) Replay continuation compounds again: one more replay round beat everything on aggregate while flipping rites, replicating the line's replay-is-active-intervention law. Per the frozen drift-versus-content reading, the candidate's aggregate loss to replay is a content-opportunity-cost result at matched exposure, not retention damage (retention was byte-equal to the parent). The goal's wall is now precisely two families wide: menders and sirens are frozen for every arm at every seed at this tier configuration.

## Terminal Disposition

No later event is authorized here. Seed 78,144 is consumed and recorded; do not re-run any stage or benchmark the composites from this directory. The published `replay_repeat` composite (aggregate 0.5081, receipts under `runs/`) is the strongest known artifact and the presumptive parent for successors; `axis_curriculum` remains the only artifact with a positive axis-installability reading.

## Knowledgebase Update

- Program evidence updated: local promotion, pilot negative, per-family record, and the replay-compounding observation recorded.
- Program backlog updated: the menders/sirens wall and the replay-compounding line are queued with calibration notes.
- Claim ledger updated: no; single-seed pilot events do not mint claims.

## Artifacts

- `data/sft_axis160.jsonl`, `data/corpus_manifest.json`: frozen axis corpus.
- `data/sft_blend.jsonl`: frozen replay pool (byte-identical to predecessors).
- `data/stream_manifest.json`, `data/stream_token_receipt.json`: exposure receipts.
- `data/local_tasks_seed88014.jsonl`, `data/local_input_seed88014.jsonl`, `data/local_design_receipt.json`: frozen two-instrument gate.
- `reports/preregistration.md`, `reports/design_review.md`: prospective contract and adversarial authorization.
- `reports/artifact_manifest.yaml`: external parent and conditional model-artifact plan.

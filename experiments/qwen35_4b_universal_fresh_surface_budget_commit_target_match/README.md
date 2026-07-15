# Fresh-Surface Budget-Commit Universal Curriculum

Re-dose the proven 160-row designed distribution on six disjoint fresh surfaces from the authenticated parent, and ablate 40 of those rows into a bounded-check budget-commit lesson, under a three-axis exact-exposure replay control and a quadrupled 104-task local gate.

**Status:** in-progress · since 2026-07-14 · model-free construction under way; no model event has run

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: mid-density near-miss (19/26 with parse and caps each one item short); close-weight redistribution negative; terminal successful-sibling availability stops closing harvest-based supervision.

## Question

Is the 160-row designed dose surface-general — does it install when rendered on six entirely fresh surfaces from the current parent — and does substituting 40 rows with a designed bounded-computation budget-commit lesson repair the termination seam (cap contacts, unparsed answers) at exactly matched exposure?

## Hypothesis

The designed dose binds to structure, not surface tokens, so a fresh-surface re-dose from the replay-refreshed parent should reproduce the mid-density accuracy gain on a gate that renders only original surfaces (a built-in transfer test). The remaining deployment seam is termination; a lesson whose content is monitored bounded computation with a mandatory canonical commit — never a loss-weight knob and never an idealized long trace — should reduce cap contacts and unparsed answers without losing semantics. Exact three-axis exposure matching (forward tokens, loss-bearing targets, absolute loss mass) against an active replay continuation makes any win attributable to content.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: authenticated explicit composite `replay_after_close` (weights `7ab4c419...e2e6`); runtime LoRA is forbidden.
- Treatment corpora (construction seed 77,116): `designed_fresh` = 160 rows at the frozen designed160 per-skill quotas on six fresh surfaces (greek, elements, animals, ordinals, gems, digraphs) with fresh separators, attributes, and routing capabilities; `budget_commit` = a deterministic 120-row subset of the same corpus plus 40 budget lessons (hard check allowance, stop-on-first-hit, mandatory `BUDGET` commit on exhaustion, decoy planted immediately past the cutoff). A banned-vocabulary audit proves zero leakage of predecessor surfaces, gym family names, and public benchmark family names.
- Streams: 1,280 shared position-aligned replay rows plus a 240-row variable block per arm; the two treatment blocks (160 treatment + 80 replay filler) and the 240-row replay control block are matched EXACTLY on forward tokens, nonzero loss-bearing targets, and absolute loss mass (MILP, namespace seed 55,117).
- Training: one event per arm, 1,520 rows, 190 optimizer steps, batch 1, accumulation 8, LR 1e-5, rank 32, alpha 64, think/close weights 0.2/0.2, seed 51, warm start from the parent adapter (`bb59d3bd...154d`), zero skipped rows required.
- Local gate: fresh seed 88,013, 104 tasks (8 per each of 13 skills) from the ORIGINAL-surface generator, explicit merged composites on the pinned vLLM geometry, greedy, natural thinking, 1,024-token cap. Bars per candidate: parsed ≥ 96, correct ≥ 68, cap contacts ≤ 8, route abstentions ≤ 4, execute/induct/probe ≥ 4/8 each, plus strict wins over parent AND replay on total correct and the 24-row execute+induct+probe subtotal. Single-winner promotion: higher total, then target subtotal, then fewer caps, then `budget_commit`.
- Conditional aggregate: sealed fresh seed 78,143, quick tier, think budget 1,024, same-backend paired event over base, parent, replay control, and the promoted candidate. Promotion requires the candidate to strictly lift the aggregate AND every one of the ten public families versus base, and to strictly beat both the replay control and the parent on aggregate.
- Hidden boundary: `benchmarks/` remains unread; independent higher-tier confirmation and matched-compute sample-more remain mandatory before any universal claim.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_fresh_surface_budget_commit_target_match/scripts/run.py --smoke
```

Checkpointed stages (each requires its prerequisite committed at a clean, pushed, green `main`):

```bash
.venv/bin/python -B experiments/qwen35_4b_universal_fresh_surface_budget_commit_target_match/scripts/run.py --stage train-control
.venv/bin/python -B experiments/qwen35_4b_universal_fresh_surface_budget_commit_target_match/scripts/run.py --stage train-designed
.venv/bin/python -B experiments/qwen35_4b_universal_fresh_surface_budget_commit_target_match/scripts/run.py --stage train-budget
.venv/bin/python -B experiments/qwen35_4b_universal_fresh_surface_budget_commit_target_match/scripts/run.py --stage merge-arms
.venv/bin/python -B experiments/qwen35_4b_universal_fresh_surface_budget_commit_target_match/scripts/run.py --stage local
.venv/bin/python -B experiments/qwen35_4b_universal_fresh_surface_budget_commit_target_match/scripts/run.py --stage benchmark
```

## Results

No model event has run. Model-free construction (corpora, exposure match, local gate design) is in progress.

## Interpretation

None yet.

## Knowledgebase Update

- Program evidence updated: pending the first result.
- Program backlog updated: this trial claims the queued bounded-computation successor slot.
- Claim ledger updated: no; no capability result exists.

## Artifacts

- `data/sft_fresh_designed160.jsonl`, `data/sft_fresh_budget160.jsonl`, `data/corpus_manifest.json`: frozen fresh-surface corpora.
- `data/sft_blend.jsonl`: frozen replay pool (byte-identical to predecessors).
- `data/stream_manifest.json`, `data/stream_token_receipt.json`: three-axis exposure receipts.
- `data/local_tasks_seed88013.jsonl`, `data/local_input_seed88013.jsonl`, `data/local_design_receipt.json`: frozen local gate.
- `reports/preregistration.md`, `reports/design_review.md`: prospective contract and adversarial authorization.
- `reports/artifact_manifest.yaml`: external parent and conditional model-artifact plan.

# Policy-Supported Successful-Sibling Universal Curriculum

Test whether the deployed parent’s own short verifier-correct sampled trajectories can turn fresh greedy failures into transferable supervision under exact-exposure replay.

**Status:** in-progress · since 2026-07-14 · greedy collection is complete; model-free failure freezing is the next gated stage

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic training data install a general feature that improves the held-out aggregate without regressing any reported family?
- Prior anchors: C54's shortest-success compression signal; the terminal on-policy-prefix negative; the terminal clean-restart negative.

## Question

Does the prior local failure come from teaching hand-authored trajectories outside the deployed parent’s policy support? This trial keeps the fresh greedy-failure selector but replaces every oracle restart with a short successful trajectory sampled from the same authenticated parent.

## Hypothesis

On greedy-failure tasks, a verifier-correct sibling proves that the complete reasoning and answer path already lies inside the parent’s sampling support. Distilling the shortest such path from the original prompt should bank a reachable decision policy rather than merely teach answer emission, and should therefore beat an independently trained exact-exposure replay continuation.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, pinned at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: authenticated explicit composite `replay_after_close` from `qwen35_4b_universal_on_policy_prefix_repair_token_match`; runtime LoRA is forbidden.
- Task source: 624 fresh executable procedural tasks, 48 per each of 13 universal skills, construction seed 77,115.
- Collection: one natural-thinking greedy event at seed 66,115; then, only for its published hard failures, one natural-thinking `n=16` event at seed 66,116 with temperature/top-p/top-k `0.6/0.95/20`.
- Candidate source: four tasks per skill whose sampled sibling naturally stops, closes thinking, exactly matches executable truth, uses a canonical answer tail, and stays within 768 thinking tokens. The shortest qualified sibling wins. There is no oracle-trace fallback.
- Baseline: unchanged authenticated parent.
- Active control: independent same-parent replay continuation, later matched exactly on forward tokens, loss-bearing target tokens, absolute loss mass, updates, and aligned replay rows.
- Fresh local gate: unchanged 26-task, two-per-skill gate at seed 88,011; strict total and execute/induct/probe wins over both parent and replay are mandatory.
- Hidden-label boundary: `benchmarks/` remains unread. Aggregate seed 78,141 stays sealed until local promotion.
- Final claim boundary: even a strict all-family aggregate pilot still owes independent higher-tier confirmation and a matched-compute sample-more baseline.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_successful_sibling_target_match/scripts/run.py --smoke
```

Checkpointed stages:

```bash
.venv/bin/python -B experiments/qwen35_4b_universal_successful_sibling_target_match/scripts/run.py --stage collect-greedy
.venv/bin/python -B experiments/qwen35_4b_universal_successful_sibling_target_match/scripts/run.py --stage prepare-siblings
.venv/bin/python -B experiments/qwen35_4b_universal_successful_sibling_target_match/scripts/run.py --stage collect-siblings
.venv/bin/python -B experiments/qwen35_4b_universal_successful_sibling_target_match/scripts/run.py --stage select-siblings
```

Every stage requires its prerequisite receipt committed on clean, synchronized `main`. Each result is checked, rebased, pushed, and held until both repository workflows are green before the next event.

## Results

Model-free construction and the sole greedy event are complete. The 624 source rows and oracle-free input regenerate byte-for-byte, contain 48 rows for every skill, and have zero message overlap with the two closest predecessors or reserved local seeds 88,000–88,011. From pushed-green commit `0038fba1`, the authenticated parent produced 624/624 rows and 296,259 sampled tokens at 859.6 tok/s in 392.0 wrapper seconds. Raw/metadata/log/receipt hashes are `e91313c0...f556` / `0e82ae73...15ce` / `f1657151...ca4` / `cee1f19d...4962`; no recovery or rerun occurred. Failure grading, sibling sampling, training, local evaluation, and benchmark access remain unopened.

## Interpretation

No capability inference exists yet. The current evidence is design provenance plus an authenticated ungraded greedy collection. The sibling pool, selection rule, availability stop, and oracle-fallback prohibition were fixed before this parent output and cannot now change.

## Knowledgebase Update

- Program evidence updated: operational collection evidence is recorded in the backlog; capability evidence remains pending.
- Program backlog updated: this trial claims the queued successful-sibling successor.
- Claim ledger updated: no; no capability result exists.

## Artifacts

- `data/collection_tasks_seed77115.jsonl`: executable truth, never passed whole to the model.
- `data/greedy_input_seed66115.jsonl`: oracle-free model input.
- `data/collection_task_manifest.json` and `data/design_receipt.json`: frozen provenance.
- `reports/preregistration.md` and `reports/design_review.md`: prospective contract and adversarial authorization.
- `reports/artifact_manifest.yaml`: external parent and future adapter/composite plan.

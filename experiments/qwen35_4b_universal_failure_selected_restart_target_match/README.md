# Failure-Selected Counterfactual Restart Curriculum

**Status:** in-progress · since 2026-07-14 · paired training and fresh-local design complete; replay-control merge is next

This experiment tests whether selecting the stronger parent's fresh procedural
failures and teaching clean verified restarts can beat exactly exposure-matched replay
without conditioning on the parent's failed trajectory.

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can one Qwen3.5-4B checkpoint acquire execution, induction,
  verification, repair, state, and commitment behavior without trading away any
  held-out benchmark family?
- Prior anchors: C50's emission-seam installation result, C58's partial success from
  context-removal recomputation, and the terminal negative
  `qwen35_4b_universal_on_policy_prefix_repair_token_match`.

## Question

Does selecting tasks the deployed parent fails, then supervising a concise verified
solution from the original prompt, install broader competence than either continued
replay or conditioning the correction on the parent's long wrong prefix?

## Hypothesis

The predecessor solved on-policy data availability but put 47,123 wrong parent tokens
in the candidate's conditioning context and gave it 33,421 fewer supervised target
tokens than replay. A counterfactual restart moves the intervention before the error:
selection remains on-policy at the task level, while every trainable row begins at the
original prompt and contains a full truth-audited solution and answer. Exact matching
of forward tokens, loss-bearing target tokens, and absolute loss mass will isolate
that mechanism from extra compute or extra supervision.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: the published explicit `replay_after_close` composite, weight SHA-256
  `7ab4c419...36e2e`; runtime LoRA is forbidden.
- Collection substrate: 624 fresh procedural tasks, 48 per each of the 13 universal
  skills, construction seed 77,114. Truth source SHA-256 is `81edc9ea...de304`;
  oracle-free runner input SHA-256 is `25382689...0f5b`.
- Collection: one vLLM event, natural thinking, greedy `n=1`, seed 66,114,
  1,024-token cap, 4,096 context, and identical pinned runner geometry.
- Failure rule: cap contact, missing answer, wrong answer, or more than 128 thinking
  tokens. Hard correctness/cap failures rank before budget-only failures.
- Selection: exactly four failures per skill, 52 total, deterministic seed 55,114.
  An undersupplied skill ends the experiment before training.
- Frozen control: same-parent replay continuation. Both arms have 320 rows, 200
  byte-identical aligned replay rows, 40 effective-batch-eight updates, 297,731
  forward tokens, 126,796 loss-bearing targets, and absolute loss mass 27,632.8.
- Training warm start: the published replay adapter, weights/config SHA-256
  `bb59d3bd...5154d` / `0dfd9bda...120f`. Each arm continues independently from
  that same adapter.
- Train seed: 48. Fresh local seed: 88,010. Conditional aggregate seed: 78,140.
- Hidden-label boundary: no benchmark item, transcript, source, or detailed result is
  read. Aggregate access remains sealed until strict local promotion.

## Run

The CPU smoke path reauthenticates task construction, collection, selection, source
tokenization, materialized streams, independent token validation, and the second
adversarial review:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/qwen35_4b_universal_failure_selected_restart_target_match/scripts/run.py --smoke

```

Both paired training events have completed. The smoke path reauthenticates their
tracked receipts/logs and external adapters plus the separately frozen fresh-local
design. After that design is published and both repository workflows are green, the
only authorized next event is:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/qwen35_4b_universal_failure_selected_restart_target_match/scripts/run.py \
  --stage merge-control

```

Candidate merge requires the published control-merge receipt. Local generation
requires both published current-arm merge receipts and runs with `--stage local`.

## Results

The one preregistered parent event completed from pushed-green commit `1744e753`:
624/624 completions, 304,013 sampled tokens, 879.9 tok/s, and 394.96 seconds of
wrapper wall time. Rollout/metadata/log/receipt SHA-256 values are
`4bf15134...1099f`, `b43b3a0...1206d`, `668e9b70...369ff`, and
`1d35c63a...2b381`. The receipt records a clean `main` preflight, the authenticated
merged replay parent, no recovery or generation rerun, `benchmark_data_read=false`,
and a sealed aggregate seed.

This is collection evidence, not a capability result. Failure composition and quota
availability are now frozen by the separately checkpointed mining stage:

- 602/624 rows were eligible and 228 were hard correctness/cap failures.
- Every skill cleared the four-row quota; availability ranged from 40 to 48.
- The selected 52 rows are exactly four per skill: 40 hard failures and 12 correct but
  over-budget cases. Hard-failure availability was below four for abstain, count,
  route, and select, so their remaining slots prospectively used budget-only rows.
- Selected reasons total 29 cap contacts, 26 missing answers, 13 wrong answers, and
  51 over-budget flags; reasons overlap by row.
- Inventory/restart/selection/summary hashes are `c19d3de7...66240`,
  `022b1ea4...d951f`, `567d6b02...b662`, and `2e8a2192...e28ddf`.
- All 52 rows restart from the original prompt and zero contain a parent prefix. At
  this selection checkpoint training was unauthorized; benchmark/aggregate gates
  remain sealed.

This is still construction evidence, not a capability result.

Exact exposure is feasible without modifying any target, duplicating any row, or
truncating any sequence. A deterministic integral solver froze disjoint 68-row
candidate-filler and 120-row control blocks around the inherited 200-row shared core.
Independent encoding of the final files confirmed:

- 320/320 encoded rows and zero skips in each arm;
- exact equality at 297,731 forward tokens, 126,796 nonzero target tokens, and
  absolute loss mass 27,632.8;
- exactly 200 byte-identical rows at the same stream positions;
- zero parent-prefix tokens and four clean restarts per each of 13 skills;
- source-token/manifest/control/candidate/final-receipt hashes
  `ac9b9c8a...0bd6`, `7ba55045...91de1`, `7a8d4566...b5078`,
  `28deb20e...3190`, and `52a761ef...170`.

The candidate has 16,414 more total thinking-span tokens and 16,414 fewer masked
context tokens because of differing zero-weight forced-close composition, but equal
answer tokens, close tokens, actual loss-bearing tokens, and weighted loss mass. The
second review records this residual sequence-composition difference and authorizes
only replay-control training after publication. This remains construction evidence,
not a capability result.

From pushed-green exact-exposure commit `821d50d4`, the replay control independently
continued the authenticated parent for exactly 40/40 steps. It encoded 320/320 rows
with zero skips, completed in 297.3 trainer seconds (318.70 wrapper seconds), and
reported final train loss 0.3873. Receipt/log/adapter-config/adapter-weight hashes are
`3a9cc1ea...6d49`, `3bedc341...f25`, `dce1095c...f8f6`, and
`5840757d...b1c`; the adapter is 169,903,320 bytes. The preflight binds clean pushed
`main` at `821d50d4`, the frozen stream, and the original parent adapter. This is an
authenticated training event, not capability evidence.

From the separately pushed-green control checkpoint `2c78e655`, the candidate then
continued the same original parent for exactly 40/40 steps. It encoded 320/320 rows
with zero skips, completed in 298.5 trainer seconds (315.33 wrapper seconds), and
reported final train loss 0.5838. Receipt/log/adapter-config/adapter-weight hashes are
`6aa5c3f1...9871`, `c8572c88...202a`, `6915787d...7f50`, and
`2072c5c8...39bc`; the adapter is 169,903,320 bytes. Its receipt embeds and
reauthenticates the published control prerequisite while proving the candidate warm
start remained the original parent. Training loss is not a capability comparison.
No merge or evaluation has run.

The separately reviewed fresh-local design now freezes 26 new procedural items at
seed 88,010, exactly two per universal skill, and compares the unchanged replay
parent, matched-exposure replay continuation, and counterfactual-restart candidate.
Source/input/design-receipt SHA-256 values are `7b69473b...975f`,
`6efefc92...15e2`, and `124bbf99...2db5`.
Every arm uses the same explicit-composite vLLM runner and geometry. Promotion
requires the candidate's absolute 17/26 floor plus strict wins over both controls on
total correct and the six execute/induct/probe items. Complete seven-file composite
tree manifests, pre/post-arm model and Git authentication, strict 78-row receipt
shape, and durable failure receipts close deployment and transaction ambiguity.
Aggregate seed 78,140 remains sealed. This is frozen design evidence, not a
capability result.

## Interpretation

This trial distinguishes two ideas the predecessor confounded: selecting data from
the parent's actual failures, and conditioning training on the parent's failed
trajectory. A positive result would support task-level failure selection plus clean
recomputation, not long-prefix repair. A negative result would reject this balanced
13-skill restart package at the matched exposure, not all on-policy learning.

## Knowledgebase Update

- Program backlog: register as the active result-separated universal successor.
- Program evidence: update after the first preserved model event.
- Claim ledger: unchanged until held-out evidence warrants a claim.

## Artifacts

- `idea_intake.md` — closest duplicate and mechanism decision.
- `reports/preregistration.md` — frozen identities, gates, and checkpoint order.
- `reports/design_review.md` — adversarial pre-rollout review.
- `data/rollout_tasks_seed77114.jsonl` — executable truth and oracle restarts.
- `data/parent_rollout_input_seed66114.jsonl` — oracle-free vLLM input.
- `data/rollout_task_manifest.json` and `data/design_receipt.json` — freshness and design receipts.
- `data/failure_inventory_seed66114.json` — complete frozen failure inventory.
- `data/counterfactual_restart_source.jsonl` — 52 clean selected restarts.
- `data/restart_selection_receipt.json` and `data/selection_summary.json` — quota and composition receipts.
- `data/sft_blend.jsonl` and `data/predecessor_stream_manifest.json` — self-contained authenticated replay lineage.
- `data/source_token_lengths.json` — exact trainer-encoder source measurements.
- `data/stream_manifest.json`, `data/replay_control.jsonl`, and
  `data/counterfactual_restart_candidate.jsonl` — exact integral partition and frozen streams.
- `data/stream_token_receipt.json` — independent final-stream exposure validation.
- `reports/compute_review.md` — second adversarial review and control-only authorization.
- `reports/local_design_review.md` — explicit-composite and fresh-local adversarial review.
- `data/local_tasks_seed88010.jsonl`, `data/local_input_seed88010.jsonl`, and
  `data/local_design_receipt.json` — executable truth, hidden-free input, and frozen local protocol.
- `src/vllm_runner.py` — pinned same-backend runner with explicit-composite gate.

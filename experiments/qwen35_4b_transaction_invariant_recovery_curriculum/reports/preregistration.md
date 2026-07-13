# Preregistration: transaction-invariant recovery curriculum

Frozen before any candidate training or Qwen generation for this experiment.

## Claim under test

Action-seam supervision on diverse transactional repairs, mixed with complete
conditional-recovery task blocks, installs a transferable validate→copy→commit
program in one Qwen3.5-4B checkpoint. It must beat both matched replay-only
training and matched-compute sampling while retaining locality and the broader
coding loop.

## Immutable lineage and banks

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Training parent: merged recovery-action checkpoint, weight SHA-256
  `991d2da239ac2878dccac72ece419b00f37ac15d6a970538f384c36162b89aea`.
- Locality and Menagerie incumbent: C54 apex, weight SHA-256
  `c933168075a0fc011f31ee54f83d722cc91423ce1a0ac19bfe40dfab92f608d5`.
- Frozen recovery replay SHA-256
  `c2cb0f1cb151f7c23bdb5bef44483e48481d0bafb0a8273e145ff36f3fa8e552`.
- Primary: 24 transaction task blocks + 24 replay blocks. Control: 48 replay
  blocks. Each block has exactly seven transitions, for 336 rows/arm.
- Both banks have zero think loss and exactly 38,248 weighted action tokens per
  operator per epoch. Training is rank 32, alpha 64, LR 3e-5, batch 4 × seven
  transition-stratified accumulation steps, six epochs, seed 43.

The primary is fixed before training. Replay-only is never eligible to replace
it, even if it happens to score higher.

## Substrate and firewall

Transaction train families are `batch_inventory`, `ledger_transfers`,
`seat_groups`, `quota_claims`, `flag_batch`, and `rename_batch`. Transfer
families are `atomic_reservations`, `atomic_debits`,
`atomic_membership_moves`, and `atomic_patchset`. The first is a frozen sentinel
from the predecessor; the latter three are new API skins.

For every generated task and seed, the initial and partial workspaces must fail
visible and hidden executable checks and the full patch must pass both. Only
public issue text/source/visible tests reach the model. Hidden executable code,
hidden output, and repair objects may appear only inside the host evaluator.
Nothing under `benchmarks/` may be read or imported; Menagerie is invoked only
through its public CLI after all white-box gates pass.

## Matched compute and interface

All agent calls use the copied vLLM 0.24 runner and merged checkpoints with 512
thinking tokens plus 512 answer tokens.

- Deep recovery: one trajectory × six calls = 6,144 reserved tokens/case.
- Sample-more: two trajectories × three calls = 6,144 reserved tokens/case,
  scored pass-if-either by hidden evaluation.
- Normal loop: one × eight calls versus two × four calls, both 8,192 reserved.

Every arm shares backend, engine geometry, prompts, budgets, and task manifests.
Report sampled tokens, turns, invalid actions, answer-cap contacts,
verification, commit, and conditional-transition retention.

## Order and gates

### Locality

Before any behavior generation, compare primary to C54 apex on the frozen 48
fresh non-coding contexts. Require median centered non-target logit drift ≤0.15
and mean entropy delta ≥−0.05. Record mean varentropy delta diagnostically.
Failure stops all behavioral and benchmark exposure.

### Trained-family calibration

Generate parent and replay-only controls first. Before primary exposure, require
the theoretical success ceiling of 1.0 to make all bars attainable. Primary
then must satisfy:

- success ≥0.80;
- success ≥ parent +0.15 and replay-only +0.10;
- rejected-patch valid changed-within-two ≥0.80;
- failed-test changed-patch-within-two ≥0.70;
- invalid-action and answer-cap rates each ≤ parent +0.02.

### Transaction transfer dev and confirm

For each 64-case recovery block (four families × eight tasks × two states),
generate parent deep, replay-only deep, and parent sample-more before primary.
Prove the bars feasible, then require primary:

- ≥ parent +0.10, replay-only +0.05, and sample-more +0.05;
- paired case-bootstrap 95% lower bound versus parent ≥0 (10,000 resamples);
- rejected changed-within-two ≥0.80 and failed changed-within-two ≥0.70;
- invalid-action and answer-cap rates each ≤ parent +0.02;
- nonnegative family delta versus parent on at least three of four families and
  no family below −0.10.

Development uses seed 86500. Only an all-pass result authorizes the unchanged
confirmation at seed 86600. No threshold, family, or model adaptation occurs
between blocks.

### Broad recovery and normal retention

On `lease_cache`, `quorum_value`, `pattern_router`, and `rate_buckets` at seed
86700, run 40 normal cases and 80 controlled-recovery cases for parent and
primary. Require recovery success and normal success each ≥ parent −0.03;
normal verified-given-success ≥0.75 and commit-given-verified ≥0.70; both
transition rates ≥0.80/0.70; and invalid/cap deltas ≤+0.02.

### Menagerie

Only all prior gates authorize fresh paired `quick` and `medium` CLI events.
Use seeds absent from the public registry and compare candidate vs C54 apex on
the same backend/decode. The benchmark gate requires one tier delta ≥+0.02 and
no tier delta <−0.03. Preserve every event and aggregate receipt. A white-box
positive with benchmark failure is still a procedural-coding result, not a
general capability unlock.

## Stop labels

- `LOCALITY_FAIL`
- `CALIBRATION_INFEASIBLE`
- `CALIBRATION_FAIL`
- `TRANSACTION_DEV_INFEASIBLE`
- `TRANSACTION_DEV_FAIL`
- `TRANSACTION_CONFIRM_INFEASIBLE`
- `TRANSACTION_CONFIRM_FAIL`
- `BROAD_RETENTION_FAIL`
- `MENAGERIE_FAIL`
- `TRANSACTION_INVARIANT_CAPABILITY_POSITIVE`

All negative controls and stops remain preserved. No new experiment is added to
this result-bearing directory after the first full run.

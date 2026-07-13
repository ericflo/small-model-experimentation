# Preregistration: validation-policy counterexample curriculum

Frozen before any candidate/control training or Qwen generation.

## Claim under test

A one-transition residual action-seam curriculum from near-correct failed-test
states installs the distinction negative→`ValueError` versus missing or
insufficient→`False` into the transaction-trained Qwen3.5-4B policy. The gain
must transfer across new public data representations, repair the known atomic-
reservation sentinel, beat matched extra training and matched-compute sampling,
and preserve locality and the full conditional tool loop.

## Immutable lineage

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Training parent: predecessor transaction candidate, weight SHA-256
  `1cf5fbca317808d6d00225f5cd533c82c7e1602b2b2e5e2da8f4307b01941ba3`.
- Locality and Menagerie incumbent: C54 apex, weight SHA-256
  `c933168075a0fc011f31ee54f83d722cc91423ce1a0ac19bfe40dfab92f608d5`.
- Frozen prior bank SHA-256
  `9c196d1e7e49881bbf151e1575c98811bcdca66e6ef38858c34f60f1256b9315`;
  frozen receipt SHA-256
  `8c2c339bd78399fbfac1d29053bee6f92935ed97df52ce4d2d36c902bce6e63e`.

An immutable receipt hashes the config, intake, preregistration, design review,
task/bank/runtime/training/analyzer/benchmark code at the pushed design commit.
GPU/model stages refuse to run if any frozen file changes or that commit is not
an ancestor of `HEAD`.

## Treatment and control

The prior source contains 24 transaction blocks and 24 complete recovery
blocks, each with exactly seven state→action rows. The control relabels all 336
rows for an extra update from the learned parent. The treatment uses the same
48 blocks but replaces only `diagnosis_to_changed_patch` in each recovery block
with the corresponding row from 24 fresh policy tasks. Thus 24/336 rows differ
and 312/336 retain prior behavior. No `inspect_to_patch`, rejected-patch,
verify, commit, or generic full-solution policy row is added.

Both arms have 48 task blocks, 336 rows, the same seven transition strata and
operator counts, zero think loss, and exactly 38,248 weighted action tokens per
operator per epoch. Candidate bank SHA-256 is
`940da93e3849e7a2ebfb1555add666e8a4039f13548dea556c93191541cc305a`;
control bank SHA-256 is
`52424053bfae8192cb44772d661121b8c1b5f83418ca18bf799eb9812392c10e`.
Training is three epochs, LR 2e-5, rank 32, alpha 64, dropout .05, batch four,
seven transition-stratified accumulation steps, max length 4096, seed 47.
Primary is fixed to `policy_counterexample`; the control is never eligible for
posthoc selection.

## Procedural substrate and independence

Training families are `policy_warehouse_orders`, `policy_compute_claims`,
`policy_credit_holds`, `policy_job_allocations`, `policy_cargo_manifests`, and
`policy_part_bundles`, with bundle-map, record, and tuple representations.
Transfer families are the known `atomic_reservations` sentinel plus fresh
record `policy_power_draws`, tuple `policy_token_spends`, and bundle-map
`policy_lane_bookings`. The sentinel is gated separately and never counts as a
fresh family.

Public-content digests exclude task ID/split. Bank (24) and trained calibration
(24) are internally unique and disjoint. Development (32) and confirmation
(32) are internally unique and mutually disjoint. Every initial and partial
workspace fails both visible and hidden executables; every oracle passes both.
The partial source already implements copy, existence/capacity validation,
atomic application, input nonmutation, and `False` rejection; only the negative
exception is missing.

## Firewall and inference parity

Hidden executables, hidden results, and repair objects stay host-side. No code
under `benchmarks/` is read or imported. Menagerie is invoked only through its
public CLI after all white-box gates and its raw output is immediately reduced
to aggregate/per-family scores.

All coding arms use the copied vLLM 0.24 runner and merged checkpoints with 512
thinking + 512 answer tokens. Deep recovery is one trajectory × six calls;
sample-more is two × three, both reserving 6,144 tokens/case. Normal is one ×
eight versus two × four, both 8,192. Backends, engine geometry, prompts,
budgets, and task/content manifests match byte-for-byte. Official children set
`PYTHONHASHSEED=0` and `PYTHONDONTWRITEBYTECODE=1`.

## Ordered gates

### Locality

Before behavior generation, compare candidate directly with C54 apex on 48
fresh non-coding contexts. Require median centered non-target logit drift ≤.15
and mean entropy delta ≥−.05. Record mean varentropy delta diagnostically; it
does not label examples or weight tokens. Failure stops all behavior and
Menagerie.

### Trained-family calibration

Generate parent and matched-control receipts before candidate exposure and
prove the theoretical ceiling makes every bar attainable. Candidate must have:

- success ≥.80, parent +.15, and control +.10;
- rejected-patch valid changed-within-two ≥.80 and failed-test changed-patch-
  within-two ≥.70;
- invalid-action and answer-cap rates each ≤ parent +.02.

### Policy transfer development and confirmation

Each block contains 64 recovery cases: four families × eight repositories ×
two starting states. Generate parent deep, matched control deep, and parent
sample-more before candidate. Candidate must have:

- success ≥ parent +.10, control +.05, and sample-more +.05;
- paired case-bootstrap 95% lower bound versus parent ≥0 (10,000 resamples);
- atomic-reservations success ≥.50;
- nonnegative delta versus parent on all three genuinely fresh families and no
  fresh-family regression below −.10;
- rejected changed-within-two ≥.80, failed changed-within-two ≥.70,
  verified-given-success ≥.75, and commit-given-verified ≥.70;
- invalid-action and answer-cap rates each ≤ parent +.02.

Development uses seed 87300. Only all-pass development authorizes unchanged,
content-disjoint confirmation at seed 87400. No threshold, family, model, or
decode adaptation occurs between them.

### Broad retention

On `lease_cache`, `quorum_value`, `pattern_router`, and `rate_buckets` at seed
87500, compare parent/candidate on 80 controlled-recovery and 40 normal cases.
Require recovery and normal success each ≥ parent −.03; verified ≥.75; commit
≥.70; rejected/failed transition ≥.80/.70; invalid/cap delta ≤+.02.

### Menagerie

Only all prior gates run quick seed 71301 and medium seed 71302, each paired
candidate versus C54 apex on the public CLI with the same backend/decode. At
least one tier delta must be ≥+.02 and neither may be <−.03. A white-box
positive with benchmark failure is procedural coding transfer, not a general
capability unlock.

## Stop labels

- `LOCALITY_FAIL`
- `CALIBRATION_INFEASIBLE`
- `CALIBRATION_FAIL`
- `POLICY_DEV_INFEASIBLE`
- `POLICY_DEV_FAIL`
- `POLICY_CONFIRM_INFEASIBLE`
- `POLICY_CONFIRM_FAIL`
- `BROAD_RETENTION_FAIL`
- `MENAGERIE_FAIL`
- `VALIDATION_POLICY_CAPABILITY_POSITIVE`

Every negative, control, and stopped receipt remains preserved. No follow-up is
added to this result-bearing directory after its first full run.

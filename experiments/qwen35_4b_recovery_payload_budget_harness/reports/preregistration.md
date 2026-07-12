# Preregistration: payload-capable recovery agent harness

Frozen before any 512-answer-token behavioral evaluation or fresh-locality
measurement.

## Claim

This experiment tests whether the fixed, locality-safe λ=.18 checkpoint already
contains a transferable verifier-conditioned coding-recovery policy that the
predecessor's 256-token tool-answer slot prevented from being deployed. It does
not change weights and cannot rescue or relabel the predecessor's registered
policy-gate failure.

## Immutable Inputs

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Frozen models: C54 apex (`base`), matched happy-action, full recovery-action,
  and action→reason λ=.18 candidate. Exact weight hashes are in the config.
- Candidate was chosen by the completed predecessor before this harness was
  designed: 58/60 calibration recovery and 0.104 original-block locality.
- Calibration tasks: seed 84700, six training families, five tasks/family, two
  controlled public recovery states/task (60 cases).
- Transfer development: seed 84800, four untouched families, ten tasks/family
  (80 recovery cases and 40 normal cases/arm).
- Transfer confirmation: seed 84900, identical sizes, independently generated.
- Locality: 48 new non-coding contexts with no content-hash overlap with either
  predecessor locality block. The file hash is frozen in config.

## Harness and Compute Contract

Every arm uses the same copied vLLM runner, engine geometry, prompt, tool API,
call limit, and 512-thinking/512-answer split. No HF/vLLM behavioral comparison
is allowed.

- Deep recovery: one greedy trajectory × six calls × 1,024 tokens = 6,144
  reserved tokens/case.
- Sample-more recovery: two stochastic trajectories × three calls × 1,024 =
  6,144.
- Deep normal: one × eight × 1,024 = 8,192.
- Sample-more normal: two × four × 1,024 = 8,192.

This is a matched increase in the maximum action payload, not a candidate-only
compute grant. Actual sampled tokens, answer-cap hits, invalid actions, calls,
and turns are reported for every arm.

## Transition Definitions

Controlled states are unchanged. Hidden tests remain host-only.

- `rejected_patch_valid_changed_within_two`: within the first two generated
  turns after a public rejected patch, workspace-changing `PATCH` occurs either
  immediately or after exactly one valid `INSPECT`; `INVALID→PATCH` does not
  satisfy this metric. The intervening operator sequence and immediate rate are
  retained diagnostically.
- `failed_test_changed_patch_within_two`: a workspace-changing patch appears
  within two generated turns after the public visible-test failure.
- `passed_test→commit`: commit must occur after a passing visible test that
  follows the final patch.

The two-turn rejected definition is fixed from post-stop predecessor forensics,
where all 30 λ=.18 cases changed within two and solved, before any new-budget
output. It measures the requested conditional recovery while permitting
defensible reinspection.

## Stage 1: Fresh Locality

On the 48 new contexts, compare candidate with apex using the same exact method:
exclude apex top-20 continuations, center raw logits, and take median absolute
non-target drift. Candidate passes iff all values are finite, context count and
tokenization match, drift ≤0.15, and mean entropy change ≥−0.05 nats. Record
varentropy without a hard threshold. Failure stops before any 512-answer
behavioral run.

## Stage 2: Calibration Feasibility and Gate

Run base, happy, and action controls first. Use metric hard ranges to prove every
absolute/delta gate attainable before candidate evaluation. If feasible, run
the candidate once on the same 60 cases.

Candidate passes iff all hold:

- recovery success ≥base +0.05, happy +0.03, and action +0.03;
- invalid actions/turn ≤base +0.02;
- answer-cap hits/turn ≤base +0.02;
- valid rejected changed-within-two ≥0.80 and ≥base +0.10;
- failed-test changed-within-two ≥0.60 and ≥base +0.05;
- verified-given-success ≥0.85; and
- commit-given-verified ≥0.80.

There is no candidate selection or budget selection. Failure stops before any
transfer task is generated/evaluated.

## Stage 3: Transfer Development

Run frozen controls first: base deep, happy deep, action deep, base sample-more,
base with explicit recovery scaffold, and base normal. Prove every gate feasible
before candidate recovery/normal evaluation.

Candidate passes iff all hold:

- recovery ≥base +0.05, happy +0.03, action +0.03, sample-more +0.03, and
  scaffold +0.03;
- paired casewise bootstrap 95% lower bounds versus base and sample-more ≥0
  (10,000 resamples, frozen seeds);
- valid rejected changed-within-two ≥0.80 and ≥base +0.10;
- failed-test changed-within-two ≥0.60 and ≥base +0.05;
- normal success ≥base −0.03;
- normal verified-given-success ≥0.70 and ≥base −0.05;
- normal commit-given-verified ≥0.65 and ≥base −0.05;
- recovery invalid actions/turn and answer-cap hits/turn each ≤base +0.02;
- at least three of four family deltas versus base are nonnegative and none is
  below −0.10; and
- fresh locality remains passing.

Failure stops before confirmation and Menagerie.

## Stage 4: Transfer Confirmation

Repeat the complete control-first, feasibility-first evaluation at seed 84900
with identical thresholds. No adaptation, fallback, budget change, prompt
change, or family-specific rule is permitted.

## Stage 5: Menagerie

Only after both transfer blocks pass may fresh paired seeds be assigned through
the public benchmark CLI. Run apex and candidate on quick and medium with
identical seeds/settings. Require at least one tier +0.02 and no tier below
−0.03, with paired task-level differences and uncertainty reported. Never read
or import benchmark sources, items, transcripts, family code, or detailed
results; the CLI score instrument is the only interface.

## Stop Labels

- `FRESH_LOCALITY_FAIL`
- `CALIBRATION_INFEASIBLE`
- `PAYLOAD_MECHANISM_FAIL`
- `TRANSFER_DEV_FAIL`
- `TRANSFER_CONFIRM_FAIL`
- `MENAGERIE_NULL`
- `CAPABILITY_ELICITATION_POSITIVE`

All stops and controls are preserved. A result-bearing failure may only motivate
a new experiment; no threshold or interface is changed in this directory.

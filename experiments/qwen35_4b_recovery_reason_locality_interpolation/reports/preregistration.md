# Preregistration: locality-first recovery-reason interpolation

Frozen before any scaled checkpoint was merged or evaluated.

## Claim

The experiment tests whether a locality-compliant point on the full-dose
recovery-action→recovery-reason weight segment improves verifier-conditioned
coding recovery and transfers to unseen procedural repository families. It does
not test whether lowering the plan loss during training would produce the same
checkpoint.

## Immutable Inputs

- Model lineage: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Base: frozen C54 `apex_replay` merged checkpoint.
- Endpoints: parent `recovery_action` and `recovery_reason` r32/alpha64 LoRAs,
  trained from the same base with identical rows, batch order, seed, optimizer,
  and 120-step schedule; only the plan-token mask/weight differs.
- Parent artifact, endpoint, and calibration-result SHA-256 values are frozen in
  `configs/default.yaml` and verified before execution.
- Original locality screen: the exact parent's 48 contexts.
- Independent locality confirmation: 48 new non-coding contexts committed
  before execution, with disjoint content hashes.
- Calibration: six training families, seed 84700, five tasks/family, two public
  recovery states/task, 60 cases total.
- Transfer development: four held-out families, seed 84800, ten tasks/family,
  80 recovery cases plus 40 normal-loop cases per arm.
- Transfer confirmation: the same four generators with seed 84900 and the same
  sizes. No task from either block has been generated, inspected, or evaluated
  before this experiment passes the preceding gates.

## Intervention and Ladder

For each adapted module, reconstruct the two LoRA deltas in float32 and merge:

```text
delta(lambda) = delta_action + lambda * (delta_reason - delta_action)
W(lambda) = W_apex + delta(lambda)
```

The only eligible scaled values are `0.10`, `0.18`, `0.24`, and `0.30`.
`lambda=0` (action) and `lambda=1` (reason) are endpoint controls. Merges cast
once to bfloat16, disable TF32 for delta arithmetic, and preserve hashes and
per-module norm receipts. The ladder may not be refined, extended, or shifted
after observing locality or behavior.

## Inference Contract

All behavioral arms use the copied parent vLLM runner and identical engine
geometry. Greedy deep recovery reserves six calls of 512 thinking + 256 answer
tokens. Matched sampling uses two stochastic trajectories of three calls, for
the same 4,608-token/case reservation. Normal deep and matched-sampling arms
similarly reserve 6,144 tokens/case. Same seeds do not license backend mixing;
no HF behavioral arm is compared with vLLM.

The looping harness exposes only tree/read/search/exact-replacement patch/test/
submit actions. Rejected-patch and failed-visible-test starting states are
deterministic public states. Hidden executables and their output never enter
model context.

## Stage 1: Locality Screen

Run apex and all six points (`0`, four scales, `1`) on the original 48 frozen
non-coding contexts. For each context, exclude apex's top-20 continuations,
center both raw-logit vectors, and compute the median absolute non-target
difference. Also compute exact distribution entropy and varentropy.

A point passes iff:

- all values are finite and all 48 contexts tokenize identically;
- median non-target centered-logit drift is at most `0.15`; and
- mean entropy change is at least `-0.05` nats.

Varentropy is diagnostic because no prior calibrated safety threshold exists.
No scaled point passing means immediate stop before any scaled behavioral
evaluation. Only locality-passing points, plus the action endpoint if it passes,
may enter calibration. The known reason endpoint is never eligible.

## Stage 2: Frozen Calibration Selection

Reuse checksum-frozen base, happy, and action parent results. Evaluate only
locality-pass scaled points on the exact same 60 cases. A candidate is eligible
iff all of the following hold:

- recovery success is at least base +0.05;
- success is at least happy +0.03 **or** the mean of rejected immediate change,
  failed immediate diagnose/revise, and failed changed-patch-within-two exceeds
  happy by at least 0.08;
- invalid actions/turn are no more than base +0.02;
- rejected-patch immediate changed-patch rate is at least 0.60; and
- failed-test changed-patch-within-two rate is at least 0.60.

Among eligible points, select lexicographically by overall success,
worst-scenario success, negative invalid-action rate, transition composite, and
negative lambda. Thus exact ties prefer less plan movement. This selection is
made once. If no candidate is eligible, stop before confirmation and transfer.

## Stage 3: Independent Locality Confirmation

Evaluate only the selected point on the new 48-context block with the identical
drift and entropy thresholds. Failure stops the experiment. No fallback to a
different scale is permitted, and the confirmation block may not inform scale
selection.

## Stage 4: Held-out Transfer Development

Run base, happy action, full recovery action, matched sampling, external
recovery scaffold, and base normal-loop controls first. Before candidate
evaluation, prove every absolute and delta gate is attainable under the metric's
hard range and write `analysis/transfer_dev_feasibility.json`.

The selected point passes only if every configured gate passes:

- recovery success ≥ base +0.05, happy +0.03, matched sampling +0.03, and
  external scaffold +0.03;
- paired bootstrap 95% lower bounds versus base and matched sampling ≥0;
- success ≥ action-only −0.02, and either invalid actions improve by at least
  0.03 or rejected-patch immediate change improves by at least 0.10;
- rejected immediate change and failed changed-patch-within-two are each ≥0.60
  and ≥base +0.05;
- normal success ≥base −0.03; verified-given-success ≥0.70 and ≥base −0.05;
  commit-given-verified ≥0.65 and ≥base −0.05;
- recovery invalid actions/turn ≤base +0.02;
- at least three of four families are nonnegative versus base and no family is
  below base −0.10; and
- independent locality confirmation remains passing.

Paired bootstrap uses 10,000 casewise resamples with frozen seeds. A failure
stops before confirmation and Menagerie.

## Stage 5: Held-out Transfer Confirmation

Repeat the complete control, feasibility, candidate, and gate battery at seed
84900. Thresholds do not change. Failure leaves Menagerie sealed.

## Stage 6: Menagerie

Only after both transfer blocks pass may fresh paired seeds be assigned through
the public benchmark CLI. Run quick and medium tiers for frozen apex and the
single selected checkpoint with identical seeds and inference settings. Require
at least one tier to improve by `+0.02` and neither tier to regress by more than
`0.03`; report paired task-level differences and uncertainty. No benchmark
content, family implementation, item, transcript, or detailed result may be
read, imported, or used for training or selection.

## Decision Labels

- `NO_LOCAL_REGION`: no scaled point passes the screen.
- `LOCAL_BUT_NO_BEHAVIOR`: local points exist but none clears calibration.
- `LOCALITY_CONFIRM_FAIL`: selected calibration winner fails fresh locality.
- `TRANSFER_DEV_FAIL` or `TRANSFER_CONFIRM_FAIL`: white-box capability does not
  generalize sufficiently; no Menagerie exposure.
- `MENAGERIE_NULL`: all white-box gates pass but black-box criteria fail.
- `CAPABILITY_ELICITATION_POSITIVE`: independent locality, both family-held-out
  blocks, matched-compute controls, and Menagerie criteria all pass.

All negative controls and stop receipts are preserved. No threshold, scale,
candidate, family, or fallback may be changed inside this result-bearing
directory after execution begins.

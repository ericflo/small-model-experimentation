# Locality-first recovery-reason interpolation

**Status:** finished

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can a conditional recovery policy survive locality and
  family-held-out transfer, then improve the black-box Menagerie instrument?
- Direct predecessor: `qwen35_4b_verifier_conditioned_recovery_bank`.
- Prior anchors: C50 (emission-seam signal placement), C52 (exact-logit
  locality), C54 (weight-space interpolation can be non-convex), and C28
  (reasoning helps only when its content is useful).

## Question

Does the weight-space segment between the parent's locality-safe
recovery-action adapter and its behaviorally stronger but non-local
plan-supervised sibling contain a point that retains full action learning,
repairs invalid recovery behavior, and transfers to unseen procedural coding
families?

## Hypothesis

The parent arms differ only in a nominal 5% plan loss over byte-identical rows,
with the same base, seed, batches, and schedule. Action-only reached 85.0%
trained-family recovery at 0.098 locality drift but emitted 19.1% invalid turns;
the plan arm reached 91.7%, 5.9% invalid turns, and much shorter trajectories,
but drifted 0.303. We hypothesize that low-dose movement from action toward
reason corrects tool-call validity and conditional recovery before the broad
off-policy lexical pressure crosses the 0.15 locality ceiling.

The hypothesis fails if locality-safe points remain action-like, the chosen
point fails a second independent locality block, or the effect does not beat
the incumbent and matched-compute controls on untouched families.

## Frozen Intervention

For the two parent LoRA deltas from the same frozen C54 apex checkpoint:

```text
W(lambda) = W_apex + (1 - lambda) * delta_action + lambda * delta_reason
```

The preregistered scaled candidates are `lambda = {0.10, 0.18, 0.24, 0.30}`.
The full action (`0.0`) and reason (`1.0`) endpoints are controls. This preserves
the full useful action update while scaling only the learned action→reason
contrast; it is not equivalent to retraining with a smaller plan-token weight.

The ladder was chosen before any scaled merge or evaluation. Linear
interpolation of the known endpoint drifts predicts the 0.15 frontier near
`lambda = 0.25`, so the ladder brackets rather than searches that boundary.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e...`, warm-started from the
  frozen C54 `apex_replay` checkpoint.
- Source data: the parent's own execution-verified, replay-minimized procedural
  repository trajectories. No new training occurs.
- Calibration: the already-designated 60-case block over six training
  families; parent base/happy/action results are checksum-frozen and reused.
- Transfer: two untouched 80-case blocks over four algorithmically different
  procedural families (`transfer_dev` seed 84800, `transfer_confirm` seed
  84900). No fallback candidate is allowed after selection.
- Controls: frozen apex, matched happy-action training, full recovery-action
  endpoint, explicit runtime recovery scaffold, and two shorter apex
  trajectories with the same reserved calls/tokens.
- Primary metrics: hidden-test repository success, rejected-patch→changed-patch
  and failed-test→changed-patch retention, invalid actions, normal-loop
  solve/verify/commit retention, and paired/family deltas.
- Hidden boundary: hidden executable code and output stay host-only. Only
  booleans enter receipts. `benchmarks/` contents are never read or imported.

## Gate Order

1. Validate every parent checksum and merge all four frozen scales.
2. Run exact next-token locality on the original 48 non-coding contexts for
   both endpoints and all scales. Record entropy and varentropy. Only passing
   scales may be behaviorally evaluated.
3. Select once on calibration after hard gates for success, tool-call validity,
   and both recovery transitions. Lexicographic ties prefer fewer invalid
   actions and then less reason movement.
4. Test only that winner on a new disjoint 48-context locality block. Failure
   stops the experiment; no fallback is selected.
5. On each untouched transfer block, run frozen controls first and prove every
   registered gate mathematically reachable. The winner must beat base, happy,
   matched sampling, and scaffold; remain non-inferior to action-only while
   improving its validity or rejected-patch transition; and retain normal
   agent loops and at least three of four families.
6. Menagerie quick/medium remains sealed until both transfer blocks pass.

Exact thresholds and immutable hashes are in `configs/default.yaml`; the full
statistical contract is in `reports/preregistration.md`.

## Run

CPU and invariant smoke:

```bash
.venv/bin/python experiments/qwen35_4b_recovery_reason_locality_interpolation/scripts/run.py --smoke
```

One-scale merge and two-context GPU integration smoke:

```bash
.venv/bin/python experiments/qwen35_4b_recovery_reason_locality_interpolation/scripts/run.py --gpu-smoke
```

Resumable staged run:

```bash
.venv/bin/python experiments/qwen35_4b_recovery_reason_locality_interpolation/scripts/run.py --full
```

## Results

**Verdict: `LOCAL_BUT_NO_BEHAVIOR`; stopped at the frozen calibration policy
gate.** Transfer and Menagerie remained sealed.

All four mixtures passed the original locality screen. The path was much safer
than endpoint-linear prediction: drift rose only from 0.100 at λ=.10 to 0.121 at
λ=.30, while the reason endpoint remained the known 0.303 failure. Entropy and
varentropy also stayed inside the registered bounds.

| Arm | Recovery | Failed test | Rejected patch | Invalid/turn | Rejected immediate change | Drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 48.3% | 53.3% | 43.3% | 5.7% | 0.0% | — |
| happy action | 81.7% | 80.0% | 83.3% | 2.0% | 33.3% | 0.083* |
| action endpoint | 85.0% | 73.3% | 96.7% | 19.1% | 16.7% | 0.098 |
| λ=.10 | 88.3% | 76.7% | **100%** | 11.6% | 16.7% | 0.100 |
| λ=.18 | **96.7%** | **93.3%** | **100%** | 10.4% | 33.3% | 0.104 |
| λ=.24 | 95.0% | 90.0% | **100%** | 10.2% | 33.3% | 0.111 |
| λ=.30 | 88.3% | 76.7% | **100%** | 10.0% | 33.3% | 0.121 |
| reason endpoint | 91.7% | 100% | 83.3% | 5.9% | 83.3% | 0.303 |

`*` Happy locality is the parent's same-method exploratory control, shown only
for context. The selector admitted no candidate: every mixture exceeded the
base+2pp invalid-turn ceiling and missed the registered 60% *immediate*
rejected-patch transition bar. Full metrics and hashes are in
`reports/result_receipt.json`.

## Interpretation

The weight direction contains a real, locality-safe behavioral optimum: λ=.18
adds +11.7pp over action-only and +15.0pp over happy on the selection block. It
does **not** yet qualify as deployable capability because the registered policy
guards failed.

Exploratory failure forensics change the next intervention. All 24 “invalid”
actions ended exactly at the 256-answer-token cap with thinking already closed;
they are truncated long patch JSON, not free-form slop. Eleven trajectories had
an invalid turn and nine still solved. Likewise, all 30 rejected-patch cases
made a changed patch within two turns: 20 sensibly used `INSPECT→PATCH`, while
only ten used immediate `PATCH→VERIFY`. Thus one failed gate detects a genuine
tool-payload budget bottleneck, while the other over-specifies immediacy instead
of the requested conditional transition.

The next experiment should freeze λ=.18, give every arm a realistic larger
answer payload under matched compute, and gate rejected recovery by changed
patch within two turns. That is a new harness experiment, not a reinterpretation
of this stopped result.

## Knowledgebase Update

- Program evidence: updated with the safe interpolation curve and calibration
  stop.
- Program backlog: queues a λ=.18 matched-compute answer-budget harness
  successor with semantically faithful two-turn transition retention.
- Claim ledger/synthesis: deferred; no transfer family or Menagerie seed was
  exposed.

## Artifacts

Small receipts and final analysis are committed. Merged 4B checkpoints,
trajectories, and detailed logits live under
`large_artifacts/qwen35_4b_recovery_reason_locality_interpolation` and are
tracked by `reports/artifact_manifest.yaml`.

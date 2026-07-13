# Payload-capable recovery agent harness

**Status:** finished

## Research Program

- Program: `agentic_breadth_installation`
- Direct predecessor: `qwen35_4b_recovery_reason_locality_interpolation`.
- Prior anchors: C50 (answer-emission placement), C54 (serial-compute
  capability), and the two conditional-recovery experiments immediately above.

## Question

Does the frozen locality-safe λ=.18 recovery checkpoint become a deployable,
family-transferring coding agent when its tool payload is no longer truncated
at 256 answer tokens and rejected-patch recovery is measured across one valid
inspection rather than only the next action?

## Hypothesis

The predecessor's λ=.18 checkpoint solved 58/60 familiar recovery cases at
0.104 locality drift. Its 24 invalid actions all had closed thinking and ended
exactly at the 256-token answer ceiling inside long exact-replacement JSON.
Separately, every rejected case changed code within two turns and solved; 20/30
used `INSPECT→PATCH`. We therefore expect a 512-thinking/512-answer harness to
reveal an already-present recovery policy, reduce cap hits and invalid actions,
and preserve the candidate's advantage over base, happy, action-only, scaffold,
and matched sampling.

The hypothesis fails if the candidate remains payload-bound, loses its
advantage when all controls receive the same larger answer slot, or does not
transfer to the untouched procedural families.

## Frozen Intervention

- Weights: exactly the predecessor's λ=.18 checkpoint, SHA-256 frozen in
  `configs/default.yaml`; no training or further interpolation.
- Per call: 512 thinking tokens plus 512 answer tokens for **every** model arm.
- Deep recovery: six calls = 6,144 reserved sampled tokens/case.
- Matched sampling: two trajectories × three calls × 1,024 = the same 6,144.
- Normal loops: eight deep calls versus two × four sampled calls, both 8,192.
- Rejected transition: a changed patch within two generated turns, with the
  valid paths restricted to immediate `PATCH` or `INSPECT→PATCH`. Immediate
  change remains diagnostic.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, pinned revision `851bf6e...`.
- Calibration: the known six-family/60-case block, rerun for base, happy,
  action, and candidate under the new identical harness. It is selection/gating
  evidence only.
- Fresh locality: 48 newly committed non-coding contexts disjoint from both
  predecessor blocks; the fixed candidate must pass before behavior.
- Transfer: the still-untouched four-family blocks at seeds 84800 and 84900.
- Controls: frozen apex, happy-action, full recovery-action, explicit runtime
  scaffold, and compute-matched sample-more.
- Metrics: hidden-test recovery, invalids, answer-cap hits, rejected and failed
  conditional transitions, normal solve/verify/commit, paired bootstrap, and
  per-family deltas.
- Firewall: hidden executable code/output remain host-only. No `benchmarks/`
  source, item, transcript, or result is read or imported.

## Gate Order

1. Validate all four model hashes and run the new 48-context locality audit
   (drift ≤0.15; entropy Δ≥−0.05; varentropy diagnostic).
2. Run calibration controls first and prove every threshold reachable before
   candidate evaluation.
3. Candidate must beat base +5pp, happy +3pp, and action +3pp; retain base-level
   invalid/cap-hit rates; reach ≥80% valid rejected change within two turns and
   ≥60% failed-test change within two; and retain verification/commit.
4. Repeat the complete control-first, feasibility-first battery on
   `transfer_dev`, including scaffold and matched sampling.
5. Repeat without changes on `transfer_confirm`.
6. Only then assign fresh paired Menagerie quick/medium seeds through the public
   CLI. One tier must improve ≥2pp and neither may regress >3pp.

Exact thresholds are in `configs/default.yaml` and the statistical contract is
in `reports/preregistration.md`.

## Run

```bash
.venv/bin/python experiments/qwen35_4b_recovery_payload_budget_harness/scripts/run.py --smoke
.venv/bin/python experiments/qwen35_4b_recovery_payload_budget_harness/scripts/run.py --gpu-smoke
.venv/bin/python experiments/qwen35_4b_recovery_payload_budget_harness/scripts/run.py --full
```

## Results

**Verdict: `TRANSFER_CONFIRM_FAIL`; Menagerie remained sealed.** The matched
payload repair worked as an interface intervention. Fresh locality passed at
0.114 drift (entropy Δ −0.0059; varentropy Δ −0.0105), candidate answer-cap
hits fell to 0.5%/7.8%/7.9% of turns on calibration/dev/confirm, and all
candidate rejected- and failed-state cases changed the patch within two turns.

| Block / arm | base | happy | action | sample-more | scaffold | candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration | 36.7% | 75.0% | 96.7% | — | — | **100%** |
| transfer dev | 58.8% | 61.3% | 66.3% | 50.0% | 66.3% | **71.3%** |
| transfer confirm | 60.0% | 56.3% | **68.8%** | 46.3% | 63.8% | **68.8%** |

The dev block passed every registered gate: candidate was +12.5pp versus base
and +21.25pp versus matched sampling, with paired 95% lower bounds +3.75pp and
+8.75pp. On confirmation it retained +8.75pp versus base and +22.5pp versus
matched sampling, improved all four families, preserved normal success exactly
(25.0% versus 25.0%), but tied action-only instead of beating it by 3pp. That
single failed check stopped the run.

Exploratory paired forensics expose a stable opportunity rather than a missing
global dose. Candidate and action-only each had eight exclusive confirmation
wins, and their success union was 78.75% on both independent transfer blocks.
The losses were structured: action-only won seven confirmation
`pattern_router` rejected-patch disagreements, while the reason mixture won
four `rate_buckets` rejected-patch disagreements. Full metrics, hashes, and the
disagreement cells are committed in `reports/result_receipt.json`.

## Interpretation

The predecessor's two apparent policy failures were indeed measurement and
interface failures: a 512-token action slot removes almost all candidate
truncation, and valid `INSPECT→PATCH` recovery transfers perfectly. But the
reason mixture is not a uniformly better policy than action-only. It trades
wins across held-out algorithms and cannot claim a breakthrough from aggregate
dev selection.

The next intervention should exploit the replicated 78.75% union with a public
verifier, not another family-specific weight mixture or posthoc router. Fork
the two locality-safe policies at recovery states, execute bounded branches,
select only from visible test/rejection evidence, and bank the winning
trajectories with the same conditional-transition balance. It must beat a
compute-matched two-trajectory baseline before any distillation or Menagerie.

## Knowledgebase Update

- Program evidence: records the interface repair, independent confirm stop, and
  replicated candidate/action complementarity.
- Program backlog: replaces global dose tuning with verifier-selected branching
  and conditional winner banking.
- Claim ledger: unchanged; no Menagerie event was exposed.

## Artifacts

Small receipts and final analyses are committed. Detailed trajectories live in
`large_artifacts/qwen35_4b_recovery_payload_budget_harness` as specified by
`reports/artifact_manifest.yaml`.

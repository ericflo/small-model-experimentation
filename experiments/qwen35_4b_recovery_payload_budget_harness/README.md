# Payload-capable recovery agent harness

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

Pending the frozen run. A larger answer budget is not itself a capability gain;
the fixed checkpoint must beat every matched control on untouched families.

## Interpretation

Pending. A null after cap hits disappear would refute the harness-bottleneck
explanation. A calibration-only gain would remain familiar-family process
control, not breadth.

## Knowledgebase Update

- Program evidence: pending.
- Program backlog: pending.
- Claim ledger/synthesis: pending claim-grade transfer and Menagerie.

## Artifacts

Small receipts and final analyses are committed. Detailed trajectories live in
`large_artifacts/qwen35_4b_recovery_payload_budget_harness` as specified by
`reports/artifact_manifest.yaml`.

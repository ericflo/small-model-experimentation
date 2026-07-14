# Replay-Anchored Universal Curriculum Continuation

**Status:** in-progress · since 2026-07-13 · aggregate-only pilot remains

## Research program

- Program: `agentic_breadth_installation`
- Parent: `qwen35_4b_universal_curriculum`
- Prior anchors: C14, C53, C54, and C56

## Question

Can a low-rate continuation from the strong C53 `blend` policy install the parent's
truth-audited general procedures without displacing broad behavior when every update
window is anchored by broad replay?

The parent sequential arm made a useful but narrow update: fresh synthetic accuracy
rose from 0.5000 to 0.6923, yet the aggregate benchmark score fell 0.1385 behind
`blend` and three public families regressed below base. This experiment changes only
the integration geometry. It does not add benchmark-shaped data or inspect benchmark
items.

The parent's preregistered from-base replay union subsequently reached 0.6923 local
accuracy but failed its prospective parse-rate (0.8462) and cap-contact (4/26) gates;
its benchmark stayed sealed. That strengthens the rationale for retaining the mature
policy during integration without changing this experiment's already-frozen design.

## Frozen design

Both trained arms warm-start the immutable C53 `blend` adapter and use
`Qwen/Qwen3.5-4B` at revision `851bf6e...`, learning rate `1e-5`, LoRA rank 32 / alpha
64, batch 1 x accumulation 8, max length 4,096, `w_think=0.2`, seed 42, and one epoch
(190 optimizer steps).

- `warm_union`: 400 designed rows plus 1,120 broad replay rows.
- `replay_refresh`: the identical 1,120 replay rows plus 400 additional replay rows.
- `blend`: immutable strong control.
- `base`: pinned reserialized base control.

The exact nested doses are deterministically derived from copied, checksum-pinned
parent artifacts. Both contain 1,520 rows with zero tokenizer skips. The candidate has
1,231,404 forward tokens; the replay-only control has 1,444,589, making the compute
asymmetry conservative for the candidate.

See [idea_intake.md](idea_intake.md),
[design_review.md](reports/design_review.md), and
[preregistration.md](reports/preregistration.md) for the full rationale and gates.

## Run

The non-GPU smoke path checks deterministic bytes, source and dose hashes, exact row
counts, zero skips, the sole permitted model identity, and Python syntax:

```bash
.venv/bin/python experiments/qwen35_4b_universal_replay_anchor/scripts/run.py --smoke
```

Expensive stages are explicit and fail closed rather than overwriting prior artifacts:

```bash
.venv/bin/python experiments/qwen35_4b_universal_replay_anchor/scripts/run.py --stage train-candidate
.venv/bin/python experiments/qwen35_4b_universal_replay_anchor/scripts/run.py --stage local
.venv/bin/python experiments/qwen35_4b_universal_replay_anchor/scripts/run.py --stage train-control
.venv/bin/python experiments/qwen35_4b_universal_replay_anchor/scripts/run.py --stage merge
.venv/bin/python experiments/qwen35_4b_universal_replay_anchor/scripts/run.py --stage benchmark
```

The local screen is frozen at seed 88,003. A passing candidate receives one aggregate-
only Menagerie quick@1,024 event at seed 78,133 through the trusted vLLM gateway. The
pilot requires strict improvement over base on all ten public families, aggregate at
least as high as `blend`, and aggregate higher than the matched-step replay control.

## Results

`warm_union` consumed all 1,520 rows with zero skips, completed 190 steps in 1,320.4
seconds, and ended at finite loss 0.7727. Adapter weights SHA-256:
`26837fade89e75ffd6cc5922b0dec4a1361e93d98228c9137ca59240e1a18f18`.

On frozen local seed 88,003 it passed every promotion gate: accuracy 0.7308, parse rate
0.9615, one cap contact, and zero feasible-route abstentions. Induction and state carry
were each 0/2, so this is an installability pass rather than broad-transfer evidence.
The pass authorized the matched `replay_refresh` control. It consumed all 1,520 rows
with zero skips, completed 190 steps in 1,342.0 seconds, and ended at finite loss
0.4365. Its adapter SHA-256 is
`c296c774d20403c7de9c810bfa825dbbe22bd0683c37692a929bd2bb13e3d36a`.
Both adapters were explicitly merged: the candidate merged weight SHA-256 is
`29baf3ad182e900d01186058d795004b984a5f436958d8cca5e0ebcf199422f6`,
and the replay-control merged weight SHA-256 is
`22c61cebd6091d0b8380e2d7318b4d4db99ef24eb30b942157e44f37d26cbc9e`.
The frozen aggregate-only event is now authorized but not yet consumed.

## Artifacts

- `data/dose_manifest.json`: deterministic source lineage and dose hashes.
- `data/dose_token_receipt.json`: exact token exposure and zero-skip proof.
- `scripts/materialize_doses.py`: nested stratified dose construction.
- `scripts/train_trial.py`, `eval_curriculum.py`, `merge_trial.py`, and
  `run_benchmark.py`: fail-closed staged harnesses copied into this result-bearing
  successor.
- `reports/artifact_manifest.yaml`: external adapter and merged-model locations.

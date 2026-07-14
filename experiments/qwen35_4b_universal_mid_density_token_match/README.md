# Mid-Density Token-Matched Universal Curriculum

**Status:** in-progress · since 2026-07-13 · all three exact-token arms trained; fresh local gates and conditional paired pilot remain

## Research program

- Program: `agentic_breadth_installation`
- Parent: `qwen35_4b_universal_low_density_token_match`
- Prior anchors: `qwen35_4b_universal_curriculum`,
  `qwen35_4b_universal_replay_anchor`, and the authenticated replay-refresh policy

## Question

Can a representative 160- or 240-row truth-audited designed dose cross the fresh
local installation gate from the strong replay-refresh anchor without returning to
the parent 400-row density that failed broad retention?

## Hypothesis

The 80-row arm improved parseability and cap behavior directionally but was below the
local threshold, while the earlier 400-row arm passed locally from a weaker start and
then displaced broad capability. A mid-density dose should cross the installation
threshold while leaving more replay mass than the 400-row mixture. If neither 160 nor
240 rows passes, representative dose interpolation alone is not the missing mechanism.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e...`.
- Start: authenticated `replay_refresh` adapter from the replay-anchor experiment.
- Arms: `replay_repeat` (0 designed), `designed160`, and `designed240`.
- Shared dose: 1,280 identical replay rows in identical shuffled slots.
- Replacements: three disjoint 80-row designed blocks, each covering all 13 skills,
  and three disjoint replay blocks with exactly matching forward-token sums.
- Feasibility boundary: a representative 320-row arm was rejected before freeze
  because 320 proportional designed rows were shorter than the 320 shortest replay
  rows; exact matching would have required an unregistered length-biased curriculum.
- Compute: every arm has 1,520 rows, 190 effective-batch-8 steps, and exactly
  1,405,510 forward tokens with zero tokenizer skips.
- Training: one epoch, learning rate `1e-5`, rank 32 / alpha 64, `w_think=0.2`,
  seed 43, and max length 4,096.
- Local screen: fresh procedural seed 88,005; every arm is gated independently at
  accuracy ≥0.65, parse ≥0.90, cap contacts ≤2, and no repeated feasible-route
  abstention.
- Conditional paired pilot: aggregate-only quick@1,024 seed 78,135 on one explicitly
  merged vLLM backend.
- Controls: base, `blend`, inherited replay-refresh anchor, and a new exact-token
  replay continuation. Both designed doses are prospectively registered.
- Hidden-label boundary: invoke only the trusted aggregate gateway. Never read or
  import benchmark items, sources, transcripts, or private output.

## Run

Smoke and staged run:

```bash
.venv/bin/python experiments/qwen35_4b_universal_mid_density_token_match/scripts/run.py --smoke
```

```bash
.venv/bin/python experiments/qwen35_4b_universal_mid_density_token_match/scripts/run.py --stage train-control
.venv/bin/python experiments/qwen35_4b_universal_mid_density_token_match/scripts/run.py --stage train-d160
.venv/bin/python experiments/qwen35_4b_universal_mid_density_token_match/scripts/run.py --stage train-d240
.venv/bin/python experiments/qwen35_4b_universal_mid_density_token_match/scripts/run.py --stage local
.venv/bin/python experiments/qwen35_4b_universal_mid_density_token_match/scripts/run.py --stage merge
.venv/bin/python experiments/qwen35_4b_universal_mid_density_token_match/scripts/run.py --stage benchmark
```

## Results

Training checkpoint only. All three arms completed all 190 updates over their
authenticated 1,520-row, 1,405,510-forward-token streams with zero skips. Final
training losses were 0.4199 (`replay_repeat`), 0.6606 (`designed160`), and 0.7284
(`designed240`). The local screen and new benchmark event have not run, so there is
no transfer result yet.

## Interpretation

No result interpretation before the frozen stages complete. Local success is only a
mechanism gate and will not be described as generalized transfer.

## Knowledgebase Update

- Program evidence: pending result.
- Program backlog: this mid-density bridge is active.
- Claim ledger: unchanged; a single pilot cannot establish a universal-feature claim.

## Artifacts

- `data/source_token_lengths.json`: tokenizer-frozen source-row lengths.
- `data/dose_manifest.json`: exact representative selection, slot, and hash receipt.
- `data/dose_token_receipt.json`: zero-skip proof and exact arm exposure.
- `scripts/materialize_doses.py`: deterministic nested exact-token constructor.
- `reports/design_review.md` and `reports/preregistration.md`: frozen threats and gates.
- `reports/artifact_manifest.yaml`: external adapter and merged-model locations.

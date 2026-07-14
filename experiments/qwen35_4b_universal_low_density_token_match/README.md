# Low-Density Token-Matched Universal Curriculum

**Status:** in-progress · since 2026-07-13 · all three arms trained; local gates and paired pilot remain

## Research program

- Program: `agentic_breadth_installation`
- Parent: `qwen35_4b_universal_replay_anchor`
- Prior anchors: C50, C53, `qwen35_4b_universal_curriculum`, and the authenticated
  replay-refresh policy

## Question

Can an order-of-magnitude lower density of truth-audited designed procedures add
transfer to the replay-refreshed policy without giving back its broad gains, once the
designed and replay-only arms are matched exactly on rows, optimizer steps, slot order,
and forward-token exposure?

## Hypothesis

The preceding candidate replaced 400 replay rows (26.3% of its dose) and learned the
local procedures but lost to replay alone. At 40 or 80 designed rows (2.6% or 5.3%),
the broad replay policy should remain dominant while the abstract procedures act as a
small regularizing increment. If designed content is useful at all, one of these doses
should beat an exact-token replay continuation; if neither does, the prior gap was not
merely a compute mismatch or an excessive designed fraction.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e...`.
- Start: authenticated `replay_refresh` adapter from the parent.
- Arms: `replay_repeat` (0 designed), `designed40`, and `designed80`.
- Shared dose: 1,440 identical replay rows in identical shuffled slots.
- Replacements: two 40-row, all-skill designed halves and two replay blocks whose
  token sums match those halves exactly.
- Compute: every arm has 1,520 rows, 190 effective-batch-8 steps, and exactly
  1,429,053 forward tokens.
- Training: one epoch, learning rate `1e-5`, rank 32 / alpha 64, `w_think=0.2`,
  seed 43, max length 4,096, zero skips.
- Local screen: fresh synthetic seed 88,004; every arm is gated independently.
- Paired pilot: aggregate-only quick@1,024 seed 78,134 on one merged vLLM backend.
- Controls: base, `blend`, inherited replay-refresh anchor, and exact-token replay
  continuation. Both designed doses are prospectively registered, not adaptively
  chosen from benchmark results.
- Hidden-label boundary: invoke only the trusted aggregate gateway. Never read or
  import benchmark items, sources, transcripts, or private output.

## Run

Smoke:

```bash
.venv/bin/python experiments/qwen35_4b_universal_low_density_token_match/scripts/run.py --smoke
```

Full:

```bash
.venv/bin/python experiments/qwen35_4b_universal_low_density_token_match/scripts/run.py --stage train-control
.venv/bin/python experiments/qwen35_4b_universal_low_density_token_match/scripts/run.py --stage train-d40
.venv/bin/python experiments/qwen35_4b_universal_low_density_token_match/scripts/run.py --stage train-d80
.venv/bin/python experiments/qwen35_4b_universal_low_density_token_match/scripts/run.py --stage local
.venv/bin/python experiments/qwen35_4b_universal_low_density_token_match/scripts/run.py --stage merge
.venv/bin/python experiments/qwen35_4b_universal_low_density_token_match/scripts/run.py --stage benchmark
```

## Results

Training checkpoint only. `replay_repeat` completed all 190 updates over the
authenticated 1,520-row, 1,429,053-forward-token stream with zero skips. Its final
training loss was 0.4069 and its adapter weights hash is
`bb4f0f8d35ce51e59fb06e8fc835ef043ac8960a5c178e6a511ec75c0a622a07`.
`designed40` then completed the same 190 updates and exact exposure with zero skips;
its final training loss was 0.5128 and its adapter weights hash is
`b4ca4c0187797f57ae3259f7de1817be34aad927583c0a8728786c56b40ac4a9`.
`designed80` also completed the exact update and exposure contract with zero skips;
its final training loss was 0.5864 and its adapter weights hash is
`ba82457d127c63662b5b86b4a2e1d94ed18014651b59aefd6512690eef1dabc4`.
The local screen and new benchmark event have not run, so there is no transfer
result yet.

## Interpretation

No result interpretation before the frozen stages complete.

## Knowledgebase Update

- Program evidence: pending result.
- Program backlog: this exact-token ladder is active.
- Claim ledger: unchanged; a single pilot cannot create a universal-feature claim.

## Artifacts

- `data/source_token_lengths.json`: tokenizer-frozen per-source row lengths.
- `data/dose_manifest.json`: exact nested selection and hashes.
- `data/dose_token_receipt.json`: zero-skip proof and exact arm exposures.
- `scripts/materialize_doses.py`: deterministic nested token-matched constructor.
- `reports/design_review.md` and `reports/preregistration.md`: frozen review and gates.
- `reports/artifact_manifest.yaml`: external adapter and merged-model locations.

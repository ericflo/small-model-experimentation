# Low-Density Token-Matched Universal Curriculum

**Status:** finished

**Outcome:** completed negative on 2026-07-13; all arms failed the fresh local
gate and the benchmark remained sealed.

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

All three arms completed the exact 190-update, 1,429,053-forward-token contract with
zero skips. Fresh local seed 88,004 then produced:

| Model | Accuracy | Parse rate | Cap contacts | Gate |
| --- | ---: | ---: | ---: | --- |
| inherited `replay_refresh` anchor | 0.538 | 0.577 | 11 | diagnostic only |
| `replay_repeat` | 0.500 | 0.538 | 13 | fail |
| `designed40` | 0.500 | 0.538 | 12 | fail |
| `designed80` | 0.538 | 0.615 | 10 | fail |

Every candidate missed the frozen accuracy ≥0.65, parse ≥0.90, and cap-contact
≤2 gates; all passed the feasible-route abstention check. The 80-row dose was the
least degenerate candidate, but it only tied the inherited anchor's accuracy and
remained far from deployable parsing. No arm became eligible, so the merge stage and
aggregate-only seed 78,134 event did not run.

## Interpretation

Exact token matching closes the prior compute-exposure ambiguity for these doses:
replacing 40 or 80 of 1,520 replay rows was insufficient to install robust concise
execution into the replay-refreshed policy. The 80-row trend in parse rate and cap
contacts is directional, not a pass and not evidence of broad transfer. The result
does not say whether an intermediate dose, a termination-focused loss, or a
lower-collateral integration mechanism can cross the local threshold while preserving
the replay anchor; each requires a new result-separated experiment.

## Knowledgebase Update

- Program evidence: records the exact-token low-density local negative.
- Program backlog: closes this ladder and requires a new result-separated mechanism.
- Shared synthesis: low density alone does not resolve the install/retention tradeoff.
- Claim ledger: unchanged; no benchmark event ran and no universal-feature claim exists.

## Artifacts

- `data/source_token_lengths.json`: tokenizer-frozen per-source row lengths.
- `data/dose_manifest.json`: exact nested selection and hashes.
- `data/dose_token_receipt.json`: zero-skip proof and exact arm exposures.
- `scripts/materialize_doses.py`: deterministic nested token-matched constructor.
- `runs/local/seed88004.json`: complete fresh local receipt.
- `runs/local/seed88004_promotion.json`: authenticated fail-closed promotion decision.
- `reports/design_review.md` and `reports/preregistration.md`: frozen review and gates.
- `reports/artifact_manifest.yaml`: external adapter and merged-model locations.

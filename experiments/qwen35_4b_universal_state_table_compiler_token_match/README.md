# Natural-Language State-Table Universal Curriculum

**Status:** in-progress · since 2026-07-14 · CPU feasibility and adversarial design review remain

This result-separated successor tests whether truth-audited, variable-depth
natural-language state tables plus independent hypothesis scoring and a short
verified commit install a reusable reasoning procedure better than an exact-token
replay continuation from the same parent.

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can one cleanly installed procedure improve every held-out
  benchmark family rather than redistribute wins?
- Prior anchors: `qwen35_4b_universal_search_scaffold_token_match`,
  `qwen_trace_procedure_depth_stress`, `qwen_constrained_abi_parser`, C37, and C38.

## Question

Does matching the training interface to variable-depth natural-language execution
teach the model to maintain explicit state, compare independently simulated
hypotheses, and stop with a concise answer—without sacrificing broad replay behavior?

## Hypothesis

The failed predecessor often reached the correct final state but did not commit, and
it regressed hypothesis selection. A truth-audited table that records each natural-
language transition should make execution inspectable; separate rows that score each
hypothesis on every probe should preserve discrimination; an answer-only commit after
verification should train the missing emission seam. The mechanism is false if the
candidate cannot beat both its parent and an exact-token replay control on a fresh
unchanged local gate.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, pinned revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: authenticated `close_xi` adapter from
  `qwen35_4b_universal_close_weight_token_match`; the failed scaffold adapter is not
  inherited.
- Dataset/task source: fresh deterministic procedural synthesis owned by this
  experiment. No benchmark source, item, transcript, or result detail may be read.
- Candidate: variable-depth natural-language state execution, independent hypothesis
  scoring, verification/repair, and concise commit lessons.
- Mechanism-falsifying control: same-parent replay continuation with identical forward
  tokens, optimizer steps, backend, seed, and position-aligned shared replay.
- Primary admission: the unchanged absolute local capability gate plus paired wins
  over parent and active replay on target execution/probe behavior.
- Conditional broad admission: aggregate-only same-backend evaluation only after the
  sole candidate passes every local check; all reported families must improve before
  higher-tier confirmation or matched-compute sample-more.
- Reserved seeds: construction `77112`, training `46`, fresh local `88008`, and
  conditional aggregate `78138`.

## Run

Intake smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_state_table_compiler_token_match/scripts/run.py --smoke
```

Scientific stages are intentionally unavailable until CPU feasibility and
`reports/design_review.md` freeze the commands, byte identities, exact token counts,
reachable gates, and aggregate firewall.

## Results

No model result exists. No GPU generation, training, merge, local capability event,
or benchmark event has run.

## Interpretation

This intake changes the interface, not the dose or close-token weight. Its novelty is
an executable natural-language state table connected to independent hypothesis scores
and a verified answer seam. Any later gain belongs to that package unless a frozen
ablation separates the pieces.

## Knowledgebase Update

- Program evidence: unchanged until a result exists.
- Program backlog: this result-separated successor is active at intake.
- Claim ledger: unchanged.

## Artifacts

- `idea_intake.md`
- `configs/default.yaml`
- `scripts/run.py`
- `reports/report.md`
- `reports/artifact_manifest.yaml`

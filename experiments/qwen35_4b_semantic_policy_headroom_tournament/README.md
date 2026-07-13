# Semantic-policy headroom tournament

**Status:** in-progress · since 2026-07-13 · deterministic design preflight
passed; design lock and two parent-only model blocks remain.

Qualify non-saturated verifier-conditioned semantic conflicts before spending
another update on Qwen3.5-4B.

## Research program

- Program: `agentic_breadth_installation`.
- Direct predecessor: `qwen35_4b_validation_policy_counterexample_curriculum`.
- Parent: its unchanged learned transaction checkpoint, exact weight SHA-256
  `1cf5fb...41ba3`.

## Question

Which validation-policy conflicts still produce replicated failed-test headroom
in the transaction-trained model, under the exact looping coding harness, and
are therefore legitimate substrates for a later counterexample curriculum?

## Design

This experiment trains nothing and cannot invoke Menagerie. It crosses three
conflicts with three public representations:

- negative quantity: malformed `ValueError` versus ordinary insufficiency
  `False`;
- non-integer quantity: malformed `TypeError` versus ordinary insufficiency
  `False`;
- blank resource: malformed `ValueError` versus ordinary unknown resource
  `False`;
- bundle mappings, record dictionaries, and tuple sequences.

Nine inferred-contract families state the valid input domain and ordinary
rejection policy but require the agent to infer malformed behavior from visible
tests/failure output. Three explicit-contract controls state the exception
verbatim. Every partial implementation is otherwise correct and fails visible
and hidden tests only at the semantic conflict; every oracle passes.

Two content-disjoint blocks each contain 36 unique repositories (12 families ×
three tasks) and 72 controlled recovery cases. An inferred axis qualifies only
if failed-test success is 15–80% in both blocks, at least two of three shapes
are individually inside that band in each block, explicit-control success is
≥85%, and invalid/cap contacts remain ≤5%. At least one replicated axis must
qualify.

## Why this is different

The predecessor trained before proving that its rewritten substrate retained
the historical failure. Parent and control were already 48/48, so the
treatment effect was unidentifiable. Here, exact-substrate parent headroom is
the only outcome. Eligible axes and families are emitted mechanically by frozen
rules for use in a separate future experiment; no update, threshold change, or
benchmark escalation occurs here.

## Firewall and compute

All repositories are fresh procedural fixtures. Hidden tests and repair objects
stay host-side; only public issue/source/test/tool output reaches the model.
Both blocks use the same merged checkpoint, copied vLLM 0.24 runner, 512 think
+ 512 answer tokens, one greedy trajectory, and six turns. Nothing under
`benchmarks/` is read/imported, and Menagerie authorization is hard-coded false.

## Run

```bash
python experiments/qwen35_4b_semantic_policy_headroom_tournament/scripts/run.py --smoke
.venv/bin/python experiments/qwen35_4b_semantic_policy_headroom_tournament/scripts/run.py --lock-design <commit>
.venv/bin/python experiments/qwen35_4b_semantic_policy_headroom_tournament/scripts/run.py --gpu-smoke
.venv/bin/python experiments/qwen35_4b_semantic_policy_headroom_tournament/scripts/run.py --full
```

## Current evidence

CPU smoke and 19 unit tests pass. Both 36-repository blocks are internally
unique and mutually content-disjoint; all 12 initial/partial fixtures fail both
executable suites and every oracle passes. No model output exists yet.

## Artifacts

Committed design and compact receipts live here. Detailed parent trajectories
will live under `large_artifacts/qwen35_4b_semantic_policy_headroom_tournament`
per [`reports/artifact_manifest.yaml`](reports/artifact_manifest.yaml).

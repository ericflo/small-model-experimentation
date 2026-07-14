# Search-Scaffold Universal Curriculum

**Status:** in-progress · since 2026-07-14 · implementation, adversarial review, and all model work remain

This experiment tests whether independently supervised, executable search substates
compose into a bounded general reasoning procedure better than an exact-token replay
continuation from the same parent.

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can engineered synthetic curricula install substrate-general
  procedures that improve every held-out benchmark family without displacing the
  incumbent broad policy?
- Parent result: `qwen35_4b_universal_close_weight_token_match`.
- Prior anchors: C44/C59 (serial reasoning content is load-bearing), C56 (oracle
  trace narration is not a reusable induction circuit), and the exact-token
  mid-density/close-weight local negatives.

## Question

Can a staged executable curriculum that teaches the component operations of
two-step search—apply a proposed first operation, fit or reject a second operation,
execute a verified pair, then solve—cross the unchanged fresh local gate where
full narrated induction traces and heavier close loss did not?

## Hypothesis

The current curriculum jumps from primitive lessons to a full decomposition trace.
At deployment the model repeatedly explores candidates without a bounded decision
procedure. Training the intervening states as independently scored, truth-audited
subproblems should make candidate evaluation and rejection addressable features.
A fixed compact ledger in the final lessons should then compose those features and
commit within 1,024 tokens. This is a curriculum-structure intervention, not another
close-weight or generic-dose sweep.

## Setup

- Only model: `Qwen/Qwen3.5-4B`, pinned revision `851bf6e...`.
- Prospective parent: authenticated `close_xi` adapter from the completed predecessor.
- Synthetic source: new executable abstract-index tasks over disjoint randomized
  surfaces; no benchmark content, outputs, or family implementation is read.
- Intended candidate block: 80 rows, 16 each of apply-first, fit-second,
  reject-first, execute-pair, and bounded full-search lessons.
- Intended exposure: 200 common replay rows plus the 80 staged rows and 40 matched
  replay fillers; 320 rows, 286,814 forward tokens, and 40 updates if feasibility
  succeeds. A new replay-only arm must match all three quantities from the same parent.
- Training seed: 45. Fresh local seed: 88,007. Conditional aggregate seed: 78,137.
- Local gate: accuracy ≥0.65, parse ≥0.90, cap contacts ≤2, no repeated feasible-route
  abstention, plus at least one correct case in both `u_execute` and `u_induct`.
- Hidden-label boundary: local cases are fresh procedural experiment data. Benchmark
  access is conditional and aggregate-only through the trusted gateway; benchmark
  sources, items, transcripts, and private outputs remain unread.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/qwen35_4b_universal_search_scaffold_token_match/scripts/run.py --smoke
```

Full:

```bash
Not authorized yet. The full command will be frozen only after stream feasibility,
unit tests, preregistration, and adversarial design review pass.
```

## Results

No model training or evaluation has run. The current artifact is an intake and
feasibility scaffold only.

## Interpretation

None yet. A feasible stream and passed adversarial review are prerequisites for a
scientific run.

## Knowledgebase Update

- Program evidence: unchanged until a result exists.
- Program backlog: this result-separated staged-search successor is active.
- Claim ledger: unchanged.

## Artifacts

- `src/`
- `scripts/`
- `configs/`
- `data/`
- `runs/`
- `analysis/`
- `reports/`
- `reports/artifact_manifest.yaml`
- `idea_intake.md`

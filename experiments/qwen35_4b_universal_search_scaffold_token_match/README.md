# Search-Scaffold Universal Curriculum

**Status:** in-progress · since 2026-07-14 · both arms trained; fresh local gate pending

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
- Parent: authenticated `close_xi` adapter from the completed predecessor (weights
  `16e9dc75...c179`, config `de953bd5...7ff`).
- Synthetic source: new executable abstract-index tasks over disjoint randomized
  surfaces; no benchmark content, outputs, or family implementation is read.
- Candidate block: 80 rows, 16 each of apply-first, fit-second,
  reject-first, execute-pair, and bounded full-search lessons.
- Frozen exposure: 200 common replay rows plus the 80 staged rows and 40 matched
  replay fillers. The replay arm uses the same 200 rows plus 120 replay rows. Both
  arms have 320 rows, exactly 286,814 forward tokens, zero skips, and 40 updates;
  exactly 200 shuffled positions are byte-identical.
- Training seed: 45. Fresh local seed: 88,007. Conditional aggregate seed: 78,137.
- Local gate: accuracy ≥0.65, parse ≥0.90, cap contacts ≤2, no repeated feasible-route
  abstention, plus accuracy ≥0.50 (at least one of two) in both `u_execute` and
  `u_induct`.
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
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/qwen35_4b_universal_search_scaffold_token_match/scripts/run.py \
  --stage train-control
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/qwen35_4b_universal_search_scaffold_token_match/scripts/run.py \
  --stage train-candidate
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/qwen35_4b_universal_search_scaffold_token_match/scripts/run.py \
  --stage local
```

The local stage writes a promotion receipt even on failure. Merge and benchmark
remain sealed unless the sole candidate passes; then run `--stage merge` and
`--stage benchmark`. Each natural stage is committed, rebased, fully checked,
pushed to `main`, and verified in both GitHub workflows before the next starts.

## Results

CPU feasibility passed. The deterministic source has 80 truth-audited rows over six
surface families. The two frozen streams each contain 320 trainable rows and exactly
286,814 forward tokens at max length 4,096, with zero skips. All 43 experiment tests
and the staged smoke harness pass. The replay control has now trained from the
authenticated parent on all 320 rows with zero skips and 40/40 updates. Its final
loss is 0.4215; adapter weights/config SHA-256 are `10155232...fc538` /
`373c1426...ac9b`.
The scaffold candidate then trained independently on all 320 rows with zero skips and
40/40 updates. Its final loss is 1.492; adapter weights/config SHA-256 are
`e7957d90...84618` / `22859c76...2c4ce`. Losses are not compared across the different
target distributions. No local capability evaluation, merge, or benchmark event has
run.

## Interpretation

The design is feasible and passed adversarial review after explicitly narrowing the
claim: the full lesson demonstrates one rejected and one successful branch, not
exhaustive enumeration of the operation universe. Any gain is evidence for the
five-stage scaffold package, not proof that a general search algorithm was learned.

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

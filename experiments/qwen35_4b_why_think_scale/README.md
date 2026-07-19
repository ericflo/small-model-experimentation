# Qwen35 4B WHY-Think Scale

**Status:** finished · 2026-07-19 · NEGATIVE for the dual-channel design + POSITIVE method finding (claim C60). The synthetic `<think>` scale ladder COLLAPSES coding thinking-on (rung 2k HE −18/MBPP −15, rung 5k HE −26/MBPP −32; lower loss = worse). A 2×2 ablation isolates synthetic-think SUPERVISION as the dominant damage. Rejection-sampled NATIVE think (self-sampled, execution-verified, 3000 problems) RETAINS coding (HE +1) — same recipe+weight, synthetic→native think = +35 HumanEval. Corrected baseline puts HumanEval at 89.6% (near ceiling): the SFT-push-function-writing goal is closed; prize is agentic. See `reports/report.md` §Results.

## Research Program

- Program: `agentic_breadth_installation` (cognitive-core coding sub-program)
- Program question: can real, transferable coding capability be INSTALLED into base
  `Qwen/Qwen3.5-4B` by designed, contamination-free curricula, proven by transfer —
  and specifically, does teaching the 4B WHY a correct answer is correct SCALE to a
  peak worth an RLVR foundation, WITHOUT destroying the model's native thinking?
- Prior anchors: `experiments/qwen35_4b_coding_fitness_harness` (the shared eval,
  being fixed to thinking-on + 8192 budget). Bet #4
  (`qwen35_4b_why_comment_install`) gave the program's biggest fast gain — HumanEval
  +5 on the clean, comment-inert test — but was underpowered and did not stack.
  `qwen35_4b_why_scale_ladder` scaled that WHY curriculum but with a MINIMAL think
  block.

## Question

Qwen3.5-4B is a THINKING model whose coding performance depends on its `<think>`
trace (the repo's most-replicated finding). The prior WHY curriculum put reasoning
in inline `#WHY:` comments and left `<think>` minimal — which risks DESTROYING the
model's native thinking. Does a CORRECTED **dual-channel** WHY curriculum — a
genuine step-by-step derivation IN the `<think>` block AND the strippable `#WHY:`
comments — SCALE (climb to a peak, stay flat, or collapse) on HumanEval/MBPP
measured THINKING-ON, and become the SFT foundation for the RLVR phase?

## Hypothesis

Because the 4B's coding depends on its think trace, training WITH a rich, true
`<think>` derivation (not an empty one) should preserve and shape thinking while
teaching WHY, so the WHY signal scales at least as well as the comment-only version
without the empty-think retention risk. Each row teaches: think richly (derive the
solution and verify it with a real worked example) in the native channel, then emit
clean-but-`#WHY:`-annotated code. Because comments are inert to the execution
grader, any pass@1 gain is an unconfounded CODE improvement. Honest prior: this is a
foundation-building bet; a flat curve remains a likely, informative outcome.

## Setup

- Model: only `Qwen/Qwen3.5-4B` (rev `851bf6e8…`), measured and trained WITH
  thinking on; one fresh r32/a64 QLoRA adapter per rung from the `base_reserialized`
  composite in a single stage.
- Dataset/task source: `scripts/gen_why_think_curriculum.py` (construction seed
  95200) — 59 parameterized synthetic families producing dual-channel rows
  (`messages` = plain "write a function" prompt with spec + signature + public
  asserts; `think` = a GENUINE derivation emitted BY CONSTRUCTION, NO teacher model,
  with a real worked-example trace; `answer` = clean correct code with strippable
  `#WHY:` comments). Every row verified by real execution. Rungs at
  2000/5000/10000/20000/40000 rows sha-pinned in `data/ladder_manifest.json`; the
  corpora are large, deterministically regenerable, gitignored under
  `large_artifacts/`.
- Train/eval split: training is `spec -> dual-channel (think + #WHY code)` over
  synthetic families; evaluation is the held-out `spec -> code` HumanEval (164) +
  MBPP (200) — disjoint surfaces, grader ignores comments, measured thinking-on.
- Baseline: base `Qwen/Qwen3.5-4B`, CO-MEASURED thinking-on on the same shared
  harness per rung (no hardcoded thinking-off anchor).
- Controls: contamination firewall (banned-name audit zero hits over prompt +
  think + answer; distinctive code 7-gram overlap zero at scale); comments inert to
  the grader; the think derivation byte-verified against real execution; the base
  composite authenticated fail-closed before training.
- Primary metric: greedy pass@1 on HumanEval + MBPP (shared harness, thinking-on),
  swept per rung.
- Oracle-only metrics: none gate here; the agentic duet-eval is a follow-on confirm
  on the peak composite.
- Hidden-label boundary: benchmarks are executed, never read as data.

## Run

Smoke (no GPU, no writes) — compiles, verifies base provenance + fixture, builds a
small verified dual-channel corpus, checks the ladder-manifest shas, runs the tests:

```bash
python scripts/run.py --smoke
```

Build the ladder corpora locally from the committed manifest (CPU, model-free):

```bash
python scripts/run.py --stage gen-ladder
```

GPU stages (each gated behind a staged adversarial review; per rung; see
`reports/preregistration.md` for the exact commands and checkpoint order):

```bash
python scripts/run.py --stage train   --rows 2000    # r32/a64, epochs 1, seed 95201
python scripts/run.py --stage merge   --rows 2000    # vendored composite merger
python scripts/run.py --stage measure --rows 2000    # shared HumanEval+MBPP harness, thinking-on
# ... repeat --rows 5000 / 10000 / 20000 / 40000 and assemble the curve.
```

## Results

Not yet run. The train/merge/measure stages are gated behind staged reviews and are
a SWEEP the orchestrator runs rung-by-rung. When run,
`runs/measure/rung_<rows>.json` records each rung's four pass@1 numbers (base co-
measured thinking-on), the paired McNemar deltas, and the rung-vs-base problem
deltas; the assembled curve pass@1(rows) locates the peak.

Dual-channel construction facts already established (see `reports/report.md`):

- 59 families / 13 categories; 5000-row sample: 59/59 families, 1196 distinct `#WHY:`
  templates, **4997/5000 distinct think skeletons**, 100% unique programs; 20000-row
  build 100% unique.
- Token budget: real pinned tokenizer full render max 739 tokens (p95 619, median
  467) over 5000 rows — 0 over the 4096 cap.
- Contamination through 10000 rows: 663 banned names after whitelist, 0 whole-word
  hits over prompt + think + answer; 0 distinctive shared 7-grams (78 structural
  idioms).
- Unit tests green; `run.py --smoke` green; boundary drills refuse.

## Interpretation

Pending the sweep. A rising-then-peaking curve makes the peak rung the SFT
foundation for the RLVR phase (Phase B) and shows the dual-channel design scales WHY
without the empty-think retention risk. A flat curve reprices the WHY mechanism on
this surface. A collapse bounds the usable dose. The design goal is to preserve the
model's native thinking WHILE teaching WHY — the retention read (thinking-on
pass@1 not dropping) is itself a first-order result.

## Knowledgebase Update

- Program evidence updated: pending the ladder sweep.
- Program backlog updated: pending the ladder sweep.
- Claim ledger updated: pending the ladder sweep (design-only work manufactures no
  claim).

## Artifacts

- `scripts/gen_why_think_curriculum.py` — the dual-channel, scale-capable WHY-think
  generator (59 families, phrase-pool rationales, per-row truth audit + byte-verified
  worked-example think trace, contamination self-heal, deterministic per seed 95200).
- `scripts/build_ladder.py` — builds the five rung corpora + the sha-pinned
  `data/ladder_manifest.json`; `--verify` regenerates and checks shas.
- `scripts/contamination.py` + `data/contamination/banned_function_names.json`.
- `scripts/train_trial.py` (fail-closed per-rung trainer), vendored
  `scripts/train_think.py` + `scripts/merge_adapter.py`, `scripts/measure_transfer.py`
  (per-rung sweep via the shared harness, thinking-on), `scripts/run.py`.
- `data/ladder_manifest.json`, `data/provenance/base_reserialized.json`.
- `configs/`, `reports/` (preregistration, report, artifact manifest), `tests/`.
- `reports/artifact_manifest.yaml`.

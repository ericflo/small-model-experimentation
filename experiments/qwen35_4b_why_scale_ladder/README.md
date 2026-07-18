# Qwen35 4B WHY Scale Ladder

**Status:** in-progress · since 2026-07-18 · model-free construction frozen (generator + sha-pinned ladder + harness + tests); GPU rungs (train/merge/measure) gated behind staged reviews and not yet run.

## Research Program

- Program: `agentic_breadth_installation` (cognitive-core coding sub-program)
- Program question: can real, transferable coding capability be INSTALLED into base
  `Qwen/Qwen3.5-4B` by designed, contamination-free curricula, proven by transfer —
  and specifically, does teaching the 4B WHY a correct answer is correct SCALE with
  more (genuinely diverse) data to a peak worth building an RLVR foundation on?
- Prior anchors: `experiments/qwen35_4b_coding_fitness_harness` — HumanEval 76.2%
  (0.7622), MBPP 56.5% (0.565), agentic duet-eval 8/35. Bet #4
  (`qwen35_4b_why_comment_install`) gave the program's biggest fast gain — HumanEval
  +5 (0.7622 -> 0.7927) on the clean, comment-inert test — but was underpowered
  (McNemar p=0.33), flat on the agentic loop, and did not survive combination.

## Question

Does teaching the 4B WHY each line of correct code is correct — as inline `#WHY:`
comments the grader ignores — SCALE? Trained at genuinely diverse rung sizes
(2000, 5000, 10000, 20000), does HumanEval/MBPP pass@1 CLIMB to a peak (WHY is a
real but underpowered signal), stay FLAT (the +5 was noise), or COLLAPSE (the
narrow synthetic surface overfits)? The peak rung is the SFT foundation for the
subsequent RLVR phase.

## Hypothesis

The 504-row WHY gain was the low-N corner of a rising curve, not noise. With a
generator that produces GENUINELY diverse data at scale (>= 50 families, >= 300
distinct WHY reasoning patterns, ~100% unique programs — removing the saturation
that would have faked a null), pass@1 should climb with real scale to a peak
before overfit/collapse. Because comments are inert to the execution grader, any
gain is an unconfounded CODE improvement. Honest prior: P(a rung meaningfully beats
base, >= +3 problems with retention) ~= 0.45; a flat curve remains a likely,
informative outcome that reprices the WHY family.

## Setup

- Model: only `Qwen/Qwen3.5-4B` (rev `851bf6e8…`); one fresh r32/a64 QLoRA adapter
  per rung, trained from the `base_reserialized` composite in a single stage.
- Dataset/task source: `scripts/gen_why_scale_curriculum.py` (construction seed
  94100) — 59 parameterized synthetic families producing `spec -> correct
  solution` rows where each meaningful line carries a trailing `#WHY:` causal
  comment, emitted BY CONSTRUCTION (NO teacher model), every row verified by real
  execution. Rungs at 2000/5000/10000/20000 rows are sha-pinned in
  `data/ladder_manifest.json`; the corpora are large, deterministically
  regenerable, and live gitignored under `large_artifacts/`.
- Train/eval split: training is `spec -> #WHY:-commented correct code` over
  synthetic families; evaluation is the held-out `spec -> code` HumanEval (164) +
  MBPP (200) — disjoint surfaces, grader ignores comments.
- Baseline: base `Qwen/Qwen3.5-4B`, co-measured on the same shared harness per rung
  (must beat it, not just move — "sample more" is the standing bar).
- Controls: contamination firewall (whole-word banned-name audit zero hits;
  distinctive code 7-gram overlap zero at scale); comments inert to the grader; the
  base composite authenticated fail-closed before training.
- Primary metric: greedy pass@1 on HumanEval + MBPP (shared fitness harness,
  `experiments/qwen35_4b_coding_fitness_harness/scripts/eval_pass1.py`, referenced
  not copied), swept per rung.
- Oracle-only metrics: none gate here; the agentic duet-eval is a follow-on confirm
  on the peak composite.
- Hidden-label boundary: benchmarks are executed, never read as data; the ladder
  curve is read once per rung from the four pass@1 numbers.

## Run

Smoke (no GPU, no writes) — compiles, verifies the base provenance + fixture,
builds a small verified corpus, checks the ladder-manifest shas, runs the tests:

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
python scripts/run.py --stage train   --rows 2000    # r32/a64, epochs 4, seed 94101
python scripts/run.py --stage merge   --rows 2000    # vendored composite merger
python scripts/run.py --stage measure --rows 2000    # shared HumanEval+MBPP harness
# ... repeat --rows 5000 / 10000 / 20000 (epochs 2 / 1 / 1) and assemble the curve.
```

## Results

Not yet run. The install/merge/measure stages are gated behind staged reviews and
are a SWEEP the orchestrator runs rung-by-rung. When run,
`runs/measure/rung_<rows>.json` records each rung's four pass@1 numbers, the paired
McNemar deltas, and the rung-vs-base problem deltas; the assembled curve
pass@1(rows) locates the peak. Separate deployable evidence (a pass@1 gain over
base, a real code improvement because the grader ignores comments) from the
retention guard.

Model-free construction facts already established (see `reports/report.md`):

- 59 families / 13 categories; 5000-row sample: 59/59 families, 1196 distinct
  normalized WHY templates, 100% unique programs; 20000-row build 100% unique
  (raw draw ~82%).
- Contamination: 663 banned benchmark names after whitelist, 0 whole-word hits; 0
  distinctive shared 7-grams (78 shared structural idioms) at 5000 rows.
- Token budget: full training render max 499 tokens (median 337) over 5000 rows —
  well under the 4096 cap; 0 rows truncate.
- 52 unit tests green; `run.py --smoke` green; boundary drills refuse.

## Interpretation

Pending the sweep. A rising-then-peaking curve makes the peak rung the SFT
foundation for the RLVR phase (Phase B) and funds the agentic confirm. A flat curve
reprices the WHY-comment mechanism (does not scale on this surface) and takes RLVR
from base instead. A collapse bounds the usable WHY dose and makes the small rung
the foundation.

## Knowledgebase Update

- Program evidence updated: pending the ladder sweep.
- Program backlog updated: pending the ladder sweep.
- Claim ledger updated: pending the ladder sweep (design-only work manufactures no
  claim).

## Artifacts

- `scripts/gen_why_scale_curriculum.py` — the scale-capable, high-diversity WHY
  curriculum generator (59 families, phrase-pool rationales, per-row truth audit,
  contamination self-heal, deterministic per seed).
- `scripts/build_ladder.py` — builds the four rung corpora + the sha-pinned
  `data/ladder_manifest.json`; `--verify` regenerates and checks shas.
- `scripts/contamination.py` + `data/contamination/banned_function_names.json`.
- `scripts/train_trial.py` (fail-closed per-rung trainer), vendored
  `scripts/train_think.py` + `scripts/merge_adapter.py`, `scripts/measure_transfer.py`
  (per-rung sweep via the shared harness), `scripts/run.py`.
- `data/ladder_manifest.json`, `data/provenance/base_reserialized.json`.
- `configs/`, `reports/` (preregistration, report, artifact manifest), `tests/`.

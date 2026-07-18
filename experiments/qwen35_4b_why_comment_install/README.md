# Qwen35 4B WHY-Comment Install

**Status:** finished · 2026-07-18 · TARGET-SPECIFIC WEAK POSITIVE — HumanEval +5 (biggest fast gain, clean inert-to-grading test) but agentic flat (8/35); complementary to self_repair; not individually significant. Retrained 4 epochs (epoch-1 undertrained).

## Research Program

- Program: `agentic_breadth_installation` (cognitive-core coding sub-program)
- Program question: can real, transferable coding capability be INSTALLED into
  base Qwen/Qwen3.5-4B by designed, contamination-free curricula, proven by
  transfer to held-out coding — and specifically, does teaching the 4B WHY a
  correct answer is correct (its causal/generating structure) install what
  teaching a passive skill (bet #1, NULL) and a loop behavior (bet #2, weak
  positive) did?
- Prior anchors: base coding baselines from
  `experiments/qwen35_4b_coding_fitness_harness` — HumanEval 76.2% (strong
  function coder), MBPP 56.5%, agentic duet-eval 23% (8/35, weak agent). Bet #1
  (`qwen35_4b_exec_trace_install`, execution-tracing) was NULL; bet #2
  (`qwen35_4b_self_repair_install`, self-repair) was a WEAK POSITIVE (HumanEval
  +3, agentic 8/35 -> 10/35, 3-vs-1 discordant), teaching that loop/behavior
  curricula beat passive-skill ones.

## Question

Does teaching the 4B WHY a correct answer is correct — by training on `spec ->
correct-solution` rows where EACH meaningful line of the correct code carries a
trailing `#WHY:` comment giving the true causal reason that line is correct —
TRANSFER to real coding (HumanEval + MBPP pass@1, comments IGNORED by the
grader), and (the primary target) move the agentic duet-eval, without regressing
function-writing?

## Hypothesis

The mechanism most likely to escape `install != convert` is teaching the
GENERATING REASON of a correct answer, not just the answer. The inline-comment
variant is the CLEANEST test: code comments are INERT to execution grading, so
the WHY hypothesis is testable DIRECTLY with ZERO annealing — train the model to
write richly-`#WHY:`-commented code, eval pass@1 with comments ignored; if the
CODE improved, the WHY-annotation worked, unconfounded. The training prompt is a
plain `spec -> function` framing with NO instruction to comment, so the
WHY-writing behavior is the model's DEFAULT and fires on a plain eval prompt.
Honest prior on a MEANINGFUL install (>= 3-problem gain with retention): ~35%;
NULL remains a likely verdict and would fund the think-block WHY variant (bet #3).

## Setup

- Model: only `Qwen/Qwen3.5-4B` (rev `851bf6e8…`); ONE fresh r32/a64 QLoRA
  adapter trained from the `base_reserialized` composite in a single stage.
- Dataset/task source: `data/sft_why_comment.jsonl` — 504 `spec -> correct
  solution` rows built by `scripts/gen_why_comment_curriculum.py` (construction
  seed 92450). Every row is verified by real execution (strip the `#WHY:`
  comments and the clean code passes all asserts; the commented code runs
  identically; the marker is strippable; every comment is line-specific and
  non-boilerplate). The task and solution are built BY CONSTRUCTION and the
  `#WHY:` text is emitted mechanically — NO teacher model.
- Train/eval split: training is `spec -> #WHY:-commented correct code`;
  evaluation is the held-out `spec -> code` HumanEval (164) + MBPP (200) —
  deliberately disjoint surfaces, with the grader ignoring comments. The agentic
  duet-eval is the primary real target, run manually as a follow-on.
- Baseline: base Qwen/Qwen3.5-4B on the same shared harness.
- Controls: contamination firewall (whole-word banned-benchmark-name audit, zero
  hits; distinctive code n-gram overlap, zero); comments inert to the grader; the
  base composite authenticated fail-closed (tree + weights) before training.
- Primary metric: greedy pass@1 on HumanEval + MBPP (shared fitness harness,
  `experiments/qwen35_4b_coding_fitness_harness/scripts/eval_pass1.py`,
  referenced not copied).
- Oracle-only metrics: none gate here; the agentic duet-eval is a follow-on
  confirm, not part of this cell's frozen consequence.
- Hidden-label boundary: the frozen, TIGHTENED two-directional consequence
  (INSTALLED_CODING requires a >= 3-problem gain / RETENTION_FAIL / NULL) is read
  once from the four pass@1 numbers; benchmarks are executed, never read as data.

## Run

Smoke (no GPU, no writes):

```bash
python scripts/run.py --smoke
```

GPU stages (each gated behind a staged adversarial review; see
`reports/preregistration.md` for the exact commands and checkpoint order):

```bash
python scripts/run.py --stage train     # r32/a64, 1 epoch, seed 92451
python scripts/run.py --stage merge     # vendored composite merger
python scripts/run.py --stage measure   # shared HumanEval+MBPP harness, both arms
```

Agentic confirm (manual follow-on on the merged composite — the PRIMARY real
target, base 8/35): run the duet-eval gen4 harness with `--model-override` set to
`large_artifacts/qwen35_4b_why_comment_install/merged/why_comment`, exactly as
bet #2's measure review documented for the self_repair composite.

## Results

Not yet run. The install/merge/measure stages are gated behind staged reviews.
When run, `runs/measure/transfer_summary.json` records all four pass@1 numbers
(base/treatment x HE/MBPP, counts + fractions), the per-problem paired deltas,
and the frozen, tightened verdict. Separate deployable evidence (transfer, which
is a real code improvement because the grader ignores comments) from the
retention guard.

## Interpretation

Pending the sealed measurement. INSTALLED_CODING makes why_comment the program's
reference composite and funds the agentic confirm plus the comment-strip anneal;
RETENTION_FAIL realizes the commented-target generation shift; NULL means teaching
WHY inline reshuffles-without-raising like the passive-skill bets and funds the
pre-committed think-block WHY variant (bet #3), not a re-roll.

## Knowledgebase Update

- Program evidence updated: pending measurement.
- Program backlog updated: pending measurement.
- Claim ledger updated: pending measurement.

## Artifacts

- `scripts/` — WHY-comment curriculum generator (construction + safe execution +
  inline-rationale audit), contamination module, vendored trainer/merger,
  fail-closed `train_trial.py`, `measure_transfer.py`, `run.py`.
- `data/sft_why_comment.jsonl` — the 504-row curriculum + `curriculum_receipt.json`;
  `data/provenance/base_reserialized.json`; `data/contamination/`.
- `configs/`, `reports/`, `tests/`
- `reports/artifact_manifest.yaml`

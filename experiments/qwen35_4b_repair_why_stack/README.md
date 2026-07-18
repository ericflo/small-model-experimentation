# Qwen35 4B Repair + Why Stack

**Status:** finished · 2026-07-18 · NULL (mixture dilution) - corpus-union washed out both effects (HumanEval +5->+1, agentic 10->7); correct combination is task-vector weight arithmetic (next)

## Research Program

- Program: `agentic_breadth_installation` (cognitive-core coding sub-program)
- Program question: can real, transferable coding capability be INSTALLED into
  base Qwen/Qwen3.5-4B by designed, contamination-free curricula, proven by
  transfer to held-out coding — and specifically, do the two individually
  weak-positive ingredients (self-repair loop behavior; why-comment causal
  reasoning) COMBINE to capture BOTH of their target-specific gains and clear
  significance?
- Prior anchors: base coding baselines from
  `experiments/qwen35_4b_coding_fitness_harness` — HumanEval 76.2% (125/164,
  strong function coder), MBPP 56.5% (113/200), agentic duet-eval 23% (8/35,
  weak agent). Bet #1 (`qwen35_4b_exec_trace_install`, execution-tracing) was
  NULL. Bet #2 (`qwen35_4b_self_repair_install`, self-repair) was a WEAK POSITIVE
  on the LOOP target (agentic 8/35 -> 10/35, HumanEval +3). Bet #4
  (`qwen35_4b_why_comment_install`, why-comment) was a WEAK POSITIVE on the
  FUNCTION target (HumanEval +5, agentic flat). The two help DIFFERENT targets.

## Question

If self-repair (a loop behavior) and why-comment (causal reasoning) each produced
a real but weak, target-SPECIFIC coding gain — repair on the agentic loop, why on
per-function correctness — does STACKING them (training one fresh adapter on the
UNION of the two committed curricula) capture BOTH gains (HumanEval ~+5 AND
agentic ~10/35) and cross the significance line the individual bets could not?

## Hypothesis

The two ingredients are COMPLEMENTARY, not redundant: why-comment teaches the 4B
the generating reason of correct code (function correctness), self-repair teaches
the detect-and-fix loop (multi-step agentic behavior). Because they target
disjoint failure modes and neither regressed the other's target, their effects
should ADD rather than interfere. If, after stacking, BOTH the HumanEval gain
(~+5, from WHY) and the agentic gain (~10/35, from repair) appear, the stack
works AND the individual weak signals are confirmed real. If the stack is flat,
the two weak signals were likely noise. Honest prior on a MEANINGFUL install
(>= 3-problem HumanEval/MBPP gain with retention): ~40% — above each parent's
prior because we are combining two believed-in positive ingredients, but the
double dose carries a real interference/forgetting risk and neither parent was
individually significant.

## Setup

- Model: only `Qwen/Qwen3.5-4B` (rev `851bf6e8…`); ONE fresh r32/a64 QLoRA adapter
  trained from the `base_reserialized` composite in a single stage.
- Dataset/task source: `data/sft_repair_why_stack.jsonl` — the deterministic
  UNION of the two already-built, already-verified, already-committed source
  corpora, COPIED into `data/source_corpora/` (sha-pinned) and combined by
  `scripts/build_corpus.py` (shuffle seed 93570): 504 self_repair rows + 504
  why_comment rows = 1008 rows, deterministically interleaved (combined sha
  `2462c93e…`). NO new generation.
- Train/eval split: training is the `spec -> #WHY:-commented correct code` +
  `spec -> repair-episode` union; evaluation is the held-out `spec -> code`
  HumanEval (164) + MBPP (200) — deliberately disjoint surfaces, with the grader
  ignoring comments. The agentic duet-eval is the primary real target, run
  manually as a follow-on.
- Baseline: base Qwen/Qwen3.5-4B on the same shared harness.
- Controls: contamination firewall re-run on the UNION (whole-word
  banned-benchmark-name audit, 0 hits over 1008 rows; distinctive code n-gram
  overlap, 0 — inherited from the two clean parents and re-verified present-only);
  comments inert to the grader; the base composite authenticated fail-closed
  (tree + weights) before training.
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

Rebuild the union corpus (deterministic, CPU-only; reproduces the combined sha):

```bash
python scripts/build_corpus.py              # writes corpus + receipt
python scripts/build_corpus.py --verify-corpus  # fail-closed verify only
```

GPU stages (each gated behind a staged adversarial review; see
`reports/preregistration.md` for the exact commands and checkpoint order):

```bash
python scripts/run.py --stage train     # r32/a64, 4 epochs, seed 93571
python scripts/run.py --stage merge     # vendored composite merger
python scripts/run.py --stage measure   # shared HumanEval+MBPP harness, both arms
```

Agentic confirm (manual follow-on on the merged composite — the PRIMARY real
target; base 8/35, self_repair 10/35, why_comment 8/35): run the duet-eval gen4
harness with `--model-override` set to
`large_artifacts/qwen35_4b_repair_why_stack/merged/repair_why_stack`, exactly as
bets #2 and #4's measure reviews documented for their composites.

## Results

Not yet run. The train/merge/measure stages are gated behind staged reviews. When
run, `runs/measure/transfer_summary.json` records all four pass@1 numbers
(base/treatment x HE/MBPP, counts + fractions), the per-problem paired deltas, and
the frozen, tightened verdict. Separate deployable evidence (transfer, a real code
improvement because the grader ignores comments) from the retention guard.

Construction facts already established (model-free):

- Union corpus sha `2462c93ea2a8dcfbd9413e1c6115ed1456ad438e5dabfdc01e924be6148ddbe5`,
  1008 rows, 504 self_repair + 504 why_comment, deterministically interleaved
  (shuffle seed 93570), reproducible from the two sha-pinned source copies.
- Source shas verified: self_repair `920cb228…`, why_comment `040be350…`.
- Contamination on the UNION: 0 whole-word banned-benchmark-name hits over all
  1008 rows; 0 distinctive shared 7-grams between the union's executable code and
  the benchmark code (present-only HF-cache aid; inherited-clean by set union of
  the two audited-clean parents).
- 53 unit tests green; `run.py --smoke` green; boundary drills refuse.

## Interpretation

Pending the sealed measurement. INSTALLED_CODING makes the stack the program's
reference composite and funds the agentic duet-eval confirm — the real test of
whether the two weak ingredients' signals are additive and real. RETENTION_FAIL
realizes the double-dose generation shift (reconsider dose/mix). NULL — no
>= 3-problem gain — with a flat agentic confirm would price the two individual
weak signals as likely noise rather than additive real effects; a preserved
boundary finding, not a re-roll.

## Knowledgebase Update

- Program evidence updated: pending measurement.
- Program backlog updated: pending measurement.
- Claim ledger updated: pending measurement.

## Artifacts

- `scripts/build_corpus.py` — deterministic UNION builder (source-sha verify +
  seeded interleave + banned-name audit + receipt); `contamination.py`; vendored
  `train_think.py` / `merge_adapter.py`; fail-closed `train_trial.py`;
  `measure_transfer.py`; `run.py`.
- `data/sft_repair_why_stack.jsonl` — the 1008-row union + `stack_corpus_receipt.json`;
  `data/source_corpora/` — the two sha-pinned source corpus copies;
  `data/provenance/base_reserialized.json`; `data/contamination/`.
- `configs/`, `reports/`, `tests/`
- `reports/artifact_manifest.yaml`

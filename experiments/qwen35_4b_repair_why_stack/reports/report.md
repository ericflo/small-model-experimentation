# Qwen35 4B Repair + Why Stack — Report

**Design-frozen report.** The model-free construction (the union corpus, its
deterministic build script + receipt, the contamination re-audit, the vendored
pipeline, the tests, the lineage package) is complete and verified; the
train/merge/measure GPU stages are gated behind staged adversarial reviews and
have not run. Results will be appended to the Results section once the sealed
measurement is read.

## Summary

Lifecycle 36 — the STACK of the cognitive-core coding program's two positive
coding bets. Bet #2 (self-repair, a LOOP behavior) was a weak positive on the
agentic target (agentic 8/35 -> 10/35, HumanEval +3); bet #4 (why-comment, WHY
causal reasoning) was a weak positive on the FUNCTION target (HumanEval +5,
agentic flat). They are COMPLEMENTARY and target-specific, and neither regressed
the other's target. This cell tests whether STACKING them — training one fresh
r32/a64 LoRA on the UNION of the two already-committed 504-row curricula,
deterministically interleaved into 1008 rows — captures BOTH gains (HumanEval ~+5
AND agentic ~10/35) and clears the significance the individual weak bets could
not. It merges onto base and is measured for TRANSFER + RETENTION on the shared
HumanEval + MBPP fitness harness under a frozen, TIGHTENED two-directional
consequence (INSTALLED_CODING requires a >= 3-problem gain).

This is LEANER than a fresh curriculum: both source corpora are already built,
verified, and committed. This cell COMBINES them (no new generation), so the union
carries no contamination risk beyond that of its two parents — verified, not
assumed. The why-comment rows are inert to execution grading, so a pass@1 gain is
an unconfounded CODE improvement.

## Research Program Fit

The program's target is the base 4B's agentic/multi-step cognition gap: HumanEval
76.2% (strong function coder) vs duet-eval 23% (weak agent). Bet #1 (a passive
skill) reshuffled without raising; bet #2 (a loop behavior) nudged the agentic
loop; bet #4 (causal reasoning) nudged per-function correctness. The two positive
bets pointed at DISJOINT failure modes. This stack attacks the additivity
question: does combining two believed-in, target-specific positive ingredients
capture BOTH gains at once? HumanEval/MBPP serve as the fast transfer + retention
signal; the agentic duet-eval (base 8/35, self_repair 10/35, why_comment 8/35) is
the PRIMARY real target, run manually as a follow-on (not gated by this cell).

## Method

- **Union build** (`scripts/build_corpus.py`, shuffle seed 93570). The two source
  corpora are COPIED into `data/source_corpora/` (sha-pinned) and combined by a
  deterministic, fail-closed builder: each source sha is verified against its pin
  BEFORE combining (abort on mismatch); the 504 self_repair + 504 why_comment
  non-blank JSONL lines are concatenated in the frozen order (self_repair, then
  why_comment) EXACTLY as their bytes appear (no re-serialization, so each row's
  encoding is preserved); the 1008 lines are then deterministically shuffled with
  `random.Random(93570)` so the two kinds INTERLEAVE. Combined sha
  `2462c93ea2a8dcfbd9413e1c6115ed1456ad438e5dabfdc01e924be6148ddbe5`; the sha is a
  pure function of the two source shas + the shuffle seed and is verified stable
  across two independent rebuilds. `data/stack_corpus_receipt.json` documents the
  source shas, combine order, shuffle seed, final sha, and row count by kind (504
  self_repair + 504 why_comment).
- **Contamination re-audit** (`scripts/contamination.py`). The committed banned set
  of all 668 HumanEval + MBPP function names (663 after the language whitelist;
  name set byte-identical to both parents' fixtures) — zero whole-word hits over
  all 1008 rows (prompt + think + answer). A present-only code n-gram aid — zero
  distinctive shared 7-grams between the union's executable code (docstrings +
  comments stripped) and benchmark solution code; the union's code 7-grams are the
  union of the two clean parents' code 7-grams, so 0 by set union, re-verified over
  the combined corpus. No new generation -> no new contamination risk beyond the
  union.
- **Install** (`scripts/train_trial.py` -> vendored `scripts/train_think.py`). One
  fresh r32/a64 adapter, 4 EPOCHS (the converged recipe for the high-entropy
  why-comment rows the union contains), lr 1e-5, batch 1, grad-accum 8, max-length
  4096, w_think 0.2, w_close 0.2, seed 93571 (126 optimizer steps/epoch), from the
  `base_reserialized` composite. The base is authenticated FAIL-CLOSED (in-cell
  provenance copy + full tree manifest + full 9 GB weights hash) before training.
- **Merge** (vendored `scripts/merge_adapter.py`) with `--base-model` = the base
  composite -> `merged/repair_why_stack`.
- **Measure** (`scripts/measure_transfer.py` -> SHARED harness, referenced not
  copied). Base and repair_why_stack, HumanEval 164 + MBPP 200, greedy pass@1,
  identical vLLM path; all four numbers (counts + fractions) + per-problem paired
  deltas + the frozen, tightened verdict recorded. The grader ignores comments, so
  a pass@1 gain is an unconfounded CODE improvement.

## Results

Pending the sealed measurement. `runs/measure/transfer_summary.json` will carry
`pass_at_1{base,repair_why_stack}{humaneval,mbpp}`, the pass counts, the McNemar
b/c paired deltas per dataset, and the frozen consequence. Deployable transfer
evidence is a >= 3-problem pass@1 gain; the retention guard is the paired dataset
staying within 0.02. The agentic duet-eval confirm (the LOOP direction) is the
manual follow-on that decides whether both weak signals are additive and real.

Construction facts already established (model-free):

- Union corpus sha `2462c93ea2a8dcfbd9413e1c6115ed1456ad438e5dabfdc01e924be6148ddbe5`,
  1008 rows, 504 self_repair + 504 why_comment, deterministically interleaved
  (shuffle seed 93570); reproducible from the two sha-pinned source copies.
- Source shas verified: self_repair `920cb228…`, why_comment `040be350…`.
- Contamination on the UNION: 0 banned-name whole-word hits over 1008 rows; 0
  distinctive shared 7-grams (present-only HF-cache aid ran; 61 shared spans, all
  pure control-flow idioms).
- 53 unit tests green (present-only cache aids RUN with the HF cache; the union is
  re-audited by kind, by banned name, and by distinctive n-gram); `run.py --smoke`
  green; boundary drills refuse.

## Controls

- Contamination firewall re-run on the UNION (banned-name audit + distinctive code
  n-gram overlap), both zero, so a benchmark movement cannot be memorization.
- Comments are inert to the execution grader, so any pass@1 gain is a CODE gain,
  not a grader artifact.
- Base composite authenticated fail-closed (tree + weights) before training and
  merge; a swapped composite aborts.
- Identical measurement path for both arms (the shared harness), so base and
  treatment pass@1 are directly comparable.
- Tightened consequence rule: a noise-level (<3-problem) bump reads NULL, fixing
  bet #1's letter-of-the-law false positive.
- The union is a deterministic, source-sha-pinned combination (no new generation),
  so the training data is a verifiable function of the two audited parents.

## Oracle Versus Deployable Evidence

Deployable evidence = a >= 3-problem HumanEval/MBPP pass@1 gain (real, held-out
`spec -> code` generation, comments ignored by the grader). The retention guard
(the other dataset within 0.02) is a control on the double-dose forgetting risk,
not a capability claim. The agentic duet-eval is the eventual deployable target and
the real test of the stack's LOOP direction, but is a manual follow-on confirm, not
gated here. No metric here uses hidden labels beyond the one-shot frozen verdict
read.

## Interpretation

Pending measurement. INSTALLED_CODING: the two complementary ingredients combined;
the stack becomes the program reference and funds the agentic duet-eval confirm —
the real test of whether BOTH weak signals are additive and real. RETENTION_FAIL:
the double-dose generation shift is realized; reconsider dose/mix. NULL — no
>= 3-problem gain — with a flat agentic confirm prices the two individual weak
signals as likely noise rather than additive real effects; a preserved boundary
finding, not a re-roll.

## Next Experiments

- If INSTALLED_CODING: run the agentic duet-eval confirm on the repair_why_stack
  composite; a confirmed agentic gain alongside the HumanEval gain makes the stack
  the program reference.
- If RETENTION_FAIL: reconsider the dose / mix ratio before any confirm.
- If NULL: do NOT re-roll this union; a flat stack (fast AND agentic) is strong
  evidence the two weak signals were noise — advance to the think-block WHY variant
  (bet #3, still queued) or a different mechanism.

## Artifact Manifest

See `artifact_manifest.yaml` — the trained adapter and merged composite live under
`large_artifacts/` (omitted from git); the union corpus + build script + receipt,
the two source corpus copies, the contamination fixture, the base provenance copy,
and the receipts are in-repo and reproducibility-critical.

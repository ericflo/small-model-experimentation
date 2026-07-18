# Qwen35 4B Repair + Why Stack Experiment Log

## 2026-07-18 — model-free construction frozen

Lifecycle 36, the STACK of the cognitive-core coding program's two positive
coding bets. Mission: install real coding capability into base Qwen/Qwen3.5-4B via
a designed, contamination-free curriculum, proven by transfer. Meta-context: bet
#2 (self-repair, a LOOP behavior) was a WEAK POSITIVE on the agentic target
(agentic 8/35 -> 10/35, HumanEval +3); bet #4 (why-comment, WHY causal reasoning)
was a WEAK POSITIVE on the FUNCTION target (HumanEval +5, agentic flat). The
cross-bet finding: the two are COMPLEMENTARY and target-specific, and neither
regresses the other's target.

The bet: STACK the two positive ingredients — train one fresh r32/a64 LoRA on the
UNION of the two already-committed 504-row curricula — and test whether the
combined effect captures BOTH gains (HumanEval ~+5 from WHY AND agentic ~10/35
from repair) and clears the significance the individual weak bets could not. This
is LEANER than a fresh curriculum: both source corpora are already built,
verified, and committed; this cell COMBINES them, it does not regenerate. No new
generation -> no new contamination risk beyond the union.

Built and verified (no GPU, no commit):

- `scripts/build_corpus.py` (union-build shuffle seed 93570) — the deterministic,
  fail-closed UNION builder. Verifies BOTH source copies' sha against their pins
  BEFORE combining (self_repair `920cb228…`, why_comment `040be350…`), concatenates
  their 504 + 504 non-blank JSONL lines in the frozen order (self_repair, then
  why_comment) EXACTLY as their bytes appear (no re-serialization), then shuffles
  the 1008 lines with `random.Random(93570)` so the two kinds INTERLEAVE. Combined
  sha `2462c93ea2a8dcfbd9413e1c6115ed1456ad438e5dabfdc01e924be6148ddbe5`; the sha
  is a pure function of the two source shas + the shuffle seed and is verified
  stable across two independent rebuilds. `data/stack_corpus_receipt.json`
  documents source shas, combine order, shuffle seed, final sha, and row count by
  kind (504 self_repair + 504 why_comment). `--verify-corpus` re-derives it twice
  in memory and fails closed against the committed corpus + receipt + pin (used by
  `run.py --smoke`).
- `data/source_corpora/sft_self_repair.jsonl` + `sft_why_comment.jsonl` — the two
  COPIED, sha-pinned source corpora (the standalone reproduction inputs, per the
  standalone-experiments doctrine: lineage = copied ordered SFT datasets, not
  cross-experiment references).
- `scripts/contamination.py` + `data/contamination/banned_function_names.json`
  (668 benchmark function names, 663 after whitelist; name set byte-identical to
  both parents' fixtures). Re-audited on the UNION: 0 whole-word hits over all
  1008 rows (prompt + think + answer); 0 distinctive shared 7-grams between the
  union's executable code (docstrings + comments stripped) and benchmark code
  (present-only HF-cache aid; 61 shared spans, all pure control-flow idioms). The
  union's code 7-grams are the union of the two clean parents' code 7-grams, so 0
  by set union.
- Vendored `scripts/train_think.py` (sha e0eca2a2…) and `scripts/merge_adapter.py`
  (sha cb9af8b4…), byte-identical to the sibling cells' trainer/merger.
- `scripts/train_trial.py` — fail-closed base authentication (in-cell provenance
  copy + tree manifest + full 9 GB weights hash); recipe r32/a64, 4 epochs, lr
  1e-5, seed 93571 (126 optimizer steps/epoch). The 4-epoch recipe is inherited:
  the why_comment rows in the union are the high-entropy `#WHY:` target that
  undertrained at 1 epoch (why bet #4 was retrained at 4 epochs).
- `scripts/measure_transfer.py` — invokes the shared fitness harness for both arms
  x both datasets; the grader IGNORES comments; frozen, TIGHTENED INSTALLED_CODING
  (>= 3-problem gain) / RETENTION_FAIL / NULL consequence, identical to bets #2/#4.
- `scripts/run.py` — checkpointed `--smoke | --stage train | --stage merge |
  --stage measure`; each GPU stage gated behind a staged adversarial review.
- 53 unit tests green (present-only HF-cache aids RUN with the cache; the union is
  re-audited by kind, by banned name, and by distinctive n-gram; source shas
  verified; deterministic-shuffle 2-build identity; consequence-rule truth table;
  base auth fail-closed); `run.py --smoke` green; boundary drills refuse.

Grep-fresh note: union-build seed 93570 and training seed 93571 are fresh
repo-wide as SEEDS (the only textual matches are incidental substrings inside
unrelated floats). No training-seed collision.

Two-directional reading (frozen): the stack succeeds only if it captures BOTH
targets. The fast HumanEval/MBPP gate tests the FUNCTION direction (the why_comment
gain); the agentic duet-eval (base 8/35, self_repair 10/35, why_comment 8/35) tests
the LOOP direction (the self_repair gain) and is the PRIMARY real target, run
manually as a follow-on. If BOTH the HumanEval gain (~+5) AND the agentic gain
(~10/35) appear, the stack works and the two individual weak signals are confirmed
real; if flat, they were likely noise. Honest prior on a MEANINGFUL install: ~40%.

GPU stages (train/merge/measure) are pending their staged reviews.

## 2026-07-18 — Transfer measurement: MIXTURE DILUTION (naive stack fails)

- Corpus-union stack vs base: HumanEval +1 (why_comment alone was +5),
  MBPP -3, agentic 7/35 (self_repair alone was 10/35). Frozen rule: NULL.
- The union DILUTED BOTH component effects - one adapter on both
  curricula at half concentration splits capacity. Confirms the
  menagerie mixture-dilution law for coding: complementary effects do
  NOT combine via corpus-mixing.
- CORRECT combination is weight-space at full magnitude: task vectors
  (base + repair_delta + why_delta). Testing that next (cheap, no
  training).

## 2026-07-18 — Task-vector combination ALSO fails (worse than dilution)

- Tested the weight-space combination (base + (self_repair-base) +
  (why_comment-base), 128/723 tensors changed, full-magnitude both
  deltas): HumanEval 126/164 (+1, same as the union - WHY gain gone),
  agentic 3/35 (WELL BELOW base 8/35). Summing two full-strength deltas
  INTERFERES/overshoots, hurting agentic badly.
- CONCLUSION: the two weak positives do NOT combine by ANY method -
  corpus-union DILUTES (both -> ~base), task-vector INTERFERES (agentic
  degrades to 3/35). Combined with neither being individually
  significant, this is strong evidence the individual effects
  (why_comment +5 HE, self_repair +2 agentic) are FRAGILE / at the noise
  floor. The SFT-curriculum thread has been thoroughly explored (6 bets)
  without a robust, combinable, significant coding gain.

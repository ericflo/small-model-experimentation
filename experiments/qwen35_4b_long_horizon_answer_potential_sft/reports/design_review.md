# Adversarial Design Review

## Verdict

`sound_with_fixes_applied`. The experiment directly addresses C51's censored interface and reaches the SFT
claim, but only after the following controls were made mandatory.

## Findings And Applied Fixes

1. **A larger maximum is still a cap.** Calling 12,288 “uncapped” would be misleading. The design calls it
   a context-safety allowance, resumes non-loop contacts once, never treats contacts as complete, and reports
   remaining censoring.
2. **Natural-close survivorship can select easy tasks.** All trace arms use the same eligible task set in
   the common-task analysis; answer-only/full-coverage results remain separate.
3. **Answer potential may select answer rehearsal.** Pre-answer-mention checkpoints, answer-copy rate,
   joint score, and task-shuffled training are mandatory. The full thought is kept, but copying cannot be
   mistaken for a strategy mechanism.
4. **Longer traces can win through token/style exposure.** Random natural traces are selected nearest to
   each treatment length; training and inference tokens are reported.
5. **Binary RFT has noisier coverage by construction.** It gets the exact same R1 rollout budget per
   candidate. Unique-task coverage and a common-task training analysis prevent a diversity advantage from
   masquerading as per-trace quality.
6. **Answer learning can explain every adapter gain.** The empty-thought arm keeps the canonical answer and
   close seam with matched updates.
7. **Shuffled traces can fail only because lengths/families differ.** Reassignment is within
   family/level/length strata and uses the same selected trace multiset.
8. **Pivot branching can silently buy more compute.** Independent N=64 remains nested and complete;
   preserved-prefix prefill and suffix tokens are counted; branch adoption is reported.
9. **Checkpoint choice could peek at rollout outcomes.** It uses likelihood scores only; rollout labels are
   unavailable to the branch constructor.
10. **The previous trace-prior comparator was missing.** The smoke runner now requires finite sampled-trace
    cumulative log-probability before the first scientific shard.
11. **Full-sequence HF scoring could disagree with vLLM.** A frozen 32-row bf16 parity gate precedes bulk
    scoring. HF is used uniformly for this internal measurement; all compared generation remains vLLM.
12. **Multiple score treatments invite cherry-picking.** Answer gain is the original primary; joint gain is
    the predeclared seam repair. Both arms train and all comparisons are reported.
13. **Near-self-distillation may make full-thought SFT inert.** This is the claim under test, not a reason to
    stop. Thought loss is 0.5, close/answer loss 1.0, and C50's empty/random/shuffled controls distinguish
    content from interface learning.
14. **A 16k selected row may OOM or be silently truncated.** Batch 1 with gradient accumulation is frozen;
    training fails on any encoded overflow. A long-sequence smoke precedes all adapters.
15. **vLLM runtime LoRA can no-op.** Only merged full checkpoints are evaluated, with a real behavioral
    on/off gate.
16. **A large raw pool can swamp git and break restartability.** Raw generation/scoring shards are external,
    atomic, checksummed, and manifest-indexed. Compact selectors, datasets, summaries, and failures stay in
    git.
17. **The experiment could again stop on a merely modest selector.** All AUROC/effectiveness stops were
    removed. Only instrument validity and minimum natural-trace availability can block SFT.
18. **Compression could rescue or confound a result.** It is prohibited here. First test full strategies;
    only a positive full-trace result licenses a fresh compression experiment.

## Remaining Limits

- One base model and a procedural atom substrate.
- Reference-answer curation is oracle-side and not deployable online.
- One primary training seed unless the registered replication trigger fires.
- The 16,384-token context remains a physical upper bound.
- R1 success is intentionally a noisy but faithful rejection-sampling baseline.

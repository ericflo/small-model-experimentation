# Adversarial Design Review

**Decision:** pass for model training after the exact bytes, seeds, and gates below are
committed and pushed.

## Main threats and dispositions

1. **The parent replay advantage was a token-dose artifact.** The new 0/40/80 arms
   each have 1,520 rows, 190 updates, and exactly 1,429,053 forward tokens. The two
   40-row replay blocks match their designed counterparts independently (16,732 and
   16,543 tokens), and all arms share the same 1,440 replay rows and slot permutation.
2. **Forward-token equality hides different loss-bearing composition.** Designed
   content necessarily changes prompt/think/answer target allocation. The receipt
   reports each component. This is the intended content mechanism, not unreported
   compute; conclusions remain about the complete curriculum, not a single token type.
3. **The replay-refresh anchor may be a one-seed fluctuation.** Base, C53 `blend`,
   inherited replay refresh, and a fresh exact-token replay continuation all run on
   the same new benchmark seed. No claim is allowed without independent quick and
   medium replication.
4. **Local score is a poor broad-transfer selector.** Local gates are safety and
   installability filters only. Both doses are registered now, and every independently
   eligible arm enters the same paired event; local ranking does not choose one dose.
5. **Two doses create multiplicity.** Both are disclosed as a small preregistered
   ladder. Any winner is exploratory until it repeats on new seeds and beats
   matched-compute sampling.
6. **Replay continuation could itself win.** It is both the mechanism control and an
   eligible synthetic curriculum. It must satisfy the same strict all-family and
   inherited-anchor gates; a single event still cannot establish universality.
7. **Adapter/backend confusion could erase the edit.** Every benchmark-eligible arm is
   explicitly merged, authenticated by nonzero adapter and full-weight hashes, and
   evaluated with base and controls on the same `qwen_vllm` backend. Runtime LoRA is
   forbidden.
8. **Benchmark targeting could contaminate the data.** The selection algorithm sees
   only copied synthetic rows, row metadata, and tokenizer lengths. It never opens the
   benchmark directory or consumes family-level semantics beyond public aggregate
   labels already emitted by the trusted gateway.
9. **Repeated replay may memorize the gym rather than generalize.** The blackbox paired
   event is the transfer arbiter. Replay is treated as an active baseline, and later
   confirmation must beat matched-compute sampling rather than call repeated training
   sufficient by itself.

## Required preflight

- Source and derived SHA-256 values match the manifest.
- Every designed dose contains all 13 skills.
- Arm line differences are exactly 40, 40, and 80 in the registered nested ladder.
- All 4,560 rows tokenize without skips and every arm totals 1,429,053 tokens.
- Smoke, dose tests, local-gate tests, and repository checks pass.

All preflight items are satisfied model-free. Training remains pending the pushed
design checkpoint.

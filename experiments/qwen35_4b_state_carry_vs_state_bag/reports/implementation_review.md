# Implementation Review

## Scope and Disposition

This records the initial pre-GPU review and the final adversarial amendment below. It is not a model result. The original review found the first harness ready for G0, but a deeper independent pass subsequently found launch blockers in pilot isolation, cross-process determinism, crossed statistics, checkpoint provenance, causal enforcement, and deployment-baseline validity. The amendment supersedes any conflicting readiness statement in the historical sections.

Disposition after amendment: **CPU unit/static contracts pass; fresh CPU smoke and full-corpus regeneration remain required before G0, which is the first model-bearing action. Training remains machine-blocked until `MODEL_SMOKE_PASS`.**

## Contracts Audited

- The only model identifier and revision accepted by configuration and loading code are `Qwen/Qwen3.5-4B` and `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- All result-bearing arms use Transformers 5.13.0. There is no experiment vLLM runner, and analysis isolates results by resolved config hash.
- The pinned model config was checked against the [official revision artifact](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json): hidden size 2560, 32 text layers, and the exact repeating three-linear/one-full-attention motif used to place R at layers 12–19.
- The pinned Transformers decoder contract was checked against the [official v5.13.0 source](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py): decoder layers return a tensor, accept the masks/position geometry used here, and use the same causal/full and recurrent-linear mask construction exercised by K=1 parity.
- State slots are one registered token repeated exactly eight times before `Query:`. Live token identity, contextual answer-token stability, causal ordering, and exact K=1 logits are hard gates.
- Carry and Bag are one code path with one edge switch. They share initialization, parameters, step signal, R-call count, prompt, aggregation, coda, and decoder-layer-token accounting.
- The first R application is adapter-free. Only extra R calls enable LoRA, and discovered trainable LoRA names must map to layers 12–19.
- Only state-slot activations cross extra R calls; all other positions reset to the first-R memory. The coda executes once.

## Adversarial Findings Fixed During Review

1. **A unique joint trajectory could still have a shallow-equivalent answer.** Generation now also rejects an earlier occurrence of the terminal queried node/checksum, including both members of every donor pair.
2. **Counterfactual swaps had avoidable surface confounds.** Pair members now share world, labels, table order, template, query, and choice order; only initial state and consequence differ.
3. **The auxiliary node ID was not observable.** The node head now predicts the visible table-row coordinate, shared by counterfactual pair members, rather than a hidden generator index.
4. **Compressed data hashes were time-dependent.** gzip filename and timestamp headers are frozen, and a regression test rebuilds the corpus twice and compares every archive hash.
5. **Counterfactual rows could bypass the split firewall.** Every structural fingerprint is now globally unique; pair membership grants no exception.
6. **Result paths were not portable.** Evaluation and sample-more summaries now store row artifacts relative to their summary, and analysis has a regression test for relocation.
7. **Retained pilot results could overwrite or pool with full results.** Analysis deterministically prefers non-pilot bundles whenever any exist and refuses duplicate scientific cells.
8. **An attractive tiny run could reach a scientific label.** All primary depths must meet the frozen pooled item minimum in addition to the three-seed requirement.
9. **The trained Carry edge cut was generated but not analyzed.** Analysis now pairs intact Carry and Bag-mode inference from the same Carry checkpoint by seed and matched task.
10. **Carry and Bag equality was asserted more strongly than receipted.** Checkpoints now contain initial trainable-value/name hashes and cumulative training prompt/layer-token totals; analysis refuses a pair if these differ.
11. **Artifact hashes were written but not enforced.** Evaluation reload verifies model/config identity plus every adapter and loop-state hash; sample-more does the same for text adapters. Prepared data is similarly checked against its clean manifest and data-contract hash.
12. **The text comparator had a different learning-rate schedule.** It now shares warmup, cosine decay, clipping, nonfinite checks, and loop-layer LoRA targeting with the recurrent arms.
13. **The deployment verdict used unpaired point estimates.** It now requires all three seed-matched Carry/text-baseline pairs and a positive hierarchical-bootstrap lower bound against exact-verifier oracle `pass@N`.
14. **Sample-more omitted promised cost diagnostics.** Per-item actual sampled-token counts and synchronized generation seconds are preserved alongside the conservative layer-token allocation.
15. **Corrupt or partial numerics could pass comparisons.** G0 and both training paths now fail on nonfinite parity error, loss, or gradient norm; every trained Carry/Bag checkpoint repeats direct-model K=1 parity before evaluation.

## Historical CPU Evidence

The counts below describe the initial implementation before the final source-bound amendment. They
are retained as setup history, not accepted as current receipts.

- 25 unit and static-contract tests pass.
- CPU smoke returns `CPU_SMOKE_PASS`, distinct Carry/Bag reference mechanics, equal compute receipts, three task families, and zero benchmark reads.
- A complete default-size corpus build succeeds: 12,000 train rows, 1,024 validation rows, four 3,200-row evaluation splits, and 512 two-row counterfactual pairs.
- The complete corpus reports zero structural cross-split duplicates, zero benchmark reads, exact registered depth allocations, and deterministic content hashes.
- Setup-only analysis emits `SETUP_ONLY`; no empty or partial artifact can produce a positive verdict.
- Every Python source compiles without importing unavailable GPU packages during CPU tests.

## Residual Live Risks and Their Gates

1. **Private-library geometry drift:** the manual forward deliberately uses Qwen/Transformers internals. Exact K=1 answer-logit parity at `1e-5` is the deciding test; do not relax it.
2. **PEFT name or restoration drift:** G0 audits discovered targets and gradients, while checkpoint reload audits identities and hashes. Any failure is an implementation stop.
3. **Fast-path or memory failure:** G0 requires both Qwen fast paths, at least 44 GiB exposed VRAM, a K=4 backward pass, and peak-memory receipt before training.
4. **Optimization failure:** the paired 300-step pilot is the sole continuous-design promotion gate. Preserve a miss; do not shop seeds or thresholds.
5. **Readable but unused state:** sufficiency heads cannot establish the claim. Trained edge cuts and donor-consequence transport are separately required evidence.
6. **Overthinking or trained-horizon memorization:** full K-by-depth curves include K=4 versus matched K through 12; the unseen-K lower bound must be positive.
7. **Shallow compute remains stronger:** even a mechanistic pass is not deployable until the matched explicit-CoT oracle is decisively beaten.

## Reviewer Conclusion

The harness now fails closed on the main ways this experiment could fool us: extra compute, different initialization, shallow-equivalent tasks, probe-only state, generic swap damage, pilot leakage, incomplete cells, corrupt artifacts, and a weak or under-replicated sample-more baseline. The remaining uncertainty is exactly the uncertainty the Ada run is meant to resolve: whether the patched Qwen middle block can learn a stationary, serial state transition that keeps improving beyond its trained unrolling horizon.

## Final Amendment Validation

- 41 CPU unit/static-contract tests pass after the integrated audit.
- Every experiment Python source and script passes `py_compile`; `git diff --check` is clean.
- Dedicated pilot validation now prevents pilot training logs and checkpoint parity from reading the
  confirmatory validation corpus.
- Only the exact frozen default config can enter model-bearing stages or emit a scientific verdict;
  smoke/reduced configs are machine-labeled setup-only.
- Swap uncertainty bootstraps 512 counterfactual-pair means, not 1,024 correlated directions, while
  both preregistered donor contrasts are enforced per seed.
- Immutable labels/geometry are checked across Carry/Bag, edge-cut, and deployment pairs; sample-more
  allocation, parses, raw totals, and by-depth interface rates are recomputed in analysis.
- No fresh CPU smoke, full corpus, Qwen load, GPU inference, training, or benchmark access occurred
  during this final review. The historical receipt is explicitly superseded.

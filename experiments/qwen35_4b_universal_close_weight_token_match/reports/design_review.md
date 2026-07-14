# Adversarial Design Review

**Verdict:** pass after revision. Expensive work is authorized only after this
design checkpoint is committed and pushed to `main`.

## Primary threats and resolutions

1. **This could repeat C51's unreachable counterfactual state.** C51 injected a
   close after cap-bound thoughts and measured the answer state beyond it. Here the
   scored/trained event includes the model's autonomous `</think>` transition at the
   verified natural end of each chain. No close is injected at evaluation.
2. **C50's forced-close story can be misread.** The successful C50 recipe's nominal
   forced-close rows were skipped; its evidence supports emission-seam loss placement,
   not direct imitation of a synthetic injected close. This treatment therefore
   weights natural closes and does not manufacture post-cap answer states.
3. **Failure-targeting is post hoc.** The immediate parent's experiment-owned local
   receipt showed all three unparsed designed160 cases were cap-bound execute/induct
   cases. That result is legitimate design input for a new experiment, but seed
   88,005 is retired. Target rows are selected without outcomes and fresh local seed
   88,006 determines the result.
4. **The treatment could silently change the data.** `standard_xi` and `close_xi`
   point to the same SHA-256 `12fc613b...14f00` file and use the same training seed.
   Unit tests assert identical token ids, labels, and non-close assigned weights;
   only target-kind close-span weights change from 0.2 to 1.0.
5. **Loss normalization prevents a literally one-coordinate gradient contrast.**
   The trainer normalizes by absolute assigned loss mass per batch, so increasing two
   close-token weights slightly rescales the other target-row contributions. This is
   part of the registered close-emphasis intervention and is not described as an
   isolated logit edit. The byte-identical standard arm remains the causal control.
6. **Matched rows and steps could hide token compute.** Batch size is one, so padding
   cannot confound compute. Every arm has 320 rows, 40 updates, exactly 286,814
   forward tokens, and zero skips. The targeted and replay variable blocks each sum
   to exactly 87,454 forward tokens.
7. **Fresh lessons could duplicate the parent.** The constructor authenticates the
   parent's deterministic 160-row source-index set (hash `6228daa8...25af`) and
   asserts zero overlap with the 80 new target rows. The full indices are preserved
   in `stream_manifest.json`.
8. **More training alone could explain a pass.** `replay_repeat` starts from the same
   parent and receives the same rows, tokens, steps, learning rate, and seed. It is an
   active control, not treated as a neutral no-op.
9. **Targeted data and close weighting could be conflated.** `standard_xi` tests the
   80 fresh execute/induct rows at ordinary weights; `close_xi - standard_xi` tests
   close emphasis. Both registered arms run before local evaluation.
10. **The parent could be unauthenticated.** The wrapper rejects any warm start whose
    weights/config differ from `f05c13ae...94654` / `0cd3ca7c...91e58`.
11. **A forgiving screen could promote noise.** Fresh local seed 88,006 retains the
    parent's absolute gates: accuracy ≥0.65, parse ≥0.90, cap contacts ≤2, and no
    repeated feasible-route abstention. Only treatment arms can promote; all arm and
    parent summaries are preserved.
12. **Two candidates create winner's-curse risk.** Both are prospectively registered.
    Every locally eligible candidate enters the same paired aggregate event and is
    reported independently; there is no adaptive winner-only benchmark.
13. **Backend drift or benchmark leakage could fake breadth.** The benchmark is
    conditional, aggregate-only, and invoked exclusively through the trusted gateway.
    All arms are explicitly merged and evaluated in one `qwen_vllm` event. Benchmark
    items, family sources, transcripts, and private output remain unread.
14. **A quick pilot could be overstated.** A candidate must improve every reported
    family versus base and beat `blend`, replay refresh, its immediate parent, and
    active replay in aggregate merely to pass the pilot. A universal claim still
    requires fresh quick replication, medium@2,048, paired uncertainty, and a
    matched-compute sample-more comparison in a separate confirmation experiment.

## Frozen artifact identities

- Parent weights/config: `f05c13ae...94654` / `0cd3ca7c...91e58`.
- Source-token receipt: `064a1cce...f542f`.
- Stream manifest: `abf8b505...0966f`.
- Stream token receipt: `e2e95429...2915b`.
- Replay stream: `6ec82e29...e81d4`.
- Byte-identical standard/close stream: `12fc613b...14f00`.
- Construction seed: 77,110; training seed: 44; local seed: 88,006;
  conditional aggregate seed: 78,136.

No model training, local evaluation, merge, or benchmark event has run during this
review.

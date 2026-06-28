# qwen35_4b_retrieval_adapt_verify_scale

Standalone experiment: scale retrieval-and-adaptation on direct-sampling misses, then test whether deployable verification can turn new candidate-pool coverage into selected pass@1.

The experiment uses Qwen3.5-4B as the adapter/reranker and a verified MBPP train library as external algorithmic memory. Hidden tests are used only for evaluation summaries and oracle ceilings.

## Primary Question

Can semantic retrieval plus adaptation recover tasks missed by direct Qwen sampling, and can public-test-safe selectors distinguish the hidden-correct adaptations from visible-pass hidden-wrong ones?

## Arms

- `base_direct_k4`: 80 held-out MBPP tasks, four direct samples per task.
- `retrieval_copy_rename_top3`: copy top-3 retrieved train-library algorithms after renaming the function.
- `retrieval_adapt_semantic_top3`: Qwen adapts the top-3 semantic retrievals.
- `retrieval_adapt_random_top3`: random-library control.
- `retrieval_adapt_shuffled_top3`: shuffled-query retrieval control.
- Selector arms over residual tasks:
  - `first_visible`
  - `shortest_visible`
  - `consensus_visible` using target-independent agreement probes.
  - `frozen_qwen_visible_rerank`
  - `oracle_hidden` as non-deployable headroom.

## Main Metrics

- Direct baseline coverage on all 80 tasks.
- Residual zero-to-one pool coverage on baseline misses.
- Visible-pass hidden-wrong rate.
- Selected recovery rate for deployable selectors.
- Combined all-task coverage/pass@1 implied by direct baseline plus retrieval recovery.
- Forward-token cost.

## Artifact Layout

- `configs/experiment.json`: complete run configuration.
- `data/`: JSONL candidate pools, retrieval plan, selector records, summaries.
- `reports/`: final report, figures, machine-readable summary.
- `run_logs/`: stdout/stderr logs for each script.
- `logs/experiment_log.md`: chronological experiment log.
- `/workspace/large_artifacts/qwen35_4b_retrieval_adapt_verify_scale/`: reserved for large files such as checkpoints. This run should not put small downloadable artifacts there.

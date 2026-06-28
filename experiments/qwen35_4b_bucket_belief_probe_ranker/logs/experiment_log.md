# Experiment Log

## Setup

- Created a standalone experiment directory at `/workspace/experiments/qwen35_4b_bucket_belief_probe_ranker`.
- Created a separate large-artifact root at `/workspace/large_artifacts/qwen35_4b_bucket_belief_probe_ranker`.
- Hypothesis: a Qwen3.5-4B LoRA can improve deployable active probing if it learns to infer which output bucket contains the hidden target program, rather than only selecting probes by target-independent split quality.

## Planned Arms

- `split_top1`: choose the best remaining probe by target-independent expected survivors.
- `oracle_top8`: choose the target-aware best probe from the same top-8 split-ranked candidate probes.
- `fullpool_oracle`: choose the target-aware best probe from the full 96-case pool.
- `base_bucket_ranker`: untrained Qwen3.5-4B bucket-belief scoring over the top-8 probes.
- `sft_bucket_ranker`: QLoRA SFT bucket-belief scoring over the top-8 probes.

## Notes

- The full-pool and top-8 oracle arms are headroom measurements, not deployable policies.
- A deployable win requires `sft_bucket_ranker` to beat `split_top1` and `base_bucket_ranker`.
- If the SFT ranker stays near `split_top1`, the result means target-aware oracle headroom is not recoverable from this bucket prompt alone.

## Run Results

- Built 300 train records, 160 eval records, 619 train process states, and 334 eval process states.
- Built 4,952 train bucket examples and 2,672 eval bucket examples.
- Trained Qwen3.5-4B QLoRA bucket-belief adapter for 220 optimizer steps.
- Budget-3 hidden-all accuracy:
  - `split_top1`: 48.8%
  - `oracle_top8`: 61.3%
  - `fullpool_oracle`: 86.9%
  - `base_bucket_ranker`: 48.8%
  - `sft_bucket_ranker`: 50.0%
- Budget-3 compare-gate accuracy:
  - `split_top1`: 3.8%
  - `oracle_top8`: 23.8%
  - `fullpool_oracle`: 73.8%
  - `base_bucket_ranker`: 3.8%
  - `sft_bucket_ranker`: 5.0%
- Interpretation: bucket-belief SFT produced a real but tiny deployable lift, capturing about 1.3 points of a 12.5-point top-8 oracle gap. The larger full-pool oracle gap remains mostly unrecovered.
- Moved the generated train prompt JSONL to `/workspace/large_artifacts/qwen35_4b_bucket_belief_probe_ranker/cache/bucket_train_examples.jsonl` after training to keep the experiment directory compact.

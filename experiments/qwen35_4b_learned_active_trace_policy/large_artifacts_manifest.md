# Large Artifacts Manifest

Large files are intentionally stored outside this experiment directory so the experiment folder can be downloaded without checkpoints.

Root:

`/workspace/large_artifacts/qwen35_4b_learned_active_trace_policy`

Expected model artifacts:

- `models/sketch_lora/`: Qwen3.5-4B typed-sketch adapter and checkpoints.
- `models/policy_lora/`: Qwen3.5-4B learned active-query policy adapter and checkpoints.

Artifact audit after the final run:

- Experiment directory size: `85M`.
- Large artifact directory size: `1.3G`.
- No files larger than `50M` are stored inside the experiment directory.
- Sketch adapter used for final evaluation: `models/sketch_lora/`.
- Policy adapter used for final evaluation: `models/policy_lora/checkpoint-40/`.
- Policy checkpoint 40 was selected because its validation loss, `0.399913`, was better than checkpoint 80, `0.405868`.

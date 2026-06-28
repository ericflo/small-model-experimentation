# Large Artifacts Manifest

This experiment has local trained adapter outputs under `reports/adapters/`. That directory is intentionally ignored and must not be checked into git.

| artifact | local path | reason | regenerate |
| --- | --- | --- | --- |
| sequence-policy LoRA adapter | `reports/adapters/lora_seq_policy/` | trained adapter output | `python scripts/run_live_tool_dagger.py --root /workspace/experiments/qwen35_4b_live_tool_dagger --generate-traces --train-lora --max-repairs 2 --max-steps 60 --resume` |
| shuffled-sequence LoRA adapter | `reports/adapters/lora_shuffled_seq/` | trained adapter output control | `python scripts/run_live_tool_dagger.py --root /workspace/experiments/qwen35_4b_live_tool_dagger --generate-traces --train-shuffled-lora --max-repairs 2 --max-steps 60 --resume` |

Small evaluation summaries, predictions, figures, logs, and reports remain in git.

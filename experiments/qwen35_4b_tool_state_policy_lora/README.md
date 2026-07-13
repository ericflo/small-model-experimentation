# qwen35_4b_tool_state_policy_lora

**Status:** finished

Standalone experiment for tool-state action-policy learning.

The package uses precomputed Qwen3.5-4B table-transformation traces. Each trace contains:

- a direct JSON answer,
- an executable `transform(table)` program attempt,
- visible-example execution observations,
- repair-loop observations,
- hidden held-out exactness labels used only for training labels and evaluation.

The policy sees only deployable tool-state observations and chooses:

- `DIRECT`: commit the direct JSON table,
- `PROGRAM`: commit the final visible-verified executable program output.

Run the fast non-neural baselines:

```bash
python scripts/run_tool_state_policy.py \
  --root /workspace/experiments/qwen35_4b_tool_state_policy_lora \
  --skip-lora
```

Run the LoRA action-policy arm:

```bash
python scripts/run_tool_state_policy.py \
  --root /workspace/experiments/qwen35_4b_tool_state_policy_lora \
  --train-lora \
  --max-steps 80
```

Outputs are written under `reports/`.

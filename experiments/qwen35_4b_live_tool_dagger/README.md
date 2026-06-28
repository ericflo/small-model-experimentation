# qwen35_4b_live_tool_dagger

Standalone live tool-state DAgger-style pilot.

The experiment generates fresh Qwen3.5-4B traces for a table transformation environment:

1. produce a direct JSON answer,
2. write an executable `transform(table)` program,
3. run the program on the public example,
4. repair when the public example fails,
5. decide whether to commit the direct output or the program output.

It derives oracle action labels from held-out correctness for training/evaluation analysis, but deployed policy inputs contain only visible tool state.

Run smoke:

```bash
python scripts/run_live_tool_dagger.py \
  --root /workspace/experiments/qwen35_4b_live_tool_dagger \
  --generate-traces \
  --limit-total 6 \
  --max-repairs 1
```

Run full pilot:

```bash
python scripts/run_live_tool_dagger.py \
  --root /workspace/experiments/qwen35_4b_live_tool_dagger \
  --generate-traces \
  --train-lora \
  --train-shuffled-lora \
  --max-repairs 2 \
  --max-steps 60 \
  --resume
```

Outputs are written under `reports/`.

# qwen35_4b_foofah_strategy_discovery_live

This standalone package tests strategy discovery for Foofah-style table transformations.

The experiment uses Qwen3.5-4B locally in two roles:

1. Propose reusable strategy prompts from calibration examples.
2. Generate fresh executable `transform(table)` programs on held-out tasks under the frozen discovered strategy prompts.

Run a smoke test:

```bash
python scripts/run_strategy_discovery.py \
  --root /workspace/experiments/qwen35_4b_foofah_strategy_discovery_live \
  --limit-test 6 \
  --max-discovered 2 \
  --max-repairs 0 \
  --overwrite
```

Run the full held-out evaluation:

```bash
python scripts/run_strategy_discovery.py \
  --root /workspace/experiments/qwen35_4b_foofah_strategy_discovery_live \
  --max-discovered 2 \
  --max-repairs 1 \
  --resume
```

Outputs are written under `reports/`.

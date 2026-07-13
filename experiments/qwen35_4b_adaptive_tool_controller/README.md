# qwen35_4b_adaptive_tool_controller

**Status:** finished

This standalone experiment evaluates a small adaptive controller over a fixed table-transformation candidate pool. Each task has a direct JSON attempt and five executable-program tool attempts. The controller decides whether to stop with the direct answer or spend additional forward-token budget on tool actions.

Run:

```bash
python scripts/eval_tool_controller.py \
  --root /workspace/experiments/qwen35_4b_adaptive_tool_controller
```

Outputs:

- `reports/final_summary.json`
- `reports/report.md`
- `reports/figures/*.png`
- `reports/decisions/*.json`

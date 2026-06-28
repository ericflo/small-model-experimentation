# qwen35_4b_foofah_adaptive_program_budget_router

This standalone package evaluates adaptive budget policies over a fixed Foofah-style table-transformation candidate pool. Each task has one direct JSON completion and five executable-program strategy completions. The experiment asks whether a cheap deployable router can decide when the expensive program portfolio is worth running.

Primary comparison:

- Direct JSON only.
- Fixed program budgets: prefixes of the five-strategy portfolio.
- Adaptive public-shape router selected on pilot data and frozen on test.
- Canary routers that pay for one program strategy before deciding whether to escalate.
- Nondeployable oracle diagnostics.

Run:

```bash
python scripts/eval_adaptive_router.py \
  --root /workspace/experiments/qwen35_4b_foofah_adaptive_program_budget_router
```

Outputs are written under `reports/`.

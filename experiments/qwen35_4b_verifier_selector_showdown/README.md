# Qwen3.5-4B Verifier vs Visible Selector Showdown

**Status:** finished

## Research Program

- Program: `evidence_conditioned_selection`
- Program question: is the model's own thinking-verifier (C10) worth its cost as a deployable selector,
  and does combining it with the visible test break the C2 false-pass wall?
- Prior anchors: `qwen35_4b_generator_verifier_gap` (C10: thinking-verifier is strong standalone),
  `qwen35_4b_thinking_budget_controller` (visible-test selector, bounded by false-passes).

## Question

On one k=8 candidate pool, compare deployable selectors head-to-head at matched cost: pass@1, visible-only,
no-think verifier, thinking verifier, visible+verifier, oracle. Does self-verification beat / complement the
visible test, and is thinking-verification worth its ~5x token cost?

## Setup

- Reuse the generator-verifier pool (100 MBPP tasks x k=8 no-think candidates, with full-test labels + the
  model's thinking/no-think black-box P(correct) already computed). Add a visible-test (first-assert) label
  per candidate. Fully offline (no new generation).
- Selectors pick one candidate per task -> its true full-test pass. Cost = estimated tokens/task (stated
  assumptions). See `analysis/analyze.py`.

## Run

```bash
../../.venv/bin/python scripts/build_pool.py    # add visible-test labels (torch-free)
../../.venv/bin/python analysis/analyze.py       # selectors + false-pass breakdown + figure
```

## Results

Full results in [reports/report.md](reports/report.md).

| selector (deployable) | accuracy | ~tok/task |
| --- | ---: | ---: |
| pass@1 | 0.771 | 120 |
| visible-only | 0.850 | 960 |
| thinking verifier | 0.860 | ~4960 |
| **visible + no-think verifier** | **0.870** | **960** |
| visible + thinking verifier | 0.870 | ~4960 |
| oracle pass@8 | 0.890 | — |

The **thinking verifier is Pareto-dominated**: standalone barely beats visible (0.860 vs 0.850) at ~5x cost,
and in combination the no-think verifier ties it (both 0.870). Best deployable = **visible + no-think verifier
(0.870, ~free)**, closing 83% of the pass@1->oracle gap. C2 false-pass rate here is only 6.6%.

## Interpretation

Refines C10: cheap self-verification (no-think) + the cheap visible test is the deployable sweet spot;
thinking-verification's expensive edge only matters in verifier-only settings (no cheap execution signal).
When a visible test exists, spend tokens on it + a free no-think verifier, not on thinking-verification.

## Artifacts

- `src/tasks.py`; `scripts/build_pool.py`; `analysis/analyze.py`. `data/pool.jsonl` + copied gv_records/labels;
  `runs/summary.json`, `analysis/selectors.png`. No external artifacts.

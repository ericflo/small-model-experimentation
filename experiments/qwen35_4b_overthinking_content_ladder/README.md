# Qwen3.5-4B Overthinking Content Ladder

**Status:** finished

## Research Program

- Program: `test_time_reasoning_budget`
- Program question: does the coherent-content advantage of thinking *shrink* as the thinking budget
  grows (overthinking)? i.e. is the residual "compute/scaffold, not reasoning" reading purely the
  high-budget regime?
- Prior anchors: `qwen35_4b_thinking_content_vs_compute` (at budget 512 the gain is 100% coherent
  content: filler ≈ shuffle ≈ no_think, real +12pp, foreign collapses) and `qwen35_4b_thinking_budget_scaling`
  (behavioral overthinking optimum ~1024; its 2048 shuffle ≈ real hinted coherence stops mattering).

## Question

The content ladder at budget 512 showed coherent reasoning is the entire thinking gain. The scaling
experiment showed accuracy *peaks* ~1024 then declines, and that at 2048 shuffled ≈ real. So: does the
coherence advantage `real − shuffle` fall toward 0 as the budget grows? If yes, the "thinking ≈
compute" reading is confirmed as the overthinking regime, and coherent reasoning is the efficient-budget
story — closing the program's central question across the budget axis.

## Hypothesis

`real − shuffle` is large at the efficient budget (~+0.12 at 512) and shrinks toward 0 at 2048. filler ≈
no_think and foreign collapses at every budget (the relevance/compute facts are budget-independent).

## Setup

- Model Qwen3.5-4B frozen (bf16, fast path). MBPP sanitized `test`, 100 tasks, k=8. **Behavioral-only**
  (no activations — the separability side was noisy and isn't the question here).
- Ladder at budgets {512, 1024, 2048}: real thinking generated once per budget (capturing tokens);
  filler (contentless `.` tokens, matched to real length), shuffle (permuted real tokens), foreign
  (a different task's tokens) regenerate only the answer. no_think once (budget-independent).
- Metrics: full-test pass per (budget, condition); the headline curve is `real − shuffle` vs budget,
  plus `real − foreign` (content-is-used) and filler ≈ no_think (compute ≈ 0) at each budget.

## Run

```bash
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ../../.venv/bin/python scripts/run.py --tasks 100 --k 8 --budgets 512,1024,2048
../../.venv/bin/python scripts/verify.py
../../.venv/bin/python analysis/curve.py
```

## Results

Full results in [reports/report.md](reports/report.md). Coherence advantage (real − shuffle) vs budget:

| budget | no_think | filler | shuffle | foreign | real | coherence |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.757 | 0.724 | 0.751 | 0.041 | 0.856 | **+0.105** |
| 1024 | 0.757 | 0.738 | 0.734 | 0.030 | 0.841 | **+0.108** |
| 2048 | 0.757 | 0.745 | 0.721 | 0.033 | 0.871 | **+0.150** |

The coherence advantage **grows** with budget (+0.105 → +0.150), refuting the "overthinking washes out
coherence" hypothesis — scrambling a longer thinking region hurts more (shuffle drops) while real holds.
At every budget: filler ≈ no_think (pure compute ≈ 0), foreign catastrophic, real is the entire gain.

## Interpretation

Coherent reasoning is the entire thinking gain at **every** budget, and matters more as the budget
grows. This retires the "thinking ≈ compute/scaffold" reading (which had survived as a high-budget
caveat) and shows the scaling experiment's "2048 shuffle ≈ real" was a shuffle-protocol artifact. See
the corrected claim C9.

## Artifacts

- `src/ladder_lib.py`, `src/tasks.py`; `scripts/run.py` (multi-budget, behavioral-only), `scripts/verify.py`;
  `analysis/curve.py`. `data/records.jsonl`, `data/labels.jsonl`, `data/tasks.json` (small, in-repo).
- No external artifacts (behavioral-only; no activations cached).

# Qwen3.5-4B Overthinking Content Ladder Experiment Log

## Scaffold

Fifth experiment of `test_time_reasoning_budget`. Runs the content ladder (no_think/filler/shuffle/
foreign/real) across thinking budgets {512,1024,2048} to test whether the coherent-content advantage
(real - shuffle) shrinks as the budget grows (overthinking). Behavioral-only (no activations).

## Method

Real thinking generated once per budget (gen_real captures thinking tokens); filler/shuffle/foreign
reuse those tokens / matched length and regenerate only the answer (gen_answer). no_think once.
Headline curve: real - shuffle vs budget. Reuses ladder_lib + tasks from the content_vs_compute experiment.

## Smoke

4 tasks x k=2, budget 512: ladder generated + verified end-to-end.

## Run notes

- The full sweep crashed mid-2048 (`CUDA device not ready` in the fla kernel — answer-regen at batch 48
  over ~2000-token thinking prefixes). Fixed with budget-scaled batch sizes; the 2048 arm was recovered
  via `scripts/add_2048.py`. Lesson: batch ~24-32 clears the memory error; batch 12 (my over-correction)
  turned it into a slow ~2.4h recovery — pick the batch just below the failure point, not far below.

## Results (see reports/report.md)

Coherence advantage (real − shuffle) grows with budget: +0.105 (512) → +0.108 (1024) → +0.150 (2048).
At every budget filler ≈ no_think (0.72–0.75 vs 0.757; pure compute ≈ 0), foreign catastrophic
(0.03–0.04; the model follows misleading content to the wrong problem), real is the entire gain
(0.84–0.87). REFUTES the "overthinking washes out coherence" hypothesis and the "compute reading holds at
high budgets" caveat; the scaling experiment's 2048 shuffle ≈ real was a shuffle-protocol artifact.
Corrected conclusion: coherent reasoning is the entire thinking gain at every budget.

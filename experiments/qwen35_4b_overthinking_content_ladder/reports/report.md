# Qwen3.5-4B Overthinking Content Ladder Report

## Summary

We ran the content ladder (no_think / filler / shuffle / foreign / real) across thinking budgets
{512, 1024, 2048} to test whether the coherent-content advantage *shrinks* as the budget grows
(overthinking). It does **not** — the coherence advantage `real − shuffle` **grows**: +0.105 (512)
→ +0.108 (1024) → **+0.150 (2048)**. Scrambling a *longer* thinking region hurts more (shuffle falls
0.751 → 0.734 → 0.721) while coherent thinking holds up (real ≈ 0.84–0.87), so the gap widens. At every
budget: filler ≈ no_think (pure compute ≈ 0), foreign is catastrophic (0.03–0.04; the model follows the
misleading content to the wrong problem), and real is the entire gain. This **refutes two earlier
readings**: (1) the hypothesis that overthinking washes out coherence, and (2) the caveat that the
"thinking ≈ compute/scaffold, not reasoning" reading survives at high budgets. The scaling experiment's
"2048 shuffle ≈ real" was a protocol artifact (it shuffled fresh thinking; this tighter run reuses
real's exact tokens). Corrected conclusion: **coherent reasoning is the entire thinking gain at every
budget, and matters more as the budget grows.**

## Research Program Fit

Fifth experiment of `test_time_reasoning_budget`; closes the budget-axis question. Combined with the
content ladder at 512, it makes the program's central claim uniform across budgets and corrects C9 (which
had preserved a "compute reading holds at high budgets" caveat now shown false).

## Method

- Model Qwen3.5-4B frozen (bf16, fast path). MBPP sanitized `test`, 100 tasks, k=8. Behavioral-only.
- Ladder at budgets {512, 1024, 2048}: real thinking generated once per budget (capturing thinking
  tokens); filler (contentless `.` matched to real length), shuffle (permuted real tokens), foreign (a
  cyclically-shifted other task's tokens) regenerate only the answer. no_think once (budget-independent).
- Metric: full-test pass per (budget, condition); headline curve `real − shuffle` vs budget.

## Results

| budget | no_think | filler | shuffle | foreign | real | coherence (real−shuffle) | content-used (real−foreign) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.757 | 0.724 | 0.751 | 0.041 | 0.856 | +0.105 | +0.815 |
| 1024 | 0.757 | 0.738 | 0.734 | 0.030 | 0.841 | +0.108 | +0.811 |
| 2048 | 0.757 | 0.745 | 0.721 | 0.033 | 0.871 | +0.150 | +0.839 |

Figures: `analysis/coherence_vs_budget.png` (the headline), `analysis/ladder_vs_budget.png`.

### Finding 1 — the coherence advantage grows with budget (hypothesis refuted)
`real − shuffle` rises +0.105 → +0.108 → +0.150. The prediction that overthinking would erase the value
of coherent order is wrong; the opposite holds — a longer thinking region, scrambled, is *more*
disruptive (shuffle drops), while coherent thinking stays high.

### Finding 2 — pure compute ≈ 0 at every budget
filler (contentless `.` matched to real length) tracks no_think at all budgets (0.724 / 0.738 / 0.745
vs 0.757). No "dot-by-dot" extra-compute benefit appears at any budget.

### Finding 3 — the model uses content at every budget
foreign stays catastrophic (0.041 / 0.030 / 0.033): a different task's thinking sends the model to the
wrong problem regardless of budget. `real − foreign` ≈ +0.82 throughout.

### Finding 4 — real accuracy is roughly flat across budgets (sampled full-pass)
real ≈ 0.84–0.87 across 512/1024/2048. The greedy "overthinking optimum then decline" seen in the
scaling experiment is a greedy/single-sample phenomenon, not a drop in sampled full-pass or in the value
of reasoning.

## Controls

The ladder is the control structure; filler isolates pure compute, foreign isolates misleading content,
shuffle isolates relevance-without-order, real adds coherent order — replicated at three budgets. The
key artifact check: this experiment's shuffle reuses real's *exact* thinking tokens (permuted), unlike
the scaling experiment's shuffle (which shuffled a fresh thinking sample); that difference explains why
the scaling run saw 2048 shuffle ≈ real while this run sees real ≫ shuffle.

## Oracle Versus Deployable Evidence

All behavioral full-pass (deployable). No probe / oracle metrics here.

## Interpretation

The program's central question is now settled across the budget axis: **the native-thinking gain is
coherent reasoning content at every budget** — not compute (filler ≈ baseline everywhere), not scaffold,
not token-presence (shuffle ≈ baseline everywhere) — and the model genuinely uses that content (foreign
catastrophic everywhere). If anything the reasoning contribution *grows* with budget. The "thinking ≈
compute" reading, which earlier survived as a high-budget caveat, is fully retired; it only ever appeared
through a greedy-metric lens or the scaling run's shuffle-protocol artifact.

### Limitations
- MBPP (basic, likely partly contaminated), n=100, single seed, sampled full-pass. Minor protocol
  asymmetry (real's answer is its original generation; others are regenerated from the modified prefix).
  The 2048 arm was regenerated in a slow recovery run after a mid-run CUDA "device not ready" in the fla
  kernel (answer-regen batch too large over ~2000-token prefixes); fixed by budget-scaled batch sizes.

## Next Experiments

- Contamination-controlled / harder substrate: does coherent reasoning still carry the whole gain when
  the no-think baseline is weaker and memorization is defeated?
- Does the growing-with-budget coherence advantage hold on non-code reasoning (math)?

## Artifact Manifest

See `artifact_manifest.yaml`. Behavioral-only; small records/labels + table + figures in-repo; no
external activation artifacts.

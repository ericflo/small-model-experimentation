# Qwen3.5-4B Thinking Content vs Compute Report

## Summary

A full content/compute decomposition of the native-thinking gain — and the **definitive correction**
of the earlier "much of the gain is compute/scaffold, not reasoning." On a tight ladder where all
thinking conditions share matched thinking length (and the relevant ones share the same thinking-token
multiset), MBPP full-test pass is:

| no_think | filler | shuffle | real | foreign |
| ---: | ---: | ---: | ---: | ---: |
| 0.749 | 0.744 | 0.739 | **0.861** | 0.040 |

**filler ≈ shuffle ≈ no_think, real is +12pp above all of them, and foreign collapses to 4%.** So at
the efficient (512) budget, **100% of the behavioral thinking gain is coherent reasoning content**:
pure forward compute (contentless `.` filler) buys ~nothing (−0.005), relevant-but-scrambled tokens
buy ~nothing (−0.005), coherent order/content is the entire +0.122, and *misleading* content (another
task's thinking) is catastrophic (−0.709) because the model **follows it to the wrong problem**.

## Research Program Fit

Fourth experiment of `test_time_reasoning_budget`; the decisive content control + filler arm. It
**revises C9**: the earlier "mostly compute/scaffold" reading was a greedy-metric artifact that held
mainly at high budgets (2048 shuffle ≈ real, overthinking) and at the representational level; the
filler arm shows pure compute contributes ~0, so at the efficient budget the gain is reasoning content.

## Method

- Model Qwen3.5-4B frozen (bf16, fast path). MBPP sanitized `test`, 100 tasks, k=8, thinking budget 512.
- Real thinking generated once (capturing its thinking tokens). The other arms reuse those tokens /
  matched length and regenerate ONLY the answer from the modified prefix:
  - **filler**: that example's real-thinking *length* worth of contentless `.` tokens (pure compute + scaffold).
  - **shuffle**: the real thinking tokens, permuted (relevant content, scrambled order).
  - **foreign**: a cyclically-shifted *other* task's thinking tokens (misleading content).
  - **real**: the model's own coherent thinking. **no_think**: enable_thinking=False.
- Behavioral full-test pass + per-layer answer-token separability probe (right-padded activations,
  GroupKFold-by-task logistic, bootstrap CI, shuffled-label control).

## Results

| rung | full-pass | visible-pass | probe AUC | isolates |
| --- | ---: | ---: | ---: | --- |
| no_think | 0.749 | 0.806 | 0.646 | baseline |
| filler | 0.744 | 0.789 | 0.703 | pure compute + scaffold (contentless `.`) |
| shuffle | 0.739 | 0.781 | 0.645 | relevant tokens, scrambled order |
| real | 0.861 | 0.902 | 0.722 | relevant tokens, coherent order |
| foreign | 0.040 | 0.041 | 0.987\* | misleading content (another task's thinking) |

\* foreign probe AUC is a class-imbalance artifact (33/800 passes).

Behavioral attribution (additive ladder no_think → filler → shuffle → real):
- pure compute + scaffold (filler − no_think): **−0.005**
- token-presence / relevance (shuffle − filler): **−0.005**
- coherent order / content (real − shuffle): **+0.122**
- total (real − no_think): +0.112
- [off-ladder] misleading content (foreign − no_think): **−0.709**

### Finding 1 — pure compute buys nothing
Contentless `.` filler (matched to each example's real thinking length) lands at 0.744 ≈ no_think
0.749. The "dot-by-dot" extra-compute effect does not appear for this 4B on MBPP; the `<think>`
scaffold + extra forward passes alone add nothing.

### Finding 2 — the model uses thinking as CONTENT
Foreign thinking collapses full-pass to 0.040 — far below no-think — by *following the foreign
reasoning to the wrong problem* (verified: task `remove_Occ` fed a matrix-sort thought emits
`sort_matrix`). The model conditions its answer on the thinking content, so irrelevant content is
actively harmful, not inert.

### Finding 3 — only coherent content helps; it is the entire gain
Scrambled relevant thinking (shuffle 0.739) ≈ filler ≈ no_think — token-presence without coherent
order buys nothing on sampled full-pass. Coherent order (real 0.861) adds +12pp and accounts for the
whole gain. So at the efficient budget the thinking benefit is genuine coherent reasoning content.

### Finding 4 — representational side is noisy
Best-layer separability AUCs (no_think 0.646, filler 0.703, shuffle 0.645, real 0.722) have heavily
overlapping CIs; point estimates hint filler ≈ real > no_think ≈ shuffle but this is within noise. The
foreign AUC is a degenerate artifact. Only the behavioral ladder is robust.

## Controls

The ladder *is* the control structure: filler isolates pure compute, foreign isolates misleading
content, shuffle isolates relevance, real adds coherent order. Shuffled-label probes ≈ 0.44–0.56
(no leakage). The complete attribution leaves no unexplained component — earlier the missing piece was
filler (the only contentless arm); it is now run and ≈ 0.

## Oracle Versus Deployable Evidence

Behavioral full/visible-pass are deployable; the probe is a non-deployable, here-inconclusive
decodability diagnostic.

## Interpretation

This conclusively overturns the program's earlier sharpest claim. At the efficient budget on MBPP, the
thinking accuracy gain is **entirely coherent reasoning content** — not compute (filler ≈ baseline),
not scaffold, not token-presence (shuffle ≈ baseline) — and the model genuinely uses that content
(foreign → wrong problem). The "thinking ≈ compute/scaffold" reading survives only as the high-budget
regime (the scaling experiment's 2048 shuffle ≈ real overthinking result) and as the noisy
representational slice. The honest, complete picture for the program: thinking is a real deployable
lever (scaling), cheaply routable (controller), and — at the efficient budget — genuine reasoning the
model uses (this experiment), whose advantage washes out under overthinking and is not clearly mirrored
in internal correctness-decodability.

### Limitations
- Efficient budget (512) only; the coherence advantage is expected to shrink at high budgets
  (overthinking — confirm with a high-budget ladder). MBPP is basic, likely partly contaminated, n=100
  single seed. Minor protocol asymmetry: real's answer is its original generation; filler/shuffle/foreign
  answers are regenerated from the modified prefix (same prefix → same answer distribution).

## Next Experiments

- High-budget (1024/2048) ladder to confirm the coherence advantage shrinks under overthinking.
- Contamination-controlled / harder substrate where the no-think baseline is weaker (more headroom).
- A learned controller using internal/uncertainty signals vs the visible-test C2 wall.

## Artifact Manifest

See `artifact_manifest.yaml`. Activations (~0.65 GB, 5 conditions) external/regenerable; records, labels,
decomposition, probe results, and figure in-repo.

## Refinement (added after the budget sweep)

This report says the coherence advantage is "expected to shrink at high budgets (overthinking)" and that
the compute reading survives as the high-budget regime. The follow-up budget sweep
([`qwen35_4b_overthinking_content_ladder`](../../qwen35_4b_overthinking_content_ladder/reports/report.md))
**refuted** this: the coherence advantage `real − shuffle` does not shrink but **grows** with budget
(+0.105 → +0.108 → +0.150 at 512/1024/2048), and pure-compute filler ≈ no-think at every budget. So the
gain is coherent reasoning content at *all* budgets; the "compute" reading only ever appeared through a
greedy-metric lens and the scaling run's shuffle-protocol artifact. See claim C9 (corrected).

# Qwen3.5-4B Thinking Content vs Compute Report

## Summary

A foreign-task-thinking control completes the decomposition of the native-thinking gain — and
**corrects an earlier overstatement** that "much of the gain is compute/scaffold, not reasoning."
On a tight ladder where all conditions share the *same* thinking-token multiset (so only relevance
and order vary), behavioral full-test pass is **no_think 0.764 → foreign 0.043 → shuffle 0.739 →
real 0.859**. Foreign thinking (another task's reasoning spliced in) **collapses accuracy to 4%**:
the model *follows the foreign reasoning to the wrong problem* (verified — a string task fed a
matrix-sort thought emits `sort_matrix`). So the model genuinely **uses thinking as content**, not as
a content-free compute/format crutch. Relevant-but-scrambled thinking ≈ no thinking (shuffle 0.739 ≈
no_think 0.764), and coherent order adds **+12pp** (real 0.859). So at the efficient budget the
behavioral thinking gain **is coherent reasoning over relevant content**. The representational side
(per-layer separability probe) is noisy here — among comparable conditions AUCs are ~0.64–0.68 with
overlapping CIs (no clear ordering), and the foreign AUC of 0.99 is a degenerate class-imbalance
artifact (34/800 passes) — so this experiment's clean, robust result is behavioral.

## Research Program Fit

Fourth experiment of `test_time_reasoning_budget`, and the decisive content control the separability
report flagged. It **revises C9**: the model uses thinking as content (foreign is catastrophic), and
the efficient-budget accuracy gain is genuine coherent reasoning — the earlier "mostly compute/
scaffold" reading was a greedy-metric artifact (greedy shuffle recovered ~⅓; sampled shuffle recovers
~0), held mainly at high budgets (2048 overthinking), and was conflated with the noisy decodability
finding.

## Method

- Model Qwen3.5-4B frozen (bf16, fast path). MBPP sanitized `test`, 100 tasks, k=8, thinking budget 512.
- Real thinking generated ONCE (capturing its thinking tokens); **shuffle** permutes those tokens,
  **foreign** uses a cyclically-shifted *other* task's thinking tokens (same sample slot); both
  regenerate ONLY the answer from the modified prefix. So all conditions share one thinking-token
  multiset and matched thinking length — only relevance (shuffle vs foreign) and order (real vs
  shuffle) differ. Behavioral full-pass + per-layer answer-token separability probe.

## Results

Behavioral ladder (full-pass, n=100, 800/cond) + separability (best-layer probe AUC):

| rung | full-pass | visible-pass | probe AUC | what it adds vs the rung below |
| --- | ---: | ---: | ---: | --- |
| no_think | 0.764 | 0.802 | 0.682 | (baseline) |
| foreign | 0.043 | 0.043 | 0.994\* | irrelevant thinking content |
| shuffle | 0.739 | 0.776 | 0.636 | relevance (relevant tokens, scrambled) |
| real | 0.859 | 0.895 | 0.676 | coherent order |

\* foreign probe AUC is a degenerate artifact (34/800 passes; "wrong-problem" answers are trivially
separable from the rare correct ones) — not comparable.

Behavioral decomposition (full-pass deltas):
- adding **irrelevant** thinking (foreign − no_think): **−0.721** (catastrophic; the model follows it)
- making it **relevant** (shuffle − foreign): **+0.696** (recovers to ≈ baseline)
- adding **coherent order** (real − shuffle): **+0.120**
- total (real − no_think): +0.095

### Finding 1 — the model uses thinking as CONTENT (foreign collapses)
Foreign thinking drops full-pass from 0.764 to 0.043 — far below no-think. Spot-check: task `remove_Occ`
(string) fed another task's matrix-sort thinking emits `def sort_matrix(...)`. The model conditions its
answer on the thinking content, so irrelevant reasoning yields the wrong-problem answer. Thinking is
not a content-free compute/format crutch.

### Finding 2 — relevance is necessary, coherent order is the behavioral gain
Relevant-but-scrambled thinking ≈ no thinking (shuffle 0.739 ≈ no_think 0.764): on sampled full-pass,
scrambling removes ~all the benefit. Coherent order then adds +12pp (real 0.859). So at the efficient
512 budget the accuracy gain **is coherent reasoning over relevant content**.

### Finding 3 — representational side is noisy here
Excluding the foreign artifact, separability AUCs (no_think 0.682, shuffle 0.636, real 0.676) sit in a
~0.64–0.68 band with overlapping CIs — no clear ordering, and somewhat at odds with the prior
separability experiment (which found thinking > no-think, shuffle ≥ real). The decodability differences
across thinking conditions are small and protocol-sensitive; only the behavioral ladder is robust.

## Controls

Shuffled-label probes ≈ 0.47–0.52 (no leakage). The ladder is the control structure itself: foreign
isolates *irrelevant content*, shuffle isolates *relevance*, real isolates *coherent order*. NOTE:
foreign is not a clean "pure compute" arm — it adds *misleading* content (which the model follows),
not contentless compute; a **filler/blank-token arm** (pause tokens, no semantics) is still needed to
isolate pure forward compute.

## Oracle Versus Deployable Evidence

Behavioral full-pass and visible-pass are deployable; the probe is a non-deployable decodability
diagnostic (and inconclusive here).

## Interpretation

This corrects the program's sharpest earlier claim. Across the four experiments the honest picture is:
thinking is a real deployable lever (scaling) that is cheaply routable (controller); at the **efficient
budget its accuracy gain is genuine coherent reasoning over relevant content** — irrelevant thinking is
catastrophic because the model *follows* it, and scrambled thinking ≈ no thinking (this experiment).
That coherence advantage **washes out at high budgets** (2048 shuffle ≈ real — overthinking, from the
scaling experiment) and is **not clearly reflected in internal correctness-decodability** (small/noisy
probe differences here; shuffle ≥ real in the separability experiment). So "thinking ≈ compute, not
reasoning" was too strong: it holds for the high-budget and representational slices, but the
efficient-budget behavioral gain is reasoning.

### Limitations
- One model, one benchmark, n=100 single seed, budget 512 only. Foreign separability AUC is an
  imbalance artifact. Minor protocol asymmetry: real's answer is its original generation while
  shuffle/foreign answers are regenerated from the modified prefix (same prefix → same answer
  distribution, so minor).
- foreign ≠ pure compute (it is *misleading* content); the filler-token arm is the remaining piece.

## Next Experiments

- **Filler/pause-token arm** (B contentless tokens, no semantics) to isolate pure compute vs the
  content effects measured here — completes the {compute, presence, relevance, order} attribution.
- Repeat the ladder at a high budget (1024/2048) to confirm the coherence advantage shrinks (overthinking).
- Contamination-controlled / harder substrate where the no-think baseline is weaker.

## Artifact Manifest

See `artifact_manifest.yaml`. Activations (~0.5 GB) external/regenerable; records, labels, decomposition,
probe results, and figure in-repo.

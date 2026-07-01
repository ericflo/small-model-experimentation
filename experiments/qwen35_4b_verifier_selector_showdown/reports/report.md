# Qwen3.5-4B Verifier vs Visible Selector Showdown Report

## Summary

Head-to-head **deployable** selection signals on one k=8 MBPP candidate pool, at matched cost: does the
model's own thinking-verifier (claim C10) beat / break the visible-test false-pass wall the earlier
thinking controller was bounded by? The matched-cost lens **tempers C10**. The visible test alone is
already a strong, nearly-free selector (0.850, from pass@1 0.771 toward oracle 0.890). The **thinking
verifier is Pareto-dominated**: standalone it reaches 0.860 (barely above visible's 0.850) at **~5× the
token cost**, and in combination the **no-think verifier ties it** (both 0.870) — once the cheap visible
test does the coarse filtering, ranking the survivors is easy enough that no-think verification suffices.
**Best deployable = visible + no-think verifier: 0.870 at ~free cost**, closing 83% of the pass@1→oracle
gap by catching false-passes. So thinking-verification's expensive standalone edge (the big win in the
generator-verifier experiment) **evaporates whenever a cheap visible signal exists** — its value is
confined to verifier-only settings.

## Research Program Fit

Follows up `qwen35_4b_generator_verifier_gap` (C10) and `qwen35_4b_thinking_budget_controller` under
`evidence_conditioned_selection`. It supplies the matched-cost, deployable comparison the C10 result begged
for, and answers whether self-verification is worth its token cost as a selector.

## Method

- Reuse the generator-verifier candidate pool (100 MBPP tasks × k=8 no-think candidates, with full-test
  execution labels + the model's thinking/no-think black-box verifier P(correct) already computed). Add a
  fresh **visible-test** label per candidate (passes the first assert — the deployable signal the controller
  used). All offline.
- Deployable selectors (per task pick one candidate → its true full-test pass): pass@1 (random single);
  visible-only (first visible-passer); no-think / thinking verifier (max P(A)); visible+verifier (among
  visible-passers, max P(A)); oracle pass@k (non-deployable ceiling).
- Cost: estimated tokens/task (no-think candidate ~120; thinking-verification ~500; visible test / no-think
  verification ≈ 0 extra generation) — stated assumptions, for the Pareto picture, not measured per-item.

## Results

| selector (deployable) | accuracy | ~tokens/task | gap closed (pass@1→oracle) |
| --- | ---: | ---: | ---: |
| pass@1 (random single) | 0.771 | 120 | — |
| visible-only (first visible-pass) | 0.850 | 960 | 66% |
| no-think verifier (max P_A) | 0.800 | 960 | 24% |
| thinking verifier (max P_A) | 0.860 | ~4960 | 75% |
| **visible + no-think verifier** | **0.870** | **960** | **83%** |
| visible + thinking verifier | 0.870 | ~4960 | 83% |
| ORACLE pass@8 (non-deployable) | 0.890 | 960 | 100% |

C2 false-pass wall: visible-pass rate 0.818; **6.6%** of visible-passers full-fail (43 false-passes). Among
visible-passers, the verifier ranks true>false at AUROC 0.758 (think) / 0.701 (no-think). Figure: `analysis/selectors.png`.

### Finding 1 — the visible test is already a strong, nearly-free selector
Visible-only reaches 0.850 (66% of the oracle gap) at ~0 extra tokens. The C2 false-pass rate on this pool
is modest (6.6% of visible-passers), so there is limited headroom for any verifier to add.

### Finding 2 — the thinking verifier is Pareto-dominated
Standalone thinking-verifier 0.860 barely beats visible-only 0.850, at ~5× the token cost. Combined,
visible+thinking (0.870) equals visible+no-think (0.870): the thinking verifier's large standalone edge over
no-think (0.860 vs 0.800) **disappears** once the visible test pre-filters, because ranking visible-passers
is an easier discrimination that no-think verification handles nearly as well (AUROC 0.701 vs 0.758).

### Finding 3 — best deployable selector = visible + no-think verifier (0.870, ~free)
It beats visible-only by +2pp (catching false-passes the first-visible-passer rule commits) at the same
cost, closing 83% of the pass@1→oracle gap. Thinking-verification is not worth its 5× token cost here.

## Controls

Same pool for every selector (paired). The +2pp of visible+verifier over visible-only is small (~2 tasks at
n=100, within per-condition noise) but paired and mechanistic (it reranks visible-passers to avoid
false-passes). The oracle is the non-deployable ceiling.

## Oracle Versus Deployable Evidence

All selectors except the oracle read only visible info (visible test + the model's own black-box judgment),
so their accuracies are deployable. pass@8 = 0.890 is the non-deployable ceiling.

## Interpretation

This refines C10. Self-verification does help selection, but **cheap self-verification (no-think) combined
with the cheap visible test is the deployable sweet spot** — thinking-verification's expensive edge only
matters in verifier-only settings (no cheap execution-based signal), which is exactly the regime the
generator-verifier experiment measured. The durable lesson: when a cheap ground-truth-ish signal (a visible
test) exists, spend tokens on it plus a free no-think verifier, not on expensive thinking-verification; the
matched-cost lens flips the standalone ranking.

### Limitations
- MBPP (basic, likely contaminated), n=100, single seed; the modest C2 false-pass rate (6.6%) caps verifier
  headroom on this pool — a harder pool with more false-passes could favor the (stronger) thinking verifier.
- Token costs are estimates (stated assumptions), not measured per-item; the accuracy ranking is exact.
- Not run: the matched-cost-vs-more-sampling arm (spend the ~4000 verification tokens on more candidates to
  raise coverage) — but visible+no-think-verifier already Pareto-dominates the thinking verifier at ~5× less cost.

## Next Experiments

- Harder / contamination-controlled pool with a higher false-pass rate — does the thinking verifier's edge
  survive when the visible test is weaker?
- Matched-cost vs more-sampling: is any verifier worth its tokens vs just raising k?

## Artifact Manifest

See `artifact_manifest.yaml`. Offline; small pool/summary + figure in-repo; candidate pool reused from
qwen35_4b_generator_verifier_gap.

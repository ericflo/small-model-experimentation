# Do banking + thinking stack? Additively on recognition, not at all on planning

## Summary (honest scope up front)

This experiment asks whether the two capability levers — **banking** (C25: no-think SFT that lifts step-wise
next-op ranking at lookahead distance) and **test-time thinking** (C26: amplifies recognition) — compose. It is
a **narrow baseline**, not a bold "stacking" result, and it carries an important **scope caveat surfaced by the
user**: the banked adapter (C24) was trained **no-think** (prompt→code, no reasoning traces), so any statement
that "test-time thinking adds no planning even to the banked model" is about *test-time* thinking on a
*no-think-trained* model — **not** evidence that thinking is fundamentally useless for planning. That clean
question is the motivation for the **bank-the-thoughts** follow-up.

## Result: 2×2 {base, banked} × {no-think, think}, per-step next-op top-1 (n=40, chance 0.031)

| step | base+no-think | base+think(2048) | banked+no-think | banked+think(2048) |
|---|---|---|---|---|
| **step 1** (planning, goal 3 away) | 0.025 | 0.075 | 0.175 | 0.150 |
| step 2 (2 away, state given) | 0.000 | 0.325 | 0.150 | 0.250 |
| **step 3** (recognition, goal 1 away) | 0.275 | 0.600 | 0.525 | **0.850** |

- **Recognition (step-3): the levers STACK almost exactly additively.** Banking adds +0.25, thinking adds
  +0.325, and banked+think = 0.850 = the additive prediction (0.275 + 0.25 + 0.325); interaction ≈ 0.00. The two
  levers act on the recognition axis independently and compose.
- **Planning (step-1): no stacking.** Banking lifts (0.025 → 0.175) but test-time thinking adds ~nothing to base
  (0.025 → 0.075) *or* banked (0.175 → 0.150). The thinking effect on planning is ~0 (and not significant at
  n=40; the step-1 CIs are wide and overlapping). So **banking owns the planning lift; test-time thinking does
  not contribute to it.**
- The banked model's **thinking channel is intact** (traces coherent, median ~2500 chars, even showing
  backward-from-goal reasoning) — so the banked+think cell is a fair thinking condition, not a broken one.

## Interpretation (scoped)

Test-time thinking and answer-banking are **orthogonal on recognition** (additive) and **non-interacting on
planning** (thinking contributes 0 to planning regardless of banking). This is consistent with the arc: banking
installs the lookahead-distance ranking lift (C25); test-time thinking amplifies recognition (C26) and stacks
there. But it says **nothing** about whether *training the model to reason* (banking successful thinking traces)
would install planning-via-thinking — because this model was only ever trained to emit answers. **That is the
open question, and the reason this experiment is a baseline, not a conclusion.**

## Method

2×2 on per-step next-op RANKING (think→RANK, channel-matched to C25/C26, parse-immune). banked_1280 cells
measured in-run at budgets {0,1024,2048}; base cells inherited from C26 (identical n=40 first-40 slice, same
`think_rank` harness, same frozen held-out — verified byte-identical). `scripts/run_thinking.py` (with
`--adapter`), `scripts/analyze.py`.

## Honest limits

n=40, one seed/budget — step-1 CIs are wide (the "no stacking on planning" is "thinking effect ≈ 0 and not
significant", not a tight null). Base cells inherited from C26 (same slice/harness, so valid, but not
re-run here). Closed-set ranking is easier than free generation. The end-to-end thinking-guided search was NOT
run: the per-step 2×2 already shows step-1 planning is unmoved by thinking, so a search gated by the first move
has no mechanism to beat banked-no-think — a pre-registered null (per the design review).

## Next: bank-the-thoughts (the experiment this motivates)

Rejection-sample verified-correct depth-3 thinking traces from the banked model, SFT on
`prompt → ⟨planning trace⟩ → code`, and test whether that installs planning the model can then *use* (deployable
depth / coverage that stacks with multi-sampling). This is the clean separation of "weights can't plan" vs
"never taught to think-to-plan" that the present test cannot make.

## Artifact Manifest
See `reports/artifact_manifest.yaml`. Reuses C24 banked_1280 adapter (out of repo); no training here.

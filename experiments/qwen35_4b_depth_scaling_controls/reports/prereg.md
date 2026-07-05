# Pre-registration: depth scaling & controls (saturation, data-vs-compute, depth-4)

Logged 2026-07-04, before eval data. Three arms extending C23 (depth-3 install is data-limited: think cov@16
0.00→0.087→0.212→0.375 at N=40/160/640). Design hardened by an adversarial workflow review
(`reports/design_review.md`). All on the list 16-op DSL, interpreter brute-search explorer (no external
model), QLoRA r32/α64 epochs=3, frozen paired held-out deduped by function-signature AND op-composition
(0 leakage verified: depth-3 0/2305, depth-4 0/318).

## Arm 1 — Saturation (extend past 640)
Doses 640/1280/2560 depth-3 tool pairs (NESTED: first 640 = C23's, verified; 555/1156/2305 DISTINCT functions
respectively — so a flat curve is real saturation, not harvest-diversity exhaustion). Reuse C23 base/40/160/640
evals on the same frozen held-out. Metrics: think cov@16 (Wilson CIs) + per-sample rate + greedy@1.
- **P1:** does think cov@16 keep rising past 0.375 (640) at 1280/2560, or plateau (top-two doses' CIs overlap)?

## Arm 2 — Data-diversity vs compute (upsampled-40 control)
banked_up40 = depth-1+2 + the NESTED first-40 depth-3 pairs each ×16 (=640 examples), same size/mixture/steps
as train_640 (epochs=3 → 1920 depth-3 example-visits both). 2×2: N=40 dose (40 distinct, 120 visits, C23=0.087)
vs up40 (40 distinct, 1920 visits) vs train_640 (640 distinct, 1920 visits, C23=0.375).
- **P2:** up40 vs 640 (fixed compute): if 640 lower-CI > up40 upper-CI → DATA-DIVERSITY drives the gain; if
  overlapping → compute/mixture contributes. up40 vs N=40 (fixed diversity) isolates compute.

## Arm 3 — Depth-4 rung (does the recipe repeat?)
banked_d4 = depth-1+2 + 640 depth-3 (scaffold) + 320 depth-4 tool pairs (all verified TRUE length-4). Eval
depth-4 cov@16 (think, n=60) on a fresh 0-leak depth-4 held-out. **Baseline is the SCAFFOLD-ONLY model
(banked_640, no depth-4) on depth-4** — not raw base — since depth-3 skill may transfer. Depth-3 guardrail
(banked_d4 on depth-3 held-out) checks scaffold forgetting.
- **P3:** banked_d4 depth-4 cov@16 > scaffold_640 depth-4 cov@16 + significant → the recipe repeats one rung
  deeper (tool-search reaches depth-4, banking installs it). Refuted if ≈ scaffold (depth-4 doesn't install).

## Honest limits (from the review)
Single training seed at each point (Wilson CIs are eval-noise only; a plateau, a data-vs-compute tie, or a
depth-4 delta could be seed artifacts — ≥3 seeds deferred). Fixed epochs (data confounded with gradient
exposure except in the up40 control). Depth-4 is a single dose (installs-vs-installs-a-little not resolved).

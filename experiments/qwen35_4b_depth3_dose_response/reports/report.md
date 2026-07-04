# Depth-3 install is DATA-LIMITED, not a representational cap — and it scales into deployable single-shot

## Summary

C22 crossed the depth-3 wall with tool-seeded banking but only WEAKLY (think coverage@16 0.125, deployable
no-think ≈0). Was that a data limit or a representational cap? This dose-response answers it: **DATA-LIMITED,
decisively.** Banking N tool-found depth-3 pairs installs depth-3 coverage that rises monotonically with N and
does not plateau — and, at scale, banks into *deployable single-shot*.

| N distinct depth-3 pairs | think cov@16 (Wilson 95%) | think greedy@1 | no-think cov@16 | no-think greedy@1 |
|---|---|---|---|---|
| 0 (base) | 0.000 [0.00, 0.05] | 0.000 | 0.000 | 0.000 |
| 40 | 0.087 [0.04, 0.17] | 0.013 | — | — |
| 160 | 0.212 [0.14, 0.31] | 0.037 | — | — |
| 640 | **0.375 [0.28, 0.48]** | **0.138** | **0.338** | **0.100** |

- **Monotone rise, no plateau.** Think coverage@16 climbs 0.00 → 0.087 → 0.212 → 0.375 across N ∈ {0,40,160,640}.
  The top-dose lower CI (0.277) sits *above* the low-dose upper CI (0.17) — a clean, non-overlapping increase
  (not a bare non-significant p). Rise 40→640 = +0.29.
- **The deployable install scales too.** At N=640, no-think coverage@16 = 0.338 and **no-think single-shot
  greedy@1 = 0.10** — vs C22 (N=130) where deployable was ≈0 (test-time-only). So C22's "test-time-dominated"
  weakness was *also* just insufficient data; more tool-found solutions bank depth-3 into the weights.
- **Genuine generalization, not memorization.** The 80 held-out depth-3 tasks have **0 leakage** — none share a
  function-signature OR an op-composition with the 750 training rules — so every solved task is a NOVEL depth-3
  composition. This closes the finite-DSL memorization confound the review flagged as make-or-break.
- **Scaffold intact.** Depth-2 guardrail rose (base 0.175 → banked_640 0.50); the depth-1+2 decomposition
  scaffold did not collapse under the depth-3-heavy mixture.

## Research Program Fit

Resolves C22's central open question and, with it, the mission's "extend capability by a lot" hope: the
tool-seeded-banking recipe is not just directionally valid (C22) but **data-scalable** — more explorer-found
solutions → progressively more installed, deployable capability. Design hardened by an adversarial multi-agent
review (`reports/design_review.md`).

## Method

Substrate `list`, families' own 16-op DSL. No external model. Explorer = CPU interpreter brute-search
(640/640 depth-3 solved). Doses (NESTED, log-spaced): C21's exact depth-1+2 pairs + N tool-found depth-3
pairs, N ∈ {40,160,640}; QLoRA r32/α64, epochs=3 held constant. Eval on ONE frozen paired held-out set,
deduped vs the 640-superset by **function-signature AND op-composition** (0 leakage verified), n=80 depth-3.
Primary: think coverage@16 with Wilson 95% CIs + a dense per-sample solve rate. Deployable: no-think
coverage@16 + greedy@1 at the top dose. Guardrail: depth-2 think coverage.

## Pre-registered verdicts

- **P1 (monotone):** HELD — think coverage@16 strictly increases 0 → 0.087 → 0.212 → 0.375.
- **P2 (decision):** **DATA-LIMITED** — rise 40→640 = +0.29 (≫ +0.10) and top-dose lower CI (0.28) > low-dose
  upper CI (0.17), i.e. a clean non-overlapping increase. NOT a cap through N=640.
- **P3 (deployable):** HELD — at N=640 the depth-3 install reaches deployable no-think coverage 0.338 and
  single-shot greedy@1 0.10 (both ≈0 at C22's N=130); the deployable read scales with data too.
- **P4 (memorization control):** HELD — 0 eval solutions (function-sig or op-composition) in training; the rise
  is generalization to novel depth-3 rules.

## Interpretation

- **The deep wall is not a hard representational bottleneck at the depths tested — it is a data bottleneck.**
  C19 showed the base's depth-3 inverse is barely represented and C20 showed it is not steerable; C22 showed
  one small banking round installs only a thread. But this dose-response shows that thread *thickens smoothly
  with more tool-found training data* — through N=640 the model keeps absorbing more novel-depth-3
  compositional capability, and it converts into deployable single-shot. The apparent "cap" of C22 was a
  small-data artifact.
- **The complete, validated recipe (C13→C23), now quantitative:** to extend the frontier a depth — an explorer
  the base lacks (tool-search) reaches the rung and produces verified solutions; banking those installs the
  capability; and the amount installed *scales with the number of explorer-found solutions*, into deployable
  single-shot. Self-training is the installer, external search is the explorer, and data is the throttle.
- **Mission read:** this is "extend capability by a lot without a larger model" demonstrated and scaling — the
  only external ingredient is an interpreter-backed search (a tool), and everything installed is the fixed
  4B's own verified compositions.

## Honesty notes / limits (from the review)

- **Single training seed** (nested doses): the Wilson CIs capture *eval* noise only, not training/seed
  variance; a lone helpful/harmful example or data-order roll is not averaged out. ≥3 seeds would put error
  bars on the dose effect itself (deferred).
- **Fixed epochs**: "more depth-3 data" is physically the same event as "more depth-3 gradient exposure"; this
  study does not separate data-diversity from compute via a fixed-step + upsampled-40 control (deferred). The
  claim is the applied "more distinct tool-found pairs → more installed capability," not a pure information
  bound.
- **Search-easy bias**: brute search harvests the shorter/cheaper depth-3 programs first; the curve could cap
  on harder compositions past N=640 (untested). No plateau is observed *through* 640.

## Next Experiments

- Push doses past 640 (e.g. 1280/2560) to find where (if) the curve saturates, and characterize the
  program-length/search-difficulty of solved-vs-unsolved held-out tasks.
- Fixed-step + upsampled-40 control to split data-diversity from gradient exposure.
- Iterate the rung: with depth-3 now deployable at N=640, does tool-search harvest depth-4 more cheaply?

## Artifact Manifest

See `reports/artifact_manifest.yaml`. Key: `scripts/tool_harvest.py`, `scripts/train_lora.py`,
`scripts/eval_ladder.py` (frozen paired set + function-sig & op-composition dedup + leakage report),
`scripts/analyze.py` (Wilson CIs + CI-overlap decision), `data/train_{40,160,640}.jsonl`, `data/eval_frozen.jsonl`,
`runs/eval_*.json`, `runs/verdict.json`, `analysis/dose_response.png`, `reports/design_review.md`. Adapters
(~180MB each) omitted from git.

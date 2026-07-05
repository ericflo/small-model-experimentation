# Depth scaling & controls: no saturation through 1280, the gain is data-diversity (not compute), and the recipe repeats one rung deeper (weakly)

## Summary

Three follow-ups to C23 (depth-3 install is data-limited), each hardened by an adversarial workflow review
(`reports/design_review.md`). All on the list 16-op DSL, interpreter brute-search explorer (no external model),
QLoRA r32/α64 epochs=3, frozen paired held-out deduped by function-signature AND op-composition (0 leakage
verified: depth-3 0/2305, depth-4 0/318).

### Arm 1 — No saturation through 1280 tool-pairs
| N (depth-3 pairs) | 0 | 40 | 160 | 640 | 1280 |
|---|---|---|---|---|---|
| distinct functions | 0 | 39 | 153 | 555 | 1156 |
| think cov@16 (Wilson) | 0.00 | 0.087 | 0.212 | 0.375 [.28,.48] | **0.537 [.43,.64]** |
| deployable greedy@1 | 0.00 | 0.013 | 0.037 | 0.138 | **0.188** |

The depth-3 dose curve keeps climbing past 640 — cov@16 0.375 → 0.537 and deployable single-shot greedy@1
0.138 → 0.188. Distinct functions grow near-linearly (so a flat curve would be real saturation, not
harvest-diversity exhaustion — it isn't flat). **The data-limited regime extends at least to 1156 distinct
depth-3 functions with no plateau.** (Caveat: adjacent-dose Wilson CIs marginally overlap at n=80 — the overall
trend is unmistakable and non-adjacent doses separate cleanly, but adjacent-point significance is not
established. The 2560 dose was dropped: its training ran ~2× slower than budgeted and was blocking the eval
phase.)

### Arm 2 — The gain is data-DIVERSITY, not compute
A 2×2 at matched size/mixture/steps splits C23's "more data" gain:
| | 40 distinct funcs | 640 distinct funcs |
|---|---|---|
| 120 depth-3 visits | 0.087 (N=40 dose) | — |
| 1920 visits | 0.163 (up40) | 0.375 (train_640) |

- **Diversity at fixed compute** (up40 → train_640, both 1920 visits): 0.163 → 0.375 — **cleanly significant**
  (non-overlapping Wilson CIs: up40 [.10,.26], train_640 [.28,.48]).
- **Compute at fixed diversity** (N=40 → up40, same 40 functions): 0.087 → 0.163 — positive point estimate but
  **within noise** (overlapping CIs).

So C23's "data-limited" is genuinely **data-diversity-limited**: banking more *distinct* explorer-found
functions is what drives the gain, not merely more gradient steps on the same functions.

### Arm 3 — The recipe repeats one rung deeper, weakly
Depth-4 rung (cov@16, n=60), against the **scaffold-only baseline** (banked_640 = depth-1+2+640d3, no depth-4
data — the correct attribution reference, since depth-3 skill transfers):
| | raw base | scaffold (640 d3) | banked_d4 (+320 d4) |
|---|---|---|---|
| depth-4 cov@16 | 0.000 | 0.067 | **0.183** |
| depth-4 greedy@1 | 0.000 | 0.033 | 0.033 |

Adding 320 depth-4 tool-pairs nearly triples depth-4 coverage over the transfer baseline (0.067 → 0.183,
+0.116). But deployable greedy@1 stays flat (0.033) — a **weak/test-time-only** install, exactly the signature
depth-3 showed at low doses (C22). At n=60 the Wilson CIs marginally overlap (scaffold [.03,.16] vs banked_d4
[.11,.30]), so this is **suggestive, not conclusive**. The depth-3 guardrail held (banked_d4 scores 0.425 on
depth-3 vs the scaffold's 0.375 — no forgetting, slight improvement). By C23's logic, a depth-4 dose ladder
should strengthen it.

## Research Program Fit

Refines and stress-tests C23. Arm 1 shows the data-limited regime is deep (no saturation through 1156 distinct
functions). Arm 2 attributes the gain to genuine data-diversity, tightening "self-training installs what the
explorer finds" — it's the *distinct verified solutions* that matter. Arm 3 shows the whole recipe
(tool-search explorer + banking installer) repeats one compositional rung deeper, at the same weak-then-scales
efficiency per rung. Together: the ladder-climbing recipe is diversity-driven and rung-repeatable.

## Method

`scripts/harvest2.py` (extend depth-3 to 2560 nesting C23's 640 excluding the held-out; upsampled-40; depth-4
harvest + held-out), `scripts/train_lora.py`, `scripts/eval_ladder.py` (func-sig + op-composition dedup +
leakage report), `scripts/analyze.py` (Wilson CIs + CI-overlap verdicts). Reused C23's depth-3 frozen held-out
+ base/40/160/640 evals + banked_640 adapter (bit-identical → same numbers, no re-run).

## Pre-registered verdicts

- **P1 (saturation):** curve still rising at 1280 (0.375 → 0.537); **no saturation** through 1156 distinct
  functions (adjacent-dose significance marginal).
- **P2 (data vs compute):** **data-diversity dominates** — diversity effect cleanly significant, compute effect
  within noise.
- **P3 (depth-4):** **recipe repeats, weakly** — banked_d4 0.183 vs scaffold 0.067 (~3×), but test-time-only
  (greedy flat) and CIs marginally overlap at n=60; suggestive, needs a depth-4 dose ladder.

## Honest limits

Single training seed at every point (Wilson CIs are eval-noise only; no seed error bars). n=60–80 underpowers
adjacent-dose and depth-4 significance — point estimates are clear, adjacent CIs overlap. Fixed epochs (data
confounded with gradient exposure *except* in the deconfounding up40 control). Depth-4 is a single dose
(installs-vs-installs-a-little unresolved). 2560 dose dropped (training too slow). Harvested pairs are the
search-easy tail.

## Next Experiments

- Depth-4 dose ladder (80/320/1280) to resolve installs-vs-installs-a-little and test whether depth-4 becomes
  deployable with more data (the C23 test one rung deeper).
- ≥3 training seeds at the decision points (saturation top, up40, depth-4) for seed error bars.
- Push depth-3 past 1280 with faster training to locate the eventual saturation.

## Artifact Manifest

See `reports/artifact_manifest.yaml`. Adapters (~180MB each) moved out of repo.

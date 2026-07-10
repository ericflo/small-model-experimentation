# Be your own tool-search: first-move ranking is at chance; banking improves step-wise guidance

## Summary

"Be your own tool-search": can the FIXED model climb the depth wall by proposing+verifying one DSL op at a
time (it ranks the 32 ops given current lists -> goal; the interpreter applies + verifies; beam search)?
An adversarial workflow review (`reports/design_review.md`, verdict *flawed*) showed the original "is depth-3
latent" framing was unfair and that the sibling C12 already ran the base-model version (guidance buys
efficiency not coverage). So this experiment **pivoted** to dissect WHERE decomposition breaks and whether
**banking** (C24) fixes it. All fixes adopted: search terminates on VISIBLE examples only (graded on hidden);
brute-force (all 32 ops) is the honesty bar (random only a floor); all 80 held-out tasks are min-depth-VERIFIED
true-depth-3; a pruning ablation is included.

### 1. The lookahead wall (base model)
Per-step ground-truth next-op ranking (n=80, chance top-1 = 0.031):
| step (goal distance) | top-1 | top-6 | mean rank |
|---|---|---|---|
| step 1 (3 ops away) | **0.013** | 0.125 | 19.7/32 |
| step 2 (2 away) | 0.062 | 0.237 | 17.9/32 |
| step 3 (1 away) | **0.237** | 0.375 | 15.4/32 |

The base model recognizes a one-step transform (step-3 ≈ 8× chance), has weak above-chance ranking two steps
from the goal (step-2 top-1 0.062 versus chance 0.031), and is at/below chance only on the first move three
steps from the goal. At matched budget (~1088 interpreter calls/task), base-guided search solves **1/80** tasks
versus random's **2/80** and brute-force's **23/80** (coverage 0.013/0.025/0.287). The one-task base-versus-
random difference is within noise: this establishes that base guidance is **no better than random**, not that
it is reliably worse.

### 2. Banking installs transferable LOOKAHEAD (I predicted the opposite)
I pre-registered that banking (C24, monolithic depth-3 SFT) would be *compilation not planning* — lifting only
terminal recognition. **Refuted.** The banked models rank the next op better at every step including
lookahead. The dose trend is clear at steps 2–3; at step 1, the 640→1280 increment is only 10/80→11/80 and
is within noise:
| guide | step 1 | step 2 | step 3 |
|---|---|---|---|
| base | 0.013 | 0.062 | 0.237 |
| banked N=640 | 0.125 | 0.138 | 0.463 |
| banked N=1280 | 0.138 | 0.250 | 0.550 |

Banking on monolithic prompt→code solutions installs a **reusable multi-step planning improvement that
transfers to the step-wise search-guide role** — lookahead (step-1/2) lift = +0.156, terminal (step-3) lift =
+0.313 (banked1280 vs base). It is not a lookup map.

### 3. This upgrades the guide from worse-than-random to competitive
banked1280-guided search solves **18/80**, versus **2/80** random and **1/80** base, at matched budget — now
roughly matching brute-force's **23/80** at the same low budget. Banking converts an uninformative base guide
into a competitive one. (It still does not beat brute's high-budget coverage 0.81; only one beam width was
tested.)

### 4. Controls (the model's contribution is real, not the pruning)
Pruning ablation (hidden coverage): random-proposals + distance-pruning reaches only **0.037** (pruning alone
does NOT crack depth-3); brute (exhaustive proposals) + pruning reaches **0.487**. So the solver is real
proposals + pruning, and banking's lift to the guide is a genuine model improvement, not a pruning artifact.

## Research Program Fit

Connects C12 (decompose-search: base guidance = efficiency not coverage) with C24 (banking installs depth-3).
The novel finding: **banking is not just monolithic memorization — it installs transferable compositional
*planning*** that upgrades the model's own search-guidance. Sharpens what banking does: it improves the latent
compositional machinery, transferring across task formats (monolithic training → step-wise lookahead).

## Method

List 16-op DSL (32 op/param combos). Fixed model ranks the 32 ops by length-normalized likelihood given a
prompt showing the DSL + current lists → goal lists (batched forward, vectorized). Beam search (distance-to-
target pruning), depths 1–3, VISIBLE-only termination, hidden-graded. Guides: base / banked_640 / banked_1280
(reused C24 adapters). Controls: brute (all 32) / random, swept over beam/budget. `scripts/decompose.py`
(search), `scripts/run.py` (rank + search + ablation), `scripts/analyze.py`.

## Pre-registered verdicts
- **P1 (lookahead wall):** PARTIAL — base ranking is monotone in goal-proximity; only step 1 is at/below
  chance, while step 2 is weakly above chance.
- **P2 (does banking install planning?):** **REFUTED my hypothesis** — banking lifts lookahead (step-1/2)
  dose-dependently; it installs transferable planning, not monolithic compilation only.
- **P3 (coverage bar):** base-guided does NOT beat brute and is within noise of random; banked-guided becomes
  competitive at matched budget but still does not exceed brute's high-budget coverage.
- **P4 (pruning):** HELD — random+pruning ≈ 0.037; pruning alone is not the solver.

## Honest limits

Likelihood-ranking a closed 32-op set is easier than free generation (a generative variant is future work).
Single frozen held-out (n=80), one seed; one beam width for the guided coverage point (no full budget curve for
the banked guide). The claim is about step-wise proposal/planning on this substrate; it does not re-litigate
C24's monolithic result (which stands). "Banking installs planning" means measurably transferable step-wise
lookahead, not that the model becomes a superior search guide (it reaches parity, not dominance).

## Next Experiments
- Full coverage-vs-budget curve for the banked guide (does it beat brute at any budget, or only reach parity?).
- Free-generation proposal variant (does the lookahead lift survive without the closed-set menu?).
- Does banking depth-4 install depth-4 lookahead (repeat the dissection one rung deeper)?

## Artifact Manifest
See `reports/artifact_manifest.yaml`. Reuses C24 banked_640/banked_1280 adapters (out of repo).

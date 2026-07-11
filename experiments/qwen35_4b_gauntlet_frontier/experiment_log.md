# Gauntlet frontier: difficulty escalation past the breadth-install plateau Experiment Log

## Scaffold

Created as a new experiment scaffold.

## 2026-07-10 — frontier experiment: ablation attribution first

- Ablation arms (matched-dose, trained from base on rounds-2/3 data slices,
  merged deployment, paired vLLM quick events): recovery-only (553 fc
  examples, trimmed to fit the encode window) +0.1216 (seed 53001);
  ferrier-only (665) +0.3141 (53002); breadth-matched (665) +0.3302 (53003).
- ATTRIBUTION REVISION: a SINGLE rich verified family installs nearly the
  full aggregate effect AND nearly full cross-substrate gym transfer
  (spindle 0.748 vs breadth's 0.752; runeward 0.944 vs 0.933) — the install
  is a substrate-agnostic behavioral policy (conclude-within-budget, commit
  tersely, one-line actions), not breadth-dependent content. Breadth's
  measurable edge is axis-aligned increments (menagerie chronicle +0.750 vs
  +0.375; gym stallwright 0.517 vs 0.379). Pure emission repair
  (recovery-only) captures ~+0.12 of the delta with weak transfer. No
  format-capture interference observed from single-family training (contra
  C14-era expectations). Gym numbers include the new L5-L6 strata (base
  column is round-1 L1-L4 — directional only).
- Frontier gym landed: all 12 families extended to L1-L6 (L1-L4
  byte-identical, verified), horizons to 22; two NEW weak-axis families
  (patchwheel: rewrite-rule repair; packhouse: cart-packing optimization);
  14/14 selftests green.

## 2026-07-11 — frontier + sharp1024 verdicts: THE SECOND WALL

- Frontier adapter (union of all rounds + L3-L6 escalated harvest, 5,700+
  examples): quick +0.2716 (seed 53004) / +0.3135 (53005), medium +0.3196
  (53006). Gym-internal: L1-4 mean 0.681, L5-6 0.466 (frontier competence
  INSTALLED in-gym — base was near-zero there) — but the blackbox band did
  not move.
- Deploy-budget-matched variant (sharp1024: think<=900 + recovery, 3,756
  examples, on-policy for the quick 1024 budget): quick +0.2934 (53007) /
  +0.2376 (53008), medium +0.3418 (53009; highest medium delta but same
  ~0.41 absolute).
- packhouse scorer mystery solved (model answered the objective VALUE, not
  the assignment; prompt now names the exact ANSWER shape — post-fix gym
  eval reads 0.32 L1-4) ; patchwheel harvests correct-but-never-closes
  (recovery-arm-only contributor, like stallwright).
- Infrastructure: recurring allocator OOM after long engine cycling fixed
  by PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on the trainer.
- VERDICT: across dose, iteration, breadth, difficulty escalation, working
  recovery supervision, and deploy-budget matching, the treated model lands
  at 0.38-0.45 menagerie aggregate in every one of 9 paired decision events
  (base 0.07-0.16). The emission-policy install is a large one-time step to
  a ROBUST CEILING; the residual deficit (menders/lockpick/stockade/rites/
  warren cores) is not reachable by ANY train-on-own-verified-outputs
  variant tried. The goal's +0.32-on-both-tiers bar was not decisively
  exceeded; the plateau itself is the finding (claim C53).

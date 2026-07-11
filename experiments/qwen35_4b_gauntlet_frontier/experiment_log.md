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

## 2026-07-11 (cont.) — oracle-trace distillation: the wall holds against gold procedure supervision

- Added oracle_trace() to the 7 residual families (hand-coded solver
  procedures narrated as truth-blind think-channel text, C48 style; 1,680
  traces, selftest-gated: trace + terse ANSWER scores 1.0, <=800 words,
  deterministic). Trained from base on traces + sharp1024 mix.
- Decision events (paired vLLM merged): quick +0.2816 (seed 53010; treated
  ABSOLUTE 0.4743, the highest observed) / +0.2677 (53011); medium +0.2886
  (53012; absolute 0.4823, highest observed). Deltas remain in the
  +0.24..+0.34 band.
- FINAL READING: twelve paired decision events across seven lever classes
  (dose, iteration, breadth, difficulty, recovery supervision, deploy-budget
  matching, programmatic oracle distillation) all land in-band. The C53
  second wall holds even against externally-derived procedure supervision
  the model cannot self-generate. The goal's "+0.32 decisively on both
  tiers" bar is unreachable within this recipe family and provenance; the
  wall itself, its attribution, and the one-time +0.30 step are the
  program's deliverables.

## 2026-07-11 (cont.) — clearance campaign: quick decisively cleared, medium does not follow; tier dissociation found

- Breadth-matched arm medium (seed 53013): +0.3304 — momentarily clearing
  both tiers (quick +0.3302 at 53003) — but replication regressed to the
  band (quick +0.2669 at 53017, medium +0.2806 at 53018): arm means quick
  ~+0.299 / medium ~+0.306. Not decisive.
- CONCENTRATE arm (1,440 curated rows: 800 short-think self-examples + 420
  L1-L4 oracle traces + 220 trimmed recovery; 3 epochs): quick +0.3301
  (53014) and +0.4172 (53015; ABSOLUTE 0.5053 — first 0.50+ ever); medium
  +0.2783/+0.2432/+0.2552 (53016/53019/53020; pre-declared arm-mean metric,
  no exclusions): quick mean +0.374 DECISIVELY beyond +0.32; medium mean
  +0.259 — below.
- NEW FINDING — TIER DISSOCIATION: the short-curated mix optimizes the
  quick regime (L1-L2 atoms) at measurable cost to medium (episodes +
  L3-L4): concentrate holds the quick record (+0.417) while sharp1024
  holds the medium record (+0.342); across 19 paired events NO arm clears
  both tiers decisively. The goal's conjunctive bar is unreachable within
  this recipe family and provenance; per-tier specialists exist, a
  both-tier generalist at that level does not.

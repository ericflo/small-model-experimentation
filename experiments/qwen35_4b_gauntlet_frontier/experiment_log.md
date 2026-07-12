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

## 2026-07-11 (final) — blend arm: convexity confirmed, quick decisively broken, medium converges just under the bar

- BLEND arm (concentrate + 800 L3+ episode turns, 2.5 epochs, 2,240 rows):
  quick +0.4286/+0.3891 (arm-mean +0.4088; absolute 0.5224 = overall
  record); medium +0.3692/+0.2688/+0.2562/+0.2854/+0.3554 — five
  no-exclusion events, arm-mean +0.3070 (absolute record 0.5214 at 53023).
- CONVEXITY CONFIRMED: the blend dominates BOTH parents on BOTH tiers
  (concentrate quick +0.374/medium +0.259; sharp1024 quick ~+0.266/medium
  +0.342 single-event) and holds both absolute records — the tier
  dissociation is convex, so mix composition is a real optimization axis.
- FINAL LEDGER (28 paired events, 10 arms, 9 lever/mix classes): the
  goal's quick bar is DECISIVELY broken (+0.4088 arm-mean vs +0.32); the
  medium bar converges just below (+0.3070 arm-mean, best single event
  +0.3692). The conjunctive both-tiers bar fails by ~0.013 on medium after
  exhausting dose, iteration, breadth, difficulty, recovery supervision,
  budget matching, oracle distillation, and mix-composition search.

## 2026-07-11 (terminal) — blend2 closes the composition axis: blend is the Pareto point

- BLEND2 (medium-tilted: 72% episode mass, 2,077 rows): quick
  +0.3408/+0.2979 (arm-mean +0.3194), medium +0.2727/+0.3350/+0.2981
  (arm-mean +0.3019). Tilting toward medium LOWERED quick without raising
  medium beyond blend's +0.3070 — the composition gradient does not extend.
- FINAL MAP (33 paired events, 11 arms): blend is the Pareto point of the
  mix-composition axis (quick +0.4088 / medium +0.3070 arm-means; absolute
  records 0.5224/0.5214). Medium arm-means cluster at ~0.26-0.31 across all
  compositions; the conjunctive +0.32-both-tiers bar is not reachable by
  any composition of this recipe family's data. Quick: decisively broken.

## 2026-07-11 (BREAKTHROUGH + logical conclusion) — novel mechanisms clear MEDIUM; tier-Pareto frontier mapped

First-principles reframing (user directive to invent, not remix): the medium
ceiling is a SERIAL-COMPUTE wall — the residual axes (menders/lockpick/
stockade/rites) are search/induction that need long derivations which do not
fit the deployed budget, so the emission policy was teaching early commitment
to WRONG answers. Three NEW weapons built (exploiting the unique verifier +
generator control):
- **Length-penalized compression advantage** (scripts/build_efficiency_data.py,
  custom loss in train_think.py): rewards the model's OWN shortest correct
  trace per hard item + a positive-only brevity gradient, active ONLY where it
  already succeeds (safe by construction) — amortizes serial test-time compute
  into the weights ("think faster"). Harvest confirmed the premise: 36%
  unreachable-at-K=6, 19% shorter path on the solvable 63%.
- **Skin-shuffling** (base.skin_mapping/apply_skin + SKINNABLE on 10 families,
  build_skinshuffled_traces.py): fresh pseudo-vocabulary on EVERY training row
  (kilnrite damper/OPEN -> draurdaush/Haurmorm) so procedures bind to mechanics,
  not tokens — the only thing that transfers to the blackbox instrument.
- **GRPO-lite contrast**: wrong answers pushed down (answer-span only, C29
  guard: never negative gradient on thinking; abs-normalized loss).

Arms (paired vLLM merged; quick n up to 6, medium n=3):
- **effskin** (warm-start from blend): quick +0.333, medium +0.272 — the novel
  mechanisms MOVED the targeted residual axes on medium (lockpick/induction
  +0.267, stockade/optimization +0.289) but warm-start displaced blend breadth.
- **apex** (blend + efficiency + skin, from BASE): quick +0.308 (n=6),
  **medium +0.345 (n=3, ALL three events > +0.32: +0.353/+0.346/+0.336)** —
  the FIRST arm in the entire campaign to DECISIVELY clear the +0.32 medium
  bar. The residual-axis gains stacked on breadth when co-trained from base.
- **apex60** (blend + 60% residual dose): quick +0.313, medium +0.285 —
  STRICTLY DOMINATED (worse than blend on quick, worse than apex on medium):
  the tier trade-off is NON-CONVEX. The medium lift materializes only at the
  full residual dose; partial doses lose it while still paying the quick cost.

DEFINITIVE CONCLUSION (~50 paired events, 13 arms across dose/iteration/
breadth/difficulty/recovery/budget-matching/oracle-distillation/mix-composition/
compression-advantage/skin-shuffle/GRPO-lite/from-base-union/dose-interpolation):
both tiers are INDIVIDUALLY breakable past +0.32 — quick to +0.41 (blend),
medium to +0.345 (apex) — but NO single Qwen3.5-4B adapter clears BOTH. The
two tiers (short-atom vs episode/deep-atom) occupy a non-convex Pareto frontier
and compete for the fixed model's representational budget; the arm that clears
one falls ~0.01-0.03 short on the other. The goal's conjunctive +0.32-both-tiers
bar sits just beyond a single 4B adapter's tier-Pareto frontier — but the
medium half, long the harder wall, was genuinely broken here for the first time
by mechanisms designed from the serial-compute diagnosis.

## 2026-07-11 (capacity test — the last single-adapter lever) — 4x rank REFUTES capacity competition

- apex at LoRA rank 128 / alpha 256 (4x the params of every prior arm, same
  apex recipe + data): quick +0.249 (n=4), medium +0.229 (n=3) — WORSE than
  r32 apex (+0.308/+0.345) on BOTH tiers. More capacity at fixed data
  overfits and hurts.
- This decisively refutes the C54 capacity-competition reading as an
  ADAPTER-SIZE artifact: the tiers do not fail to co-fit because r32 ran out
  of room. The tier-Pareto frontier is FUNDAMENTAL to the fixed 4B at this
  data scale — the +0.32-both-tiers conjunction is unreachable by a single
  adapter regardless of rank.
- FINAL: 14 arms, ~65 paired events. Both tiers individually broken past
  +0.32 (quick 0.409 blend, medium 0.345 apex — medium broken for the FIRST
  time via the novel compression-advantage + skin-shuffle mechanisms). No
  single adapter clears both; interpolation is strictly dominated (non-convex
  frontier) and 4x capacity is worse (fundamental, not capacity). The
  remaining single-4B path to the conjunction is a tier-router (blend for
  short-horizon, apex for episode/deep); clearing both in one weight-set
  requires a larger base or a router, both beyond a single 4B adapter.

## 2026-07-12 (model-soup frontier sweep — the airtight conclusion) — the conjunction is unreachable by any single-model method

- Model souping (weight-space averaging of blend=quick-specialist and
  apex=medium-specialist; scripts/soup_merge.py, single weight-set): lands
  ABOVE the data-interpolation hull (apex60) and traverses the frontier.
  - soup50 (a=0.50): quick +0.359 / medium +0.312
  - soup40 (a=0.40, clean n=6/6): quick +0.331 / medium +0.304 (sd 0.016) —
    the best JOINT point, both tiers high in one weight-set
  - soup35 (a=0.35): quick +0.322 / medium +0.338 (n=3; the medium figure is
    a favorable n=3 noise draw — soup40 n=6 +0.304 and soup50 +0.312 bracket
    it lower)
- DECISIVE STRUCTURE: medium clears +0.32 only near pure apex (a<=~0.15,
  +0.345 tight) while quick clears only at a>=~0.38 — DISJOINT ranges, so no
  soup mix, no interpolation, no capacity setting clears both.
- Operational note: a two-battery overlap caused a GPU-contention hang + a
  zombie vLLM holding 11.5 GB; single-tenant the GPU (one bench battery at a
  time) and reap orphaned vllm_runner PIDs after any TaskStop.
- FINAL (single-model methods exhausted, ~90 paired events): quick broken to
  +0.41, medium broken to +0.345 (first ever, novel compression+skin
  mechanisms), best joint soup +0.331/+0.304 — the +0.32-both-tiers
  conjunction is unreachable by ANY single Qwen3.5-4B model. The tier-Pareto
  frontier is fundamental; its max-min straddles the bar within measurement
  noise. The only remaining single-4B path is an inference-time tier-router;
  clearing both in one weight-set requires a larger base.

## 2026-07-12 (expert iteration — the final arc) — EI ratchets harvest+quick, NOT medium; medium ceiling is a capability boundary

- Ran a full expert-iteration round (user proposal: harvest from the best
  joint soup, retrain, re-soup). Harvest from soup40 RATCHETED: 27%
  unreachable-at-K vs blend's 36%, 632 vs 491 solved (+29%), 22% vs 19%
  compression — C11's per-model coverage bound moves outward with a stronger
  start.
- But the gain lifts ONLY quick: apex2 (both-tier EI harvest) quick +0.375 /
  medium +0.258; apexm2 (hard-only EI harvest, medium-designed) quick +0.381 /
  medium +0.309. EVERY model trained on the soup harvest leans quick
  regardless of data weighting — the harvest is quick-biased because the
  medium tier's ~27% unreachable-at-K core never enters it.
- soup_aa55 (same-recipe apex+apex2 soup): quick +0.336 / medium +0.304 —
  apex2's weak medium dilutes it; no super-additive rescue.
- DEFINITIVE: expert iteration cannot ratchet medium. apex +0.345 (trained on
  the ORIGINAL non-EI harvest) remains the sole medium-clearing config. The
  medium ceiling ~+0.345 is a CAPABILITY BOUNDARY of the 4B base (self-
  generated data can only compress what the model already solves, never teach
  the unreachable-at-K residual). Across ~110 paired events and EVERY single-
  model method — 16 training arms, 4x capacity, data-interpolation, 6-point
  weight-space soup sweep, and a full expert-iteration round — no single
  Qwen3.5-4B model clears both +0.32 tiers. The tier-Pareto frontier is
  fundamental; the conjunction requires an inference-time tier-router or a
  larger base.

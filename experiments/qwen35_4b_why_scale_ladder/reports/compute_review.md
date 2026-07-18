# Compute Review

Scoped to the SCALE-LADDER training sweep (LoRA r32/a64 on base_reserialized
from the diverse WHY curriculum at rungs 2000/5000/10000/20000/40000, all at 1 epoch). Covers all
rungs (identical design, varying corpus size + the size-appropriate epoch
schedule). Backed by the two-lens adversarial workflow (diversity+WHY-truth
at scale; pipeline+contamination+standalone) which found ZERO MAJOR / zero
minor, plus my independent 2000-row verification (100% unique programs, 1102
WHY templates, 25/25 strip-execute pass).

- DIVERSITY (the prerequisite that makes scaling meaningful): 59 families,
  100% unique programs at 5k AND 20k (vs the saturating why_comment
  generator's 438/504), 1196 distinct normalized WHY templates (vs ~75).
  So each rung carries genuinely new (code->why) signal, not repeats.
- Training signal CORRECT + provenance-clean: every row's #WHY: is true and
  line-specific BY CONSTRUCTION (no teacher); strip comments -> code passes
  all asserts (execution-verified per row); safety/termination caps.
  Contamination-clean at scale (0 banned names, 0 distinctive shared
  7-grams at 10000). Max render 499 tokens < 4096 -> zero truncation.
- Vehicle: fresh r32/a64 adapter per rung via vendored train_think.py
  (e0eca2a2...) on base_reserialized (tree 26d8ee48..., weights b654e033...
  fail-closed), lr 1e-5, batch 1, grad-accum 8, max-length 4096, w_think
  0.2, w_close 0.2, seed 94101. EPOCHS = 1 FOR EVERY RUNG (owner directive):
  with unlimited unique data we scale DATA VOLUME, never re-show examples -
  every gradient step sees fresh data (no memorization, no epoch confound in
  the ladder). Rungs 2000/5000/10000/20000/40000 (generator holds 100% unique
  through 40000, verified). Optimizer steps = rows/8 = 250/625/1250/2500/5000.
- HYPOTHESIS: if the WHY effect is real+underpowered, HumanEval climbs with
  genuinely-diverse scale to a peak then plateaus/collapses; if noise, flat.
  This is the powered test the single 504-row bet could not give. The
  pre-collapse best composite becomes the SFT foundation for Phase B (RLVR).

**Verdict:** `PASS_CONTROL_TRAINING`.

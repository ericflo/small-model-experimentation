# Compute Review

Scoped to the dual-channel WHY-think scale-ladder training sweep (LoRA
r32/a64 on base_reserialized at rungs 2000/5000/10000/20000/40000, 1
EPOCH each). Backed by the two-lens adversarial workflow (derivation-
genuineness + example-trace-truth; pipeline+contamination+standalone+
thinking-plumbing) which found ZERO MAJOR / zero minor, plus my
independent 25-row check (25/25 strip#WHY->code passes; 25/25 think is a
genuine approach+worked-example derivation; 25/25 NOT the joined #WHY
comments; 25/25 distinct).

- CORRECTED DESIGN + FOUNDATION: measured thinking-ON (the harness was
  fixed from thinking-off/512 to thinking-on/8192; base re-baselined to
  HumanEval 147/164 = 89.6%, MBPP 151/200 = 75.5%). The curriculum
  puts a GENUINE step-by-step derivation in the <think> channel (parse
  spec -> approach decision -> build -> EXECUTED worked-example trace ->
  conclude; 4997/5000 distinct derivations, not comment-concat) AND
  clean code with strippable inline #WHY: comments in the answer. This
  REINFORCES the native thinking (not empty-think), addressing the
  worry that WHY training destroys the think block.
- Diversity/verification: 59 families, 100% unique programs through 40k,
  1196 #WHY templates, every worked-example trace independently matches
  real execution, contamination-clean at scale, all renders < 4096.
- Vehicle: fresh r32/a64 via vendored train_think.py (e0eca2a2...) on
  base_reserialized (tree 26d8ee48..., weights b654e033... fail-closed),
  lr 1e-5, batch 1, grad-accum 8, max-length 4096, w_think 0.2, w_close
  0.2, seed 95201. EPOCHS=1 for every rung (unlimited unique data ->
  scale data volume, never re-show; no memorization, no epoch confound).
- HYPOTHESIS (measured thinking-on): does the WHY-think curriculum, as
  unique data scales, (a) hold HumanEval near its 89.6% ceiling without
  wrecking it, (b) move MBPP where there is headroom, (c) reinforce
  thinking - and is there a pre-collapse peak to bank as the SFT
  foundation for RLVR? Agentic (base 23%) is the primary follow-on.

**Verdict:** `PASS_CONTROL_TRAINING`.

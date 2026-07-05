# Pre-registration: can thinking breach the lookahead wall?

Logged 2026-07-05, before full data (design hardened by an adversarial workflow review, verdict
sound_with_fixes; `reports/design_review.md`). C25 found the fixed 4B has a LOOKAHEAD WALL: in a single
forward pass it can't plan the first of 3 ops (step-1 next-op likelihood-ranking top-1 = 0.013, chance 0.031),
though it recognizes a 1-step transform (step-3 0.237). Thinking is serial test-time compute -- the natural
lookahead mechanism, and the dormant C9 lever. Does a thinking budget breach the wall, with NO training?

## Method (channel-matched, per the review)
- Substrate: list 16-op DSL (32 op/param combos). 80 min-depth-VERIFIED true-depth-3 held-out (reuse C25's).
- PRIMARY metric = **think->RANK vs no-think->RANK** (both the SAME 32-way likelihood ranking; think = generate
  a thinking trace of B tokens, close </think>, then rank). This is channel-matched to C25 and immune to
  parse/truncation noise (the abandoned generation channel).
- **HEADLINE = STEP 1** (current==input, goal 3 ops away, NO intermediate state materialized -- the only clean
  lookahead test). Steps 2/3 are reported but FLAGGED: the harness feeds them the true intermediate list, so a
  lift there is decomposition + state-materialization (interpreter feeding the model), not lookahead.
- Budgets B in {0 (no-think), 1024, 2048}. n=40.
- Internal-brute-force discriminator: capture + classify step-1 thinking traces (enumerate-and-test vs
  goal-directed); if step-1 lifts, a DSL-size sweep (does it collapse as branching grows?).

## Predictions (locked)
- **P1 (does thinking breach the wall?):** does step-1 think->rank top-1 rise materially above no-think (0.025)
  -- threshold > 0.10 and monotone in B? YES => the wall is a single-forward-pass limit breached by serial
  compute (weak claim: training-free elicitation). NO (step-1 stays ~chance) => the wall survives thinking.
- **P2 (contamination check):** steps 2/3 lift MORE than step-1 (state-materialization dominates), confirming
  step-1 is the clean test.
- **P3 (planning vs internal enumeration):** if step-1 lifts, are the traces goal-directed reasoning, or the
  model enumerating ops and mentally applying them (being its own interpreter)? The latter = weak claim only
  (serial compute lets it emulate the interpreter), NOT latent PLANNING.

## Honest limits
Ranking a closed 32-op set is easier than free generation. Single frozen held-out (n=40), one seed per budget
(cost: thinking generation is expensive even batched). The claim is scoped to step-wise next-op prediction on
this substrate. A no-lift is a valid refutation ONLY because the ranking channel is parse-immune (unlike raw
generation). Budget reported on the model-token axis; the only clean end-to-end "breach" would be a coverage
CEILING win over brute-force (deferred unless step-1 lifts).

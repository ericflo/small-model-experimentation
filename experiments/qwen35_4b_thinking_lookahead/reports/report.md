# Thinking does NOT breach the lookahead wall — but it powerfully amplifies recognition

## Summary

C25 found the fixed 4B has a **lookahead wall**: in a single forward pass it can't plan the first of 3 ops
(step-1 next-op likelihood-ranking ≈ chance) though it recognizes a 1-step transform (step-3 0.275). Thinking
is serial test-time compute — the natural lookahead mechanism, and the dormant C9 lever. Does a thinking budget
breach the wall with **no training**? Design hardened by an adversarial workflow review (verdict
sound_with_fixes): the primary metric is **think→RANK vs no-think→RANK** (think B tokens, close `</think>`,
then the SAME 32-way likelihood ranking as C25 — channel-matched, immune to parse/truncation), and the headline
is **STEP 1** (goal 3 ops away, no intermediate state materialized — the only clean lookahead test; steps 2/3
are handed the true intermediate list, so a lift there is state-materialization, not planning).

### Result (n=40, chance top-1 = 0.031)
| step | B=0 | B=1024 | B=2048 |
|---|---|---|---|
| **step 1** (3 away, no state — real lookahead) | 0.025 | 0.050 | **0.075** |
| step 2 (2 away, true state given) | 0.000 | 0.125 | 0.325 |
| step 3 (1 away, recognition) | 0.275 | 0.600 | 0.600 |

- **Thinking does NOT breach the lookahead wall.** Step-1 stays at chance across budgets (0.025 → 0.075;
  Wilson CIs all overlapping: B=0 [0.004, 0.129], B=2048 [0.026, 0.199]). Even 2048 thinking tokens do not let
  the model plan the first of 3 ops.
- **Thinking's benefit scales INVERSELY with lookahead distance:** huge for recognition (step-3, goal 1 op
  away: 0.275 → 0.600), moderate at 2 away (0.000 → 0.325), essentially zero at 3 away (real planning). So
  **thinking amplifies RECOGNITION, not PLANNING** — and only where the interpreter materializes the true
  intermediate state.
- **Internal brute-force refuted.** If thinking let the model simulate the depth-3 path in its scratchpad
  (be its own interpreter), step-1 would rise — it doesn't. The step-1 traces show confused meta-reasoning
  about the prompt, not systematic enumerate-and-test simulation.

## Research Program Fit

The killer juxtaposition with C25: **banking lifted step-1 lookahead (0.013 → 0.138, dose-dependent) while
thinking does not.** So the two capability levers are qualitatively different — **for the planning/lookahead
gap, TRAINING (banking) is required; test-time compute (thinking) alone cannot elicit it.** For recognition,
thinking is a powerful amplifier. This reconciles with C23 (base think single-shot depth-3 coverage = 0):
thinking can't do the whole composition precisely because it can't plan the first steps. It also sharpens the
mission read: "elicit latent capability without training / beat sample-more" works for RECOGNITION (thinking
helps) but NOT for multi-step PLANNING (thinking fails; banking is needed).

## Method

List 16-op DSL (32 op/param combos), 80 min-depth-verified true-depth-3 held-out (reuse C25's; used first 40).
think→rank: `gen_sequences(think=True, budget=B)` produces a ≤B-token thinking trace, forces `</think>`, then
`score_ops_prefix` ranks the 32 ops after prompt+thinking (chunked to fit the long prefix in memory). Batched
generation across tasks. `scripts/think_rank.py`, `scripts/run_thinking.py`, `scripts/analyze.py`.

## Pre-registered verdicts
- **P1 (does thinking breach the wall?):** **NO** — step-1 stays ≈ chance (0.025 → 0.075), not ≥ 0.10, CIs
  overlap. The wall is a planning gap, not a forward-pass compute limit. (Clean refutation because the ranking
  channel is parse-immune.)
- **P2 (contamination check):** HELD — steps 2/3 (materialized state) lift far more than step-1, confirming
  step-1 is the clean test and that thinking's gains come from recognition given the scaffold.
- **P3 (planning vs internal enumeration):** internal simulation refuted — step-1 flat and traces are not
  enumerate-and-test.

## Honest limits

Ranking a closed 32-op set is easier than free generation. Single frozen held-out (n=40), one seed per budget
(thinking generation is expensive even batched). Budgets ≤ 2048; a much larger budget is untested (but the flat
trend and overlapping CIs make a sudden breach unlikely). The step-1 point estimate does creep (0.025 → 0.075)
— a whisper of signal, not significant.

## Next Experiments
- Much larger thinking budgets (8k–16k) to confirm the step-1 flatline is asymptotic, not just under-budgeted.
- Does a BANKED model + thinking stack (banking installs lookahead; does thinking then amplify it)?
- Free-generation channel with robust re-prompting, to check the ranking result survives generation.

## Artifact Manifest
See `reports/artifact_manifest.yaml`. Reuses C25's frozen depth-3 held-out; no training (test-time only).

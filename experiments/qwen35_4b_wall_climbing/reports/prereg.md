# Pre-registration: can banking CLIMB the wall?

Logged 2026-07-03, before any data. C18 showed banking self-verified solutions *installs* depth-2 composition
(coverage 0.15→0.45 on held-out tasks) but that depth-3/4 did not move — though C18 confounded this by
including depth-3 examples in training. C17/C18 established depth-3 self-samples ≈ 0 (the base cannot harvest
them). This tests the mission's holy grail directly and cleanly: **if we bank ONLY depth-1+2 solutions, does
the proposal distribution shift enough that the banked model now SAMPLES depth-3 compositions the base never
could — creating depth-3 coverage from nothing, enabling a second banking round? Can the wall be climbed
rung-by-rung by pure self-training?**

## Method

Substrate `list`. No teacher — all training targets are the fixed 4B's OWN execution-verified solutions.

**Round 1 (bank shallow, test if deeper unlocks):**
1. **Harvest** depth-1+2 ONLY (weighted to depth-2: 20 depth-1 + 90 depth-2 tasks, K=40 think samples/task,
   keep hidden-correct, cap 12/task) → `train.jsonl` (a clean depth-≤2 SFT set, larger depth-2 set than C18).
2. **Bank** QLoRA-SFT (r32/alpha64, 3 epochs, single-shot prompt→code, no-think) → `banked1`.
3. **Eval** coverage@16 (think, held-out, disjoint) at depths **2, 3, 4** for `base` vs `banked1`, n=25/depth.

**Round 2 (climb — conditional on a depth-3 unlock in Round 1):**
4. If banked1 depth-3 coverage rose materially, **harvest depth-3** from `banked1` (now samplable), bank again
   (all data) → `banked2`, and eval depths 3, 4 — testing whether the *second* rung installs and whether
   depth-4 then unlocks.

## Predictions (locked)

- **P1 (install sanity):** banked1 depth-2 coverage@16 ≥ base depth-2 coverage@16 + 0.10 (banking took; the
  C18 expansion replicates with the cleaner, larger depth-≤2 harvest).
- **P2 (THE unlock test):** banked1 depth-3 coverage@16 ≥ base depth-3 coverage@16 + 0.05, with base ≈ 0 —
  i.e. banking depth-1+2 makes ≥1-in-20-ish depth-3 tasks newly samplable via length-generalization of the
  proposal. **Refuted if banked1 depth-3 ≤ base + 0.03** (depth-local: banking installs only what you can
  already sample).
- **P3 (no two-rung leap):** banked1 depth-4 coverage ≈ 0 (≤ 0.03) — climbing, if it happens, is one rung at a
  time; banking depth-1+2 does not leap to depth-4.
- **P4 (Round-2 climb, conditional on P2):** if depth-3 unlocked, banking the newly-harvested depth-3 raises
  depth-4 coverage above banked1's — a second rung, demonstrating iterative climbing.

## Decision mapping

- **CLIMBABLE** (P2 holds): banking shallow composition unlocks deeper sampling ⇒ the wall climbs rung-by-rung
  via pure self-training; Round 2 tests the next rung. The mission's "extend capability by a lot" mechanism —
  a big positive.
- **DEPTH-LOCAL** (P2 refuted): banking installs only depths already samplable; it cannot bootstrap deeper.
  The wall is coverage-*seed*-bounded — each rung must be seeded by a proposal source the base lacks, i.e.
  tool-augmented harvest (C12 decompose-search). A clean negative that sharpens C18 and motivates tool-seeding.

## Controls / honesty

- **Clean isolation:** Round-1 training set contains NO depth-3 examples (unlike C18), so any depth-3 coverage
  gain is generalization, not memorization of depth-3 mappings.
- Held-out eval tasks are disjoint from the harvest tasks (excluded by depth+op-sequence).
- Coverage@16 via the unbiased pass@k estimator; base depth-3/4 measured fresh on the same tasks (expected ≈0).
- Diversity check (unique programs) base vs banked1, to confirm no collapse (per C11).
- Base and banked evaluated in the identical think harness (isolates the banking effect).
- **Expectation, stated up front:** C18's depth-3-stayed-0 result makes DEPTH-LOCAL the likely outcome; a
  clean negative here is the honest, valuable result (it says self-banking is seed-bounded). Any unlock, even
  small, would be the surprising, mission-central positive.

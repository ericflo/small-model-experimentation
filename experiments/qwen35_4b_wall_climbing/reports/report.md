# Can banking climb the wall? No — self-banking is coverage-seed-bounded.

## Summary

C18 showed banking self-verified solutions *installs* and even *expands* composition coverage within a depth.
This tests the mission's holy grail — can banking be **iterated to climb the wall**? Specifically: if we bank
ONLY depth-1+2 solutions (which the base can harvest), does the proposal distribution shift enough that the
banked model now SAMPLES depth-3 compositions the base never could, creating depth-3 coverage from nothing and
enabling a second banking round?

**Answer: NO. DEPTH-LOCAL.** Banking a strong depth-2 composition skill produces exactly zero depth-3 coverage.

| depth | base cov@16 | banked1 cov@16 | Δ | base greedy@1 | banked1 greedy@1 |
|---|---|---|---|---|---|
| 2 | 0.12 | **0.36** | **+0.24** | 0.04 | 0.08 |
| 3 (UNLOCK test) | 0.00 | **0.00** | **+0.00** | 0.00 | 0.00 |
| 4 | 0.04 | 0.04 | +0.00 | 0.00 | 0.00 |

- **The install worked, strongly:** banking 83 depth-2 verified solutions (no depth-3 examples) tripled
  held-out depth-2 coverage (0.12 → 0.36) — a clean replication of C18's within-depth expansion, with a larger
  set and no depth-3 contamination.
- **The unlock failed, completely:** depth-3 coverage stayed at exactly 0.00. A strong depth-2 composition
  skill does **not** length-generalize "up" to make even one depth-3 task samplable. There is nothing to
  harvest for a Round 2, so the climb halts at the first rung.

## Research Program Fit

The apex of the C13–C20 compositional-wall arc, and the direct test of the mission's "extend capability by a
lot" hope. Sharpens C18 (banking) and C11-M4 (expert-iteration is coverage-bounded) into a hard cross-depth
wall, and dovetails with C12 (tool-search extends the frontier).

## Method

Substrate `list`. No teacher — all training targets are the fixed 4B's OWN execution-verified solutions.

- **Harvest (depth-1+2 ONLY):** 20 depth-1 + 90 depth-2 tasks, K=40 think samples/task, keep hidden-correct,
  cap 12/task → **130 verified `{prompt, code}` pairs, {depth-1: 47, depth-2: 83}** from 46/110 solved tasks.
  A clean depth-≤2 SFT set with 3× C18's depth-2 examples.
- **Bank:** QLoRA-SFT (r32/alpha64, 3 epochs, single-shot prompt→code, no-think) → `banked1`.
- **Eval:** coverage@16 (think, greedy@1 + 16 sampled) on HELD-OUT tasks (disjoint from harvest) at depths
  2, 3, 4, n=25/depth, base vs banked1, one identical harness.

## Pre-registered verdicts

- **P1 (install sanity, banked1 d2 ≥ base d2 + 0.10):** HELD — +0.24.
- **P2 (THE unlock, banked1 d3 ≥ base d3 + 0.05):** REFUTED — Δ = 0.00 (base and banked1 both exactly 0.00 at
  depth 3). **DEPTH-LOCAL.**
- **P3 (no two-rung leap, banked1 d4 ≈ 0):** HELD — banked1 depth-4 = 0.04 (unchanged from base).
- **P4 (Round-2 climb):** N/A — no depth-3 coverage was unlocked, so there is nothing to harvest and bank for
  a second rung.

## Interpretation

- **Self-banking is coverage-*seed*-bounded.** Banking installs — and generalizes well *within* a depth
  (depth-2 tripled on held-out tasks) — but it installs only depths the base can *already sample*. Composition
  skill does not length-generalize across a depth: a model that now covers 36% of depth-2 tasks still covers
  0% of depth-3. You cannot bootstrap the frontier upward by self-training alone.
- **This completes the mechanistic picture of the wall.** Depth-3 composition is: not represented (C19 — the
  first-op representation thins to a thread at depth 3), not steerable (C20 — adding the latent direction is
  inert), and not reachable by banking-shallow (C21 — the depth-2 skill doesn't generalize up). All three
  test-time / self-training shortcuts fail at the deep wall by the same underlying fact — the composition
  simply is not in the model's reach at depth.
- **The only way up is to seed each rung externally.** To install depth-3 you first need depth-3 solutions to
  train on, and plain sampling harvests ≈ 0 of them. So the required proposal source is **tool-augmented
  harvest** (C12 decompose-and-compose search, which cracks depth-3 that monolithic sampling can't) →
  execution-verify → bank. The precise deployment recipe: *tools reach the next rung, banking installs it, and
  only then does the base sample it — repeat.* Self-training is the installer, not the explorer.
- **Consistent with, and sharper than, C11-M4** ("banking compounds but is coverage-bounded"): the coverage
  bound is not gradual diminishing returns but a hard wall at the depth frontier — 0.36 at depth 2, a cliff to
  exactly 0.00 at depth 3.

## Honesty notes / limits

- Depth-3 coverage is measured at K=16; a much larger K might surface rare depth-3 samples — but the base is
  *also* 0.00 at K=16, so the *comparison* (banking added nothing) is fair, and the depth-2 install is plainly
  visible at the same K. The claim is "no detectable unlock," not "provably zero at infinite K."
- Diversity did not collapse (unique depth-2 programs 11.0 → 9.2), so the null is not a diversity artifact.
- Single substrate (list), single banking round from the base. A tool-seeded Round 2 (harvest depth-3 via
  decompose-search, then bank) is the natural positive-control follow-up — it should install depth-3 where
  self-banking couldn't.

## Next Experiments

- **Tool-seeded banking (the positive control this predicts):** harvest depth-3 solutions via C12
  decompose-search (not plain sampling), bank them, and confirm depth-3 held-out coverage rises — demonstrating
  that the missing ingredient was the *explorer*, not the *installer*.
- **Representation re-probe:** does banking depth-2 raise the depth-2 first-op probe (C19) while leaving depth-3
  a thread? Confirms banking installs representation exactly at the trained depth.

## Artifact Manifest

See `reports/artifact_manifest.yaml`. Key: `scripts/harvest.py`, `scripts/train_lora.py`, `scripts/eval_ladder.py`,
`scripts/analyze.py`, `scripts/common.py`, `data/train.jsonl`, `runs/eval_{base,banked1}.json`, `runs/verdict.json`,
`analysis/wall_climbing.png`. The trained adapter (`runs/banked1_adapter`, ~180MB) is omitted from git.

# Pre-registration: tool-seeded banking — does an external explorer + banking cross the depth-3 wall?

Logged 2026-07-04, before eval data (design hardened by an adversarial multi-agent review; see
`reports/design_review.md`). C21 showed self-banking is coverage-*seed*-bounded: banking depth-1+2
self-solutions installed depth-2 (held-out coverage 0.12→0.36) but unlocked ZERO depth-3 coverage
(base 0.00 → banked1 0.00), because the base samples ≈0 depth-3 solutions to bank. The C21 prediction: the
missing ingredient is the **explorer**, not the **installer**. This tests it: seed banking with depth-3
solutions found by an **interpreter-backed search** (a tool), then measure whether depth-3 held-out coverage
rises.

## Method

Substrate `list`, families.LIST_PRIMS (the SAME 16-op DSL as eval/C21 — the search is written directly on
`families.py`, NOT C12's 23-op `decompose_lib`, so there is no vocabulary mismatch). No external model.

- **Explorer (tool-search, CPU-only):** for each depth-3 TRAIN task, brute-enumerate the interpreter over the
  substrate's own primitives (BFS with global state-dedup, max_depth 3) to DISCOVER an op-sequence correct on
  all task examples; render it to code via `families.reference_code`. This is the fixed-4B-allowed toolchain
  (interpreter + enumeration), no external model. (C12 found model-*guided* search ≈ *brute* at depth-3, so
  the crack is composition-structure + interpreter, not the model's planning; brute is the purest, most
  reproducible explorer. Framed honestly as "interpreter-backed search," not "the model's planning.")
- **Seed set = C21's EXACT depth-1+2 self-harvest pairs + the tool-found depth-3 pairs.** The ONLY delta from
  C21's `banked1` is the added depth-3 seeds; everything else (LoRA r32/α64/3ep/lr2e-4, the 130 depth-1+2
  pairs, 8-visible `ident_prompt`) is held identical.
- **Bank:** QLoRA-SFT single-shot prompt→code → `banked_tool`.
- **Eval:** on a **frozen, paired** held-out set (generated once, reused by every arm) at depths 2/3/4,
  n=40/depth, with **behavioral function-signature dedup** — any eval task whose function (outputs on a fixed
  probe set) matches ANY training rule is excluded (catches alternate decompositions, not just equal
  target_ops). Primary metric: **think coverage@16** (matched exactly to C21). Secondary: **no-think greedy@1
  and coverage@16** (deployable, banked-into-weights vs test-time search). base vs banked_tool, identical harness.

## Predictions (locked)

- **P1 (explorer works):** tool-search solves ≥ 90% of depth-3 TRAIN tasks (interpreter enumeration cracks
  depth-3 that monolithic sampling gets ≈0) → build ≥ 100 tool-found depth-3 pairs.
- **P2 (THE test — does the installer install, given the explorer?):** banked_tool depth-3 **think**
  coverage@16 ≥ 0.15 AND ≥ base + 0.10, with ≥ 5 DISTINCT depth-3 tasks unlocked (not one fluke). Base depth-3
  ≈ 0.00 (upper 95% CI for 0/40 ≈ 0.09), so the bar clears binomial noise. **Refuted if banked_tool depth-3 ≤
  base + 0.05** (banking cannot install depth-3 even with perfect training data — a deeper representational
  wall, C19).
- **P3 (generalization, not memorization):** the depth-3 lift is on the frozen, behaviorally-deduped held-out
  set (disjoint rules from training) — so any gain is compositional generalization to NOVEL depth-3 rules.
- **P4 (deployable + next rung):** report whether the depth-3 gain survives in no-think greedy@1 (banked into
  weights, not just test-time search), and whether depth-4 coverage rises (the next rung now partly samplable).

## Decision mapping

- **CONFIRMED** (P2 holds): tool-seeded banking crosses the depth-3 wall where self-banking (C21) could not →
  the C21 recipe is validated: **tools explore, banking installs.** The frontier is extendable rung-by-rung by
  seeding each rung with tool-search. Combined with C21, the precise mechanism: self-training is the installer,
  external search is the explorer, and both are required.
- **REFUTED** (P2 fails): even a strong, clean depth-3 training set does not install depth-3 (coverage stays
  ~0) ⇒ a deeper wall than C21 implied — banking cannot install depth-3 at all (consistent with C19's
  thin depth-3 representation). A bigger, more pessimistic negative.

## Controls / honesty (from the adversarial review)

- Search locked to families.LIST_PRIMS (16 ops, families' PARAM ranges); harvested tasks re-verified true-
  depth-3 by the collapse-rejected generator.
- No leakage: hidden examples never enter training text (search targets visible I/O; code renders a program,
  no labels); the hidden set is only an execution-oracle for which programs to keep — identical to C21.
- Frozen paired eval set + behavioral (function-signature) dedup vs the UNION of all training tasks.
- P2 bar set above base's binomial upper CI; report per-task cov_full counts + # distinct depth-3 tasks
  unlocked, not just the mean.
- Report deployable no-think greedy@1 to separate banked-into-weights capability from test-time think-search.

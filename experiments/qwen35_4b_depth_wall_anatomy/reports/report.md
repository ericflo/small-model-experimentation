# Qwen3.5-4B Depth-Wall Anatomy Report

## Summary

Insight-first anatomy of the fixed 4B's compositional wall (the "depth-3 frontier" every C11/C12 headline
routes through), with **pre-registered predictions** (`reports/prereg.md`) logged before each phase.
(Phase 0) A behavioral min-depth audit of ALL existing substrate tasks found the wall was **mismeasured**:
40% of nominal depth-3 tasks are shallower-equivalent, the frozen model has **never** solved a true
full-depth-3 task monolithically, and C12's decompose search really solved 17% of true depth-3 (not 0.40)
— retro-corrections landed in C12 (commit f6c2ca7). (Phase 1) On the verified factorial grid the
pre-registered **information-destruction hypothesis died** (k=0 deep compositions are NOT solvable;
op type barely matters) and a steeper law replaced it: **solve odds fall ~30× per composed op** —
against a 63-op space, identification beyond the first primitive runs at only **~2× better than chance**,
walling at depth 2. (Phase 2, decisive) A three-condition discriminator shows the wall is **100%
identification, 0% execution**: told the pipeline, the model executes at 0.90–1.00 through depth 4;
shown every intermediate state, it still cannot segment chains into the depth-1 identifications it
performs at 0.88. One mechanism — *the fixed 4B is a reliable compiler starved of hypothesis search* —
quantitatively retro-explains C10–C12 (decompose search works, its ~2× guidance efficiency, banking's
coverage bound, feedback's failure, verify≫generate).

## Research Program Fit

`structured_execution_and_compilers` + the insight-generation goal: convert the arc's central qualitative
story ("the depth wall") into measured structure — what the wall is made of (artifact / serial depth /
information destruction), and where it lives (hypothesis identification vs execution).

## Method

- **Phase 0 (CPU)**: exact BFS behavioral min-depth over all primitive pipelines vs ALL 18 examples, for
  every existing M1/M2/C12 task; restratify recorded solves. Predictions P0a/P0b.
- **Phase 1 (verified factorial grid)**: 17 cells (depth d ∈ 1–5 × destructive-ops k ∈ 0–3), n=25/cell,
  425 tasks, generator **rejects** collapsed compositions (BFS to depth min(d−1,3); d=5 may retain
  d4-equivalents — caveat). Destructive set (fixed in prereg): ops whose intermediates are unrecoverable
  from I/O (filters, dedup/unique, take/drop, chunk_sum, mod, abs, running_max). Monolithic thinking
  greedy@1 + pass@6 (hidden-graded), per-candidate visible/hidden pass, thinking length, and the first-op
  letter-logit rank (planner slice). Predictions P1–P6, P9.
- **Phase 2 (discriminator)**: same verified tasks (d {2,3,4} × k {0,2}), three conditions — **bare** I/O
  (identify + execute), **plan-given** (pipeline stated; execution only), **intermediates-shown** (state
  chains visible; observability restored). Predictions P7–P8.

## Results

### Phase 0 — the wall was mismeasured (P0a, P0b confirmed)

| finding | value |
| --- | --- |
| nominal d3 tasks that are behaviorally ≤ d2 | 40% (M1 6/15; C12 16/40; M2 25%) |
| monolithic TRUE depth-3 solves, entire corpus | **0** (all recorded d3 solves were collapsed tasks) |
| C12 decompose on collapsed vs true d3 | 16/16 vs 4/24 (17%) |
| destruction signal after collapse control (M2 true-d2) | k=0: 6/8 solved; k≥1: **0/8** |

### Phase 1 — verified factorial grid: the destruction hypothesis dies, a steeper law appears

pass@6 (n=25/cell, hidden-graded, verified-depth tasks):

| d\k | 0 | 1 | 2 | 3 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.88 | 0.72 | — | — |
| 2 | **0.16** | 0.04 | 0.08 | — |
| 3 | 0.00 | 0.04 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 |

- **P1 REFUTED**: transparent-only (k=0) compositions do NOT stay solvable at depth ≥3 (0.00 at d3–d5;
  predicted ≥0.4). **P2 REFUTED**: at fixed depth, destruction count barely matters (logistic coefficients
  per transparent vs destructive op: −3.24 vs −3.87 — statistically indistinguishable; depth-only AIC
  112.7 vs two-parameter 111.9). The destruction signal that motivated the hypothesis (M2 true-d2 k=0:
  6/8) was **primitive-mix luck in an n=8 cell** — the controlled n=25 grid eliminates it.
- What replaces it is a clean quantitative law: **the odds of solving fall ~30× per composed op**
  (logistic slope ≈ −3.5/op), uniform across op types. Anchored at d1 (0.88/0.72), this predicts d2 ≈ 0.17
  and d3 ≈ 0.008 — matching the observed 0.16 and ~0.01. Since blind guessing over the 63-op vocabulary
  would cost ~63× per op, the model's identification power beyond the first primitive is **only ~2× better
  than chance**. On genuinely novel compositions the wall is at depth **2**, not 3.
- **P5**: false-passes remain ~nil on verified tasks (3/156 visible-passers hidden-fail) — visible-test
  selection stays lossless; the C2 wall stays absent on this substrate.
- **P9 partially refuted**: first-op letter-logit rank is poor *everywhere* (median 7–13 of 23) with no
  clear destruction effect — the planner cannot identify the first op of any genuinely novel composition
  much better than chance, consistent with the ~2×-over-chance law.
- **P6** (exploratory): thinking length is budget-saturated (mean 434→492 tokens from d1→d5 against the
  512 cap; solved vs failed identical at matched depth) — uninformative at this budget; a budget sweep
  would be needed to test the serial-workspace account.

### Phase 2 — identification vs execution: the wall is 100% identification

pass@4 by cell (n=20/cell, same verified tasks, three prompting conditions):

| cell | bare I/O | plan-given | intermediates-shown |
| --- | ---: | ---: | ---: |
| d2k0 | 0.05 | **1.00** | 0.30 |
| d2k2 | 0.10 | **1.00** | 0.15 |
| d3k0 | 0.00 | **1.00** | 0.10 |
| d3k2 | 0.00 | **1.00** | 0.00 |
| d4k0 | 0.00 | **1.00** | 0.05 |
| d4k2 | 0.00 | **0.90** | 0.00 |

- **P7 CONFIRMED (strongest form)**: told the pipeline, the model executes essentially perfectly at every
  depth and destruction level. The compositional wall contains **zero execution deficit**.
- **P8 REFUTED**: full observability of intermediate states barely rescues identification (≤0.30). Each
  adjacent state-hop is a depth-1 identification the model does at 0.88 — but it cannot **segment** a shown
  chain into those solvable pieces and re-compose. (Caveat: chains add prompt clutter; a
  segmented-presentation follow-up quantifies this.)
- Convergent constant: the grid's ~2×-over-chance-per-op identification law independently matches C12's
  finding that model-guided search beat brute-force enumeration by only ~2× in efficiency — two
  measurements, one constant.

## Controls

Verified-depth generation (BFS rejection) removes the collapse artifact from the grid itself. The
destructive/transparent classification was fixed in the prereg before data. The oracle solves 425/425.
Phase-2's plan-given condition controls task content exactly (same tasks, information added).

## Oracle Versus Deployable Evidence

All measures deployable (visible info only; hidden-graded). Reference oracle bounds everything at 1.0.

## Interpretation

**The fixed 4B is a reliable compiler, not a hypothesis-searcher.** The compositional wall decomposes as:
(i) a large measurement artifact (40% shallower-equivalents; true monolithic depth-3 was always 0);
(ii) **zero** execution deficit (plan-given ≈ 1.00 through depth 4); (iii) a hard **identification** wall —
on genuinely novel compositions the model identifies each additional composed op at only ~2× better than
chance (odds ∝ ~30⁻ᵈ against a 63-op space), collapsing at depth 2 — insensitive to op type (destruction
hypothesis refuted) and barely helped by seeing intermediate states (it cannot segment chains into the
depth-1 identifications it can do).

This one mechanism retro-explains the arc quantitatively:
- **C12's decompose+interpreter search worked** because it externalizes exactly what the model lacks —
  segmentation + hypothesis search — leaving only per-step ranking and execution (which are cheap for it).
- **C12's guidance was only ~2× better than brute force** — the same ~2×-over-chance constant, measured
  independently.
- **C11's banking is coverage-bounded** because SFT teaches production/execution patterns, but the wall is
  identification: you cannot bank hypotheses you cannot identify.
- **M2's execution-feedback failure**: feedback flags wrong outputs, but the binding constraint is
  identifying the right hypothesis — which not even explicit intermediate states unlock.
- **C10's verify ≫ generate**: verifying a *given* program is execution-shaped (the model's strength);
  generating requires identification (its weakness).

Deployment corollary: for a fixed small model, **plan-conditioned execution is nearly free capability**;
the scarce resource is hypothesis search, which cheap external tools (enumeration + an interpreter)
supply. Division of labor: tools identify, the model compiles.

### Limitations
- One substrate family (list-of-int primitives), one model; the 63-op hypothesis space is enumerable
  (real-world identification may differ in structure). d5 verification capped at BFS depth 3
  (d4-equivalents possible). Thinking budget 512 saturated (P6 untestable here). Intermediates-shown
  carries prompt-format burden (segmented-presentation follow-up pending).

## Next Experiments

- **Segmented-presentation probe**: present each step's transition as separate mini-examples — if that
  rescues identification, the deficit localizes to *segmentation* specifically.
- Budget sweep at fixed depth (does thinking length become diagnostic once unsaturated — serial workspace?).
- Two-alternative identification (given two candidate pipelines, pick the consistent one) — measures
  identification as discrimination, isolating it from generation.
- Cross-substrate: does the ~30×/op identification decay constant transfer to other primitive families?

## Artifact Manifest

See `artifact_manifest.yaml`.

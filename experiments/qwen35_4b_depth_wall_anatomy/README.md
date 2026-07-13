# Qwen3.5-4B Depth-Wall Anatomy

**Status:** finished

## Research Program

- Program: `structured_execution_and_compilers`. Mission: **understand** the fixed Qwen3.5-4B's
  compositional frontier — insight-first: laws and mechanisms, not another lever.
- Attacks the arc's center of mass: every C11/C12 headline routes through "the depth-3 wall." This
  experiment decomposes that wall into measurable parts, with **pre-registered predictions**
  (`reports/prereg.md`) logged before each phase ran.

## Question

What actually sets the compositional frontier of the fixed 4B on the contamination-free substrate?
Candidate axes, cleanly separable:

1. **Substrate artifact** — nominal depth-d compositions with behavioral min-depth < d
   (`sort_asc∘reverse ≡ sort_desc`). Phase 0 audits ALL existing M1/M2/C12 tasks by exact BFS.
2. **Serial depth** — composition *length* per se (the implicit C11/C12 story).
3. **Information destruction** — ops whose intermediates are unrecoverable from visible I/O
   (filters/dedup/mod/abs/...) making the composition un-invertible / unidentifiable.
4. **Identification vs execution** — can the model *execute* a deep pipeline it is told, and does
   *seeing* intermediates rescue it (Phase 2 discriminator: bare vs plan-given vs intermediates-shown)?

## Phase 0 results (CPU audit of existing data — both predictions confirmed)

- **P0a:** 40% of nominal depth-3 tasks (M1 and C12 sets) collapse to behavioral min-depth ≤2.
- **P0b (strong form):** monolithic true-depth-3 solves = **0 across the entire corpus**; every recorded
  depth-3 solve rode on collapsed tasks. C12's decompose search: 16/16 collapsed vs 4/24 (17%) true.
  → C12's claim + report retro-corrected (commit f6c2ca7).
- Destruction signal survives the collapse control: true depth-2, k=0 → 6/8 solved; k≥1 → 0/8.

## Design (Phases 1-2)

- **Verified factorial grid**: 17 cells (d 1–5 × k 0–3), n=25/cell, generator **rejects** collapsed
  compositions (exact BFS to depth min(d−1,3)). Monolithic thinking greedy@1 + pass@6, hidden-graded;
  per-candidate visible/hidden (false-pass slice); thinking length; first-op letter-logit rank (planner).
- **Discriminator** on the same tasks (d {2,3,4} × k {0,2}): bare vs plan-given vs intermediates-shown.
- Predictions P1–P9 in `reports/prereg.md`, all logged in advance.

## Run

```bash
../../.venv/bin/python scripts/min_depth_audit.py                           # Phase 0 (CPU)
../../.venv/bin/python scripts/run_grid.py --n-per-cell 25 --k-samples 6    # Phase 1
../../.venv/bin/python scripts/run_discriminator.py --per-cell 20           # Phase 2
../../.venv/bin/python analysis/analyze_grid.py                             # analysis vs predictions
```

## Results

Full write-up in `reports/report.md`. Three findings, each pre-registered:

1. **The wall was mismeasured** (Phase 0): 40% of nominal depth-3 tasks are shallower-equivalent; true
   monolithic depth-3 was always 0; C12 retro-corrected.
2. **The destruction hypothesis died; a steeper law replaced it** (Phase 1, P1/P2 refuted): on verified
   novel compositions, solve odds fall **~30× per composed op** regardless of op type — identification
   beyond the first primitive runs at **~2× better than chance** (63-op space), walling at depth 2.
3. **The wall is 100% identification, 0% execution** (Phase 2, P7 strongest form): plan-given → 0.90–1.00
   through depth 4; intermediates-shown barely helps (≤0.30) — the model can't *segment* chains into the
   depth-1 identifications it does at 0.88.

**One mechanism explains the arc:** the fixed 4B is a **reliable compiler starved of hypothesis search** —
why decompose+interpreter search works, why its guidance was only ~2× over brute force (same constant!),
why banking is coverage-bounded, why execution feedback failed, why verify ≫ generate. Division of labor:
tools identify, the model compiles.

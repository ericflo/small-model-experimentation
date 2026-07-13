# Qwen3.5-4B Coverage vs Selection: anatomy of the generation wall

**Status:** finished

## Research Program

- Program: `structured_execution_and_compilers`
- Program question: is the fixed 4B's generation wall (C13/C16) a COVERAGE deficit (right program never
  proposed) or a SELECTION deficit (proposed but not selected)?
- Prior anchors: C10 (verify ≫ generate), C13/C16 (the wall is generation, not execution).

## Question

When bare identification ≈ 0 at depth 3, is the correct program *never sampled* or *sampled but not
selected*? This decides whether *cleverer access* (a better selector) can beat *sample more*, and where.

## Hypothesis

Pre-registered (`reports/prereg.md`): shallow depths are SELECTION-bound (coverage ≫ single-shot; a
selector recovers much of the ceiling); a crossover depth d* exists beyond which the wall is COVERAGE-bound
(coverage ≈ 0 even at K=32); the model's own verifier picks hidden-correct programs above chance among
execution-consistent candidates.

## Setup

- Model: Qwen3.5-4B (only permitted model), thinking on, budget 512, repo-standard sampling (T=0.6/top_p
  0.95). Inference-only, no training.
- Tasks: fresh verified-depth, collapse-rejected `list` + `register` compositions, depths 1–4, n=20/depth,
  8 visible + 8 hidden examples. Seed 707 (held out).
- Per task: draw K=32 identification samples; execute each vs visible and hidden.
- Selectors (no hidden labels): first@1 (single-shot), coverage@k (oracle ceiling = sample-more), vfilter
  (majority behavior among visible-passers), mverify (C10 thinking-verifier ranks visible-passers).
- Primary metric: per-depth coverage@k curve + deployable accuracy of each selector vs the ceiling; the
  crossover depth d*.

## Run

Smoke: `python scripts/run_anatomy.py --families list --smoke`
Full: `python scripts/run_anatomy.py --families list register --n-per-depth 20 --depths 1 2 3 4 --K 32 --budget 512 --seed 707 && python scripts/analyze.py`

## Results

**The wall is COVERAGE, not selection.** Selection is free: max(coverage − vfilter) = 0.00 in every cell,
90% of visible-passers pass hidden, and execution-filter / model-verifier / random-among-consistent all
recover the coverage ceiling identically. Single-shot undersells 2–5× (first@1→cov@32: list d2 0.10→0.30,
register d2 0.15→0.60, d3 0.05→0.25). Coverage-wall depth is set by hypothesis-space size (list crossover
d\*=3; register survives to d4). See `reports/report.md`, `analysis/wall_anatomy.png`, `runs/verdict.json`.

## Interpretation

You cannot beat sample-more by better selection — selection is already free; the wall is PROPOSAL/coverage.
The lever is shifting proposal (tools C12 / banking C11-C12). Confirms C10 (selection is plumbing); sharpens
C13/C16; explains C16's register floor as coverage-driven. Refuted own selection-centric predictions P3/P4.

## Knowledgebase Update

- Program evidence updated: `research_programs/structured_execution_and_compilers/evidence.md` (C17)
- Claim ledger updated: C17 added

## Artifacts

- `src/families.py`, `src/code_env.py`, `src/gen_lib.py`
- `scripts/run_anatomy.py` — sample K, grade vs visible+hidden, verifier-select
- `scripts/analyze.py` — table, coverage@k curves, wall map, verdicts, figure
- `data/tasks_{list,register}.jsonl`, `runs/anatomy.json`, `runs/verdict.json`
- `analysis/wall_anatomy.png`, `reports/prereg.md`, `reports/report.md`

# Qwen3.5-4B Cross-Family Laws

**Status:** finished

## Research Program

- Program: `structured_execution_and_compilers`
- Program question: are the C13–C15 ladder constants properties of the *model* or of the *list* substrate?
- Prior anchors: C13 (compiler/generation split), C15 (context composes; simulation length-fragile).

## Question

C11–C15 all rest on ONE substrate (integer-list pipelines). Is transcription ≈ 1.0 / simulation-decays /
identification-walls a **model-level law** or a **list artifact**? Rerun the ladder on genuinely different
fresh families and see which constants replicate.

## Hypothesis

Pre-registered (`reports/prereg.md`): transcription ≥ 0.85 all depths both new families; a family-invariant
normalized simulation-decay constant; bare identification ≈ 0 at depth ≥ 3; ordering trans > sim > bare.

## Setup

- Model: Qwen3.5-4B (only permitted model), thinking on, budget 512. Inference only, no training.
- Task source: three fresh families — **list** (16 int-list prims, anchor), **string** (13 char-edit
  prims), **register** (12 ops on a 3-int machine). Depth-graded, execution-verified, behavioral
  min-depth-BFS collapse-rejected. 100 verified tasks/family (25 × depth 1–4).
- Baseline / anchor: the list family measured through the identical harness.
- Controls: identical collapse-rejection across families; oracle 100% pass; family-aware `Step:` parser
  (unit-tested; caught a spurious string-sim-0.00 artifact before scoring).
- Primary metric: per-depth accuracy on transcription (plan→code, pass@1), simulation (final-state
  exact-match), bare identification (I/O→code, pass@4).
- Hidden-label boundary: identification graded by executing the model's `transform` against hidden I/O.

## Run

Smoke: `python scripts/run_family.py --family string --smoke`
Full: `for f in list string register; do python scripts/run_family.py --family $f --n-per-depth 25 --depths 1 2 3 4 --budget 512 --seed 303; done && python scripts/analyze.py`

## Results

**Verdict: SCOPED.** Transcription is one invariant flat line at ~1.00 across all families (**compiler
LAW**). Identification walls in all families, gap ≥ 0.84 at depth ≥ 3 (**generation-wall LAW**). Simulation
is substrate-dependent: register robust (0.92→0.72), list decays (1.00→0.56), string floored (0.24→0.00) —
C15's decay constant was list-specific. See `reports/report.md` and `analysis/crossfamily_ladder.png`.

## Interpretation

C13 is **promoted** to a model-level law across substrates ("tools identify, the model compiles" is
general). C15 is **narrowed**: externalize simulation to a tool only where the state representation is
expensive to track; for compact integer state the model simulates reliably. New sub-law: the wall's floor ≈
f(hypothesis-space size, simulability).

## Knowledgebase Update

- Program evidence updated: `research_programs/structured_execution_and_compilers/evidence.md` (C16)
- Program backlog updated: representation-swap + op-menu-size floor tests (see report Next Experiments)
- Claim ledger updated: C16 added

## Artifacts

- `src/families.py` — the three families (prims, state, input gen, collapse-rejection BFS, oracle)
- `scripts/run_family.py` — family-generic ladder runner
- `scripts/analyze.py` — cross-family table, normalized decay, verdict, figure
- `data/tasks_{list,string,register}.jsonl` — verified tasks
- `runs/ladder_{list,string,register}.json`, `runs/verdict.json`
- `analysis/crossfamily_ladder.png`
- `reports/prereg.md`, `reports/report.md`, `reports/artifact_manifest.yaml`

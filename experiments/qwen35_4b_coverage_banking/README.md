# Qwen3.5-4B Coverage Banking: does banking shift the proposal distribution?

**Status:** finished

## Research Program

- Program: `structured_execution_and_compilers`
- Program question: C17 said the wall is proposal-COVERAGE and only shifting the proposal distribution can
  beat sample-more. Does banking the model's own verified solutions do that — by CONCENTRATION or EXPANSION?
- Prior anchors: C17 (wall is coverage, selection free), C11/C12 (banking lifts pass@k).

## Question

Does QLoRA-SFT on the fixed 4B's OWN execution-verified solutions CONCENTRATE existing coverage into the
greedy sample, or EXPAND the coverage ceiling (propose programs the base never sampled — cross the wall)?

## Hypothesis

Pre-registered (`reports/prereg.md`): banking lifts single-shot at trained depths (P1); does not beat
sample-more at k=1 (P2); CONCENTRATION dominates EXPANSION (P3); no lift at untrained depth 4 (P4).

## Setup

- Model: Qwen3.5-4B (only permitted model). No teacher — targets are the model's own verified outputs.
- Harvest (TRAIN, list depths 1–3, 90 tasks): sample K=40 think, keep hidden-correct, cap 12/task → 80
  `{prompt, code}` SFT pairs (by depth {1:49, 2:24, 3:7}).
- Bank: QLoRA-SFT r32/alpha64, 3 epochs, single-shot prompt→code (no-think).
- Eval (HELD-OUT, disjoint, depths 1–4, 20/depth): 4 arms {base, banked} × {no-think, think}, greedy@1 +
  coverage@16. Sample-more baseline = base-think coverage@16.

## Run

Smoke: `python scripts/harvest.py --smoke` then `python scripts/train_lora.py --train data/train.jsonl --out runs/smoke_adapter --smoke` then `python scripts/eval_ladder.py --tag base --smoke`
Full: see `runs/launch.sh` (harvest→train→eval base/banked no-think→analyze) + `runs/launch2.sh` (base/banked think).

## Results

**Banking does BOTH.** Depth 1: **CONCENTRATION** (think greedy@1 0.60→0.80, ceiling flat). Depth 2:
**EXPANSION** (coverage@16 0.15→0.45, 3×, on held-out tasks — proposes compositions the base never sampled;
diversity even drops, so proposal mass moved onto correct programs). Depths 3–4: no move (too few/no examples).
Does not beat think sample-more at k=1, but banking+sample-more > base+sample-more. See `reports/report.md`,
`analysis/banking_coverage.png`, `runs/verdict.json`.

## Interpretation

The coverage wall is not immovable by self-training: banking expands the ceiling for held-out tasks where it
has enough verified examples (C17's proposal-shift lever working). To push deeper (depth 3+, where sampling
harvests ≈0), seed with tool-augmented harvest (C12). Refuted own P3 (concentration-only) optimistically.

## Knowledgebase Update

- Program evidence updated: `research_programs/structured_execution_and_compilers/evidence.md` (C18)
- Claim ledger updated: C18 added

## Artifacts

- `scripts/common.py`, `scripts/harvest.py`, `scripts/train_lora.py`, `scripts/eval_ladder.py`, `scripts/analyze.py`
- `data/train.jsonl` (verified SFT pairs), `data/{train,eval}_tasks.jsonl`
- `runs/eval_{base,banked,base_think,banked_think}.json`, `runs/verdict.json`
- `analysis/banking_coverage.png`, `reports/prereg.md`, `reports/report.md`
- `runs/banked_adapter/` — trained QLoRA adapter (omitted from git; regenerate via harvest+train)

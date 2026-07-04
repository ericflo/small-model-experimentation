# Qwen3.5-4B Wall Climbing: does banking shallow composition unlock deeper coverage?

## Research Program

- Program: `structured_execution_and_compilers`
- Program question: can banking be iterated to CLIMB the compositional wall — does installing depth-1+2 unlock
  depth-3 sampling?
- Prior anchors: C18 (banking installs/expands within a depth), C17 (wall is coverage), C12 (tool-search
  extends the frontier), C11-M4 (banking is coverage-bounded).

## Question

If we bank ONLY depth-1+2 self-solutions, does the banked model now sample depth-3 compositions the base never
could — bootstrapping the frontier upward by pure self-training?

## Hypothesis

Pre-registered (`reports/prereg.md`): install works (P1); depth-3 unlocks ≥ +0.05 → CLIMBABLE, else DEPTH-LOCAL
(P2); no two-rung leap to depth-4 (P3); Round-2 climb if unlocked (P4).

## Setup

- Model: Qwen3.5-4B (only permitted model). No teacher — targets are the model's own execution-verified code.
- Harvest depth-1+2 only (20 d1 + 90 d2 tasks, K=40 think) → 130 pairs {d1:47, d2:83}, no depth-3 examples.
- Bank: QLoRA-SFT r32/alpha64, 3 epochs, single-shot prompt→code → banked1.
- Eval: coverage@16 (think, held-out, disjoint) at depths 2/3/4, base vs banked1, n=25/depth.

## Run

Smoke: `python scripts/harvest.py --smoke`
Full: `bash runs/launch_r1.sh` (harvest → train banked1 → eval base → eval banked1) then `python scripts/analyze.py`

## Results

**DEPTH-LOCAL.** Depth 2: base 0.12 → banked1 **0.36** (install works, tripled, held-out). Depth 3 (unlock
test): base 0.00 → banked1 **0.00** — zero unlock. Depth 4 unchanged (0.04). Self-banking installs only depths
already samplable; it cannot climb the wall. See `reports/report.md`, `analysis/wall_climbing.png`,
`runs/verdict.json`.

## Interpretation

Composition skill does not length-generalize across a depth. Completes the wall picture: depth-3 is not
represented (C19), not steerable (C20), not reachable by banking-shallow (C21). The only way up is to seed
each rung externally with tool-search (C12), then bank — self-training is the installer, not the explorer.

## Knowledgebase Update

- Program evidence updated: `research_programs/structured_execution_and_compilers/evidence.md` (C21)
- Claim ledger updated: C21 added

## Artifacts

- `scripts/harvest.py` (depth-configurable, adapter-loadable for round 2), `scripts/train_lora.py`,
  `scripts/eval_ladder.py`, `scripts/analyze.py`, `scripts/common.py`
- `data/train.jsonl` (130 verified depth-≤2 pairs), `data/{train_tasks,eval_tasks}.jsonl`
- `runs/eval_{base,banked1}.json`, `runs/verdict.json`, `analysis/wall_climbing.png`
- `runs/banked1_adapter/` — trained adapter (~180MB, moved out of repo; regenerate via harvest+train)

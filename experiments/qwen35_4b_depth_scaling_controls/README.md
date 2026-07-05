# Qwen3.5-4B Depth Scaling & Controls: saturation, data-vs-compute, depth-4

## Research Program
- Program: `structured_execution_and_compilers`
- Three follow-ups to C23 (depth-3 install is data-limited): (1) does the dose curve saturate past 640?
  (2) is the gain data-diversity or compute? (3) does the recipe repeat one rung deeper (depth-4)?
- Anchors: C23 (data-limited dose curve), C22 (weak crossing), C13-C21 (the recipe).

## Setup
- Model: Qwen3.5-4B only. Explorer: CPU interpreter brute-search over the 16-op DSL (no external model).
- Arm 1: doses 640/1280 depth-3 tool-pairs (nested; 555/1156 distinct functions). Reuse C23 base/40/160/640.
- Arm 2: banked_up40 (40 distinct x16=640 examples, matched size/mixture/steps to train_640) vs train_640 vs N=40 dose.
- Arm 3: banked_d4 (d12+640d3+320d4) vs SCAFFOLD-only (banked_640) on a fresh depth-4 held-out set.
- QLoRA r32/a64 epochs=3; frozen paired held-out, func-sig + op-composition dedup (0 leakage: d3 0/2305, d4 0/318).

## Run
Smoke: `python scripts/harvest2.py` (CPU, builds all data). Full: harvest2 + train the 3 adapters + `bash runs/launch_eval.sh` + `python scripts/analyze.py`.

## Results
- **Arm 1: NO saturation through 1280.** cov@16 0.00/0.087/0.212/0.375/0.537 at N=0/40/160/640/1280 (distinct funcs 39/153/555/1156); deployable greedy@1 -> 0.188. (Adjacent CIs marginally overlap at n=80.)
- **Arm 2: data-DIVERSITY, not compute.** N=40 0.087 -> up40 0.163 (+compute, within noise) -> train_640 0.375 (+diversity, cleanly significant).
- **Arm 3: recipe repeats one rung deeper, weakly.** base 0.00, scaffold-transfer 0.067, banked_d4 0.183 (~3x); greedy flat 0.033 (test-time-only), CIs marginally overlap at n=60. Depth-3 guardrail 0.425 (no forgetting).
See `reports/report.md`, `analysis/scaling_controls.png`.

## Interpretation
The C13->C24 ladder-climbing recipe is diversity-driven (banking DISTINCT explorer-found solutions) and rung-repeatable (depth-4 installs weakly over the depth-3 transfer baseline). No saturation through 1156 distinct depth-3 functions.

## Knowledgebase Update
- Program evidence: `research_programs/structured_execution_and_compilers/evidence.md` (C24)
- Claim ledger: C24 added

## Artifacts
- `scripts/harvest2.py`, `scripts/train_lora.py`, `scripts/eval_ladder.py`, `scripts/analyze.py`
- `data/train_{1280,up40,d4}.jsonl`, `data/eval_frozen_d{3,4}.jsonl`, `data/tool_depth{3_2560,4}.jsonl`
- `runs/eval_*.json`, `runs/verdict.json`, `analysis/scaling_controls.png`, `reports/{prereg,report,design_review}.md`
- Adapters (~180MB each) moved out of repo; regenerable.

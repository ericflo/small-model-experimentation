# Qwen3.5-4B Depth-3 Dose-Response: data-limited or representational cap?

**Status:** finished

## Research Program
- Program: `structured_execution_and_compilers`
- Question: C22 crossed the depth-3 wall but weakly — is the install data-limited or a representational cap?
- Anchors: C22 (tool-seeded banking crossed weakly), C19 (depth-3 rep is a thread), C21 (self-banking can't seed).

## Hypothesis
Pre-registered (`reports/prereg.md`, hardened by `reports/design_review.md`): DATA-LIMITED if depth-3 think coverage rises with N (top-dose CI above low-dose CI); CAP if it plateaus.

## Setup
- Model: Qwen3.5-4B only. Explorer: CPU interpreter brute-search over the 16-op DSL (640/640 solved, no external model).
- Doses (nested): C21 depth-1+2 pairs + N tool-found depth-3, N in {40,160,640}; QLoRA r32/a64, epochs=3.
- Eval: ONE frozen paired held-out set, dedup by function-sig AND op-composition (0 leakage), n=80 depth-3. Think coverage@16 (Wilson CIs) + no-think deployable at top dose; depth-2 guardrail.

## Run
Smoke: `python scripts/tool_harvest.py --smoke`
Full: `python scripts/tool_harvest.py --n-depth3 640 && (build train_{40,160,640}.jsonl) && bash runs/launch.sh && python scripts/analyze.py`

## Results
**DATA-LIMITED.** Depth-3 think cov@16: 0.00 -> 0.087 -> 0.212 -> 0.375 (N=0/40/160/640), monotone, top-dose CI [0.28,0.48] non-overlapping with low-dose [0.04,0.17]. Deployable scales: no-think cov 0.00->0.338, greedy@1 0.00->0.10 at N=640. 0 leakage (novel rules). See `reports/report.md`, `analysis/dose_response.png`.

## Interpretation
The deep wall is a DATA bottleneck, not a hard representational cap: tool-seeded banking scales with #solutions into deployable single-shot. Completes the C13->C23 recipe (explorer + installer + data throttle).

## Knowledgebase Update
- Program evidence: `research_programs/structured_execution_and_compilers/evidence.md` (C23)
- Claim ledger: C23 added

## Artifacts
- `scripts/tool_harvest.py`, `scripts/train_lora.py`, `scripts/eval_ladder.py` (frozen + func-sig & op-comp dedup + leakage), `scripts/analyze.py` (Wilson CIs)
- `data/train_{40,160,640}.jsonl`, `data/eval_frozen.jsonl`, `runs/eval_*.json`, `runs/verdict.json`, `analysis/dose_response.png`, `reports/{prereg,report,design_review}.md`
- Adapters (~180MB each) moved out of repo; regenerable.

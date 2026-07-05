# Qwen3.5-4B: Thinking vs the Lookahead Wall

## Research Program
- Program: `structured_execution_and_compilers` / `test_time_reasoning_budget`
- Question: does test-time THINKING breach the lookahead wall (C25) with no training? (reactivates the C9 lever)
- Anchors: C25 (lookahead wall + banking lifts it), C9 (thinking is an unused lever), C23 (base think depth-3=0).

## Setup
- Model: Qwen3.5-4B only. list 16-op DSL (32 op/param combos). 80 min-depth-verified true-depth-3 held-out (reuse C25; used 40).
- PRIMARY metric = think->RANK vs no-think->RANK (channel-matched to C25, parse-immune). HEADLINE = STEP 1 (goal 3 ops away, no intermediate state = the only clean lookahead test).
- Budgets B in {0, 1024, 2048}. No training (test-time only).

## Run
`python scripts/run_thinking.py --n 40 --budgets 0 1024 2048 --steps 1 2 3` then `python scripts/analyze.py`

## Results
Thinking does NOT breach the lookahead wall. Step-1 stays at chance (0.025->0.050->0.075 at B=0/1024/2048; CIs overlap). But thinking amplifies RECOGNITION: step-3 (1 away) 0.275->0.600, step-2 (2 away) 0->0.325 -- the lift scales inversely with lookahead distance. Internal-brute-force refuted (step-1 flat; traces are meta-reasoning not enumerate-and-test). See `reports/report.md`, `analysis/thinking_lookahead.png`.

## Interpretation
For the multi-step PLANNING/lookahead gap, TRAINING (banking, C25: 0.013->0.138) is required; test-time thinking (C26) only amplifies RECOGNITION. Reconciles with C23 (base think single-shot depth-3 = 0).

## Knowledgebase Update
- Program evidence: `research_programs/structured_execution_and_compilers/evidence.md` (C26)
- Claim ledger: C26 added

## Artifacts
- `scripts/think_rank.py` (think->rank, chunked op-scoring for long thinking prefixes), `scripts/run_thinking.py` (batched thinking generation), `scripts/analyze.py`
- `data/eval_frozen_d3.jsonl` (reused from C25), `runs/results.json`, `runs/traces_B*.json`, `runs/verdict.json`, `analysis/thinking_lookahead.png`, `reports/{prereg,report,design_review}.md`

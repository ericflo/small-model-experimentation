# Qwen3.5-4B: Do Banking and Thinking Stack?

**Status:** finished

## Research Program
- Program: `structured_execution_and_compilers` / `test_time_reasoning_budget`
- Question: do banking (C25, installs lookahead-distance ranking) and TEST-TIME thinking (C26, amplifies recognition) compose?
- STATUS: honest NARROW BASELINE + scope caveat (banked model trained no-think) -> motivates bank-the-thoughts.

## Setup
- 2x2 {base, banked_1280} x {no-think, think} on per-step next-op RANKING (think->RANK, channel-matched), n=40 true-depth-3 held-out. Banked in-run; base inherited from C26 (identical slice/harness).

## Run
`python scripts/run_thinking.py --tag banked1280 --adapter <C24 banked_1280> --budgets 0 1024 2048 --steps 1 2 3` then `python scripts/analyze.py`

## Results
RECOGNITION (step-3): additive stacking -- 0.275 -> 0.525 (banking) -> 0.850 (banking+thinking), interaction ~0.00. PLANNING (step-1): no stacking -- banking lifts 0.025->0.175, test-time thinking adds ~0 (base 0.075, banked 0.150). Banked thinking channel intact. See `reports/report.md`, `analysis/stack.png`.

## Scope (important)
The banked adapter was trained NO-THINK, so 'test-time thinking adds no planning' is about test-time thinking on a model never trained to reason about this task -- NOT that thinking is useless for planning. Clean test = bank-the-thoughts.

## Knowledgebase Update
- Program evidence: (C27, Exploratory)
- Claim ledger: C27 added (scoped)

## Artifacts
- `scripts/run_thinking.py` (--adapter), `scripts/analyze.py`, `runs/results_banked1280.json`, `runs/traces_banked1280_B*.json`, `runs/verdict.json`, `analysis/stack.png`, `reports/{prereg,report}.md`. Reuses C24 banked_1280 (out of repo).

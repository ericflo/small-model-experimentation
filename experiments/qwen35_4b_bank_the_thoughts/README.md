# Qwen3.5-4B: Bank the Thoughts

**Status:** finished

## Research Program
- Program: `posttraining_and_adaptation`
- Question: does banking the REASONING (plans) install more usable depth-3 than banking the ANSWER? (the clean version of C26/C27)
- Phase 1 (this): synthetic forward-decomposition plans. Phase 2 (deferred): the model's own rejection-sampled thoughts.

## Setup
- Three fresh QLoRA from base, MATCHED data (identical prompt+code; only the trace differs): A={prompt->code}, T={prompt->plan->code}, T_corrupt={same code, mismatched plan}. Plans built from execution-verified op-sequences (input->op1->state->...->output).
- Eval frozen held-out depth-3 (n=80, 0-leakage): deployability (coverage@16 + greedy@1), step-1 planning ranking.

## Run
`python scripts/synth_traces.py --n 256 && python scripts/build_train.py` then train A/T/Tcorrupt via `train_lora_think.py`, eval via `eval_ladder.py` (deploy) + `run_thinking.py` (step-1), `python scripts/analyze.py`.

## Results
Banking correct PLANS (T, think) deploys depth-3 better than banking ANSWERS (A): cov@16 0.325 vs 0.200, greedy@1 0.050 vs 0.025. CONTENT-CAUSAL: T_corrupt (wrong plans) collapses to 0.113 (below A). TEST-TIME CHANNEL: T no-think = 0.013. Resolves C26/C27 (thinking helps once the reasoning is banked). See `reports/report.md`, `analysis/bank_thoughts.png`.

## Limits
Synthetic plans (not model's own -- Phase 2); T uses more test-compute than A (T_corrupt controls content); step-1-T-think eval too slow to complete; single seed.

## Knowledgebase Update
- Claim ledger: C28 (Promising)

## Artifacts
- `scripts/synth_traces.py`, `scripts/harvest_thoughts.py` (Phase 2), `scripts/build_train.py`, `scripts/train_lora_think.py`, `scripts/eval_ladder.py`, `scripts/run_thinking.py`, `scripts/analyze.py`
- `data/harvest_{thoughts,answers}.jsonl`, `data/train_{A,T,Tcorrupt}.jsonl`, `runs/eval_deploy_*.json`, `runs/results_s1_*.json`, `runs/verdict.json`, `analysis/bank_thoughts.png`, `reports/{prereg,report,design_review}.md`
- Adapters (~180MB each) moved out of repo.

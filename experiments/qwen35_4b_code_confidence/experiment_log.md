# Qwen3.5-4B: Does the Confidence Toolkit Survive on Real Code? Experiment Log

## Scaffold

Created as a new experiment scaffold; `src/` seeded with the shared generation/judging library (gen_lib, coverage_utils, code_env) from the coverage-line experiments.

## 2026-07-07 smoke

`eval_code_conf.py --n 4 --k 2` end-to-end on 4 MBPP records: generation, hidden-assert execution, mean-logprob, P(True) all populate. Offline HF cache serves the SANITIZED config ('prompt' field, test_list as JSON string) — adapted in `load_sanitized_test`.

## 2026-07-07 full run, attempt 1: CUDA OOM

`--n 260 --k 8` crashed in `mean_logprobs`: full-vocab float32 log-softmax over whole sequences OOMs at batch 8 x ~800 tokens (24 GB card). Fix: keep logits bf16, log-softmax in float32 over 128-token sequence chunks, batch 4, `empty_cache()` per batch. Lesson consistent with the repo's WSL2 GPU notes: do NOT launch competing GPU jobs while recovering.

## 2026-07-07 full run, attempt 2: clean

PID 428223, ~35 min: 244 problems x 9 candidates = 2,196 generations, all executed against hidden asserts (overall pass rate 0.70), checkpoint saved before logprob phase, mean_logprob for 2,196/2,196, p_true for 2,163 (33 unparseable completions skipped by design). Output `runs/code_conf.json`.

## 2026-07-07 analysis + hardening

`analyze.py` initial verdict; then hardened in-place: paired bootstrap significance for every selection delta and for within-problem AUROC vs the length baseline; P(True)-ranked abstention curve added alongside mean-logprob. Key numbers: within-problem AUROC logprob 0.693 / P(True) 0.738 / length 0.548; selection at k=8 P(True) 0.762 > self-consistency 0.717 (p=0.005) > random 0.696; visible-test execution 0.816; oracle 0.844; greedy solvability AUROC P(True) 0.837. Verdict: toolkit transfers, hierarchy inverts (single-token P(True) readout beats sequence mean-logprob). Claim C46.

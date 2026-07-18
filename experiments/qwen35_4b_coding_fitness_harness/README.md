# Coding Fitness Harness (cognitive-core program)

Shared measurement infrastructure for the cognitive-core coding program
(owner pivot, 2026-07-17): after the menagerie aggregate was shown NOT
to transfer to real agentic coding (McNemar p=1.00 vs base), the mission
became installing REAL coding capability into base Qwen/Qwen3.5-4B via
designed contamination-free curricula, proven by TRANSFER to held-out
coding. This cell is the fast fitness signal every curriculum bet is
scored against: a standalone, adapter-evaluable HumanEval + MBPP pass@1
harness (greedy, execution-graded) reusing the C46 record/prompt/execute
machinery through the pinned model_override vLLM runner, so base and
every trained composite are graded on the identical path.

**Status:** finished · since 2026-07-17 · harness built, validated
(HF-vs-vLLM cross-check to 1/164, determinism confirmed, grader
sanity-checked), base baselines recorded; reused as-is by every
curriculum lifecycle.

## Base Qwen/Qwen3.5-4B baselines (greedy pass@1, this harness)

- HumanEval: 0.7622 (125/164) — cross-validated vs the C46 HF run
  (126/164); the base is ALREADY a strong single-function coder.
- MBPP (full test, first 200): 0.5650 (113/200).
- Agentic (duet-eval gen4, external, measurement-only): 8/35 = 0.229 —
  the base is a WEAK multi-step coding agent.

The 76% vs 23% gap is the program's target: the 4B can write a function
but cannot drive a multi-step coding task. Curricula aim at that gap
(planning, state-tracking across steps, debugging loops), measured here
for retention + fast signal and on the agentic eval for the real target.

## Interface

`.venv-vllm/bin/python scripts/eval_pass1.py --dataset {humaneval,mbpp}
--n N [--model-override /path/to/merged/composite] --out FILE`
(greedy, seed 0, execution-graded; `--model-override` fingerprint-checks
a merged Qwen3.5-4B composite so trained arms are graded identically).

Caveat: numbers are vLLM-greedy; never compare to an HF run — keep every
arm on this harness.

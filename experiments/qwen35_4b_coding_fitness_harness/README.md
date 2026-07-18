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

**Status:** finished · CORRECTED 2026-07-18 · was mistakenly measuring THINKING-OFF at a 512-token budget (suppressed base ~13-19pp); FIXED to thinking-on, 8192 budget, <think>-aware extraction. Base thinking-on: HumanEval 147/164 (89.6%), MBPP 151/200 (75.5%). ALL prior coding-program measurements were thinking-off-suppressed and are superseded.

## Base Qwen/Qwen3.5-4B baselines (greedy pass@1, this harness)

CORRECTED 2026-07-18 — thinking-on, 8192-token budget (the harness had
been mistakenly measuring thinking-OFF at 512 tokens, suppressing base
~13-19pp; all prior coding-program numbers are superseded):

- HumanEval: 0.8963 (147/164) — thinking-on; NEAR CEILING (little SFT
  headroom). Was 0.7622 thinking-off.
- MBPP (full test, first 200): 0.7550 (151/200) — thinking-on; some
  headroom (53% of traces hit the 8k cap / force-close). Was 0.5650.
- Agentic (duet-eval gen4, external, measurement-only): 8/35 = 0.229 —
  already thinking-on; the base is a WEAK multi-step coding agent.

The 90% vs 23% gap is the program's target: the 4B writes functions well
but cannot drive a multi-step coding task. Function completion is nearly
maxed; the agentic gap is the real prize (RLVR). Curricula are measured
here thinking-on for retention + fast signal, and on the agentic eval
for the real target.

## Interface

`.venv-vllm/bin/python scripts/eval_pass1.py --dataset {humaneval,mbpp}
--n N [--model-override /path/to/merged/composite] --out FILE`
(greedy, seed 0, execution-graded; `--model-override` fingerprint-checks
a merged Qwen3.5-4B composite so trained arms are graded identically).

Caveat: numbers are vLLM-greedy; never compare to an HF run — keep every
arm on this harness.

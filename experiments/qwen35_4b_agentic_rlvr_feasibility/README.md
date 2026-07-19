# Qwen35 4B Agentic RLVR Feasibility

**Status:** in-progress · since 2026-07-19 · GATE PASSED (single-GPU agentic GRPO physically
works for Qwen3.5-4B) + warm-start prerequisite established. Owner /goal (2026-07-19): install
real-codebase agentic coding into the 4B via pi-coding-agent + OpenEnv, harvest→SFT→RLVR, proven
by transfer, run relentlessly.

## Research Program

- Program: `agentic_breadth_installation` (cognitive-core coding sub-program).
- Question: can real-codebase agentic coding capability be INSTALLED into base Qwen/Qwen3.5-4B by
  RLVR (execution-reward GRPO) on execution-verified coding environments, on ONE 24GB GPU?
- Prior anchors: base HumanEval 89.6% (near ceiling) but duet-eval agentic ~21-23% (a genuine
  real-codebase capability gap, not a behavior gap — see FINDINGS + claim C60); the recurring
  lesson from ~20 prior SFT installs is that happy-path SFT deletes the verifier-conditioned
  recovery policy, and on-policy verifier-conditioned recovery (RLVR) is the one parameter-local
  bright signal (`research_programs/agentic_breadth_installation/evidence.md`).

## What this cell establishes (M0 gate + integration)

1. **The single-GPU GRPO colocate loop physically CLOSES for Qwen3.5-4B** (claim C61). C49 warned
   vLLM 0.24 runtime-LoRA is a silent no-op for Qwen3.5-4B PEFT; the gate test (a movable
   length-reward smoke asserting the served policy shifts) shows the reward moves 0.028→0.38 within
   3 steps — the vLLM-served policy DOES update, so C49 does NOT bite TRL colocate.
2. **The recipe that makes two 4B copies fit one 24GB card** (all three needed):
   - monkeypatch `vllm.LLM.__init__(enforce_eager=True)` — TRL exposes no field, and Qwen3.5's
     hybrid mamba/attention arch HANGS on torch.compile/CUDAGraph capture;
   - `model_init_kwargs={"dtype":"bfloat16"}` — else HF loads fp32 (16GB) → OOM;
   - `vllm_gpu_memory_utilization=0.55` + `vllm_enable_sleep_mode=True` + small `vllm_max_model_length`
     → KV cache ~1.7GB, peak ~22.6GB.
3. **The whole agentic RLVR machine runs end-to-end**: TRL 1.8 `GRPOTrainer(environment_factory=CodingEnv)`
   drives the 4B (thinking-on) through pi-mirroring tools (read_file/write_file/list_dir/run_bash)
   multi-turn, executes them (tools/failure 0), grades with `get_reward` (test pass), does GRPO steps.
4. **The warm-start prerequisite** (claim C61): the RAW base does not produce reward VARIANCE at the
   feasible group size (num_generations ≤ 4 on 24GB; 8 OOMs) — it explores 1–2 tool calls (~120 tok)
   then quits WITHOUT writing, so every rollout leaves the stub → identical reward → zero advantage →
   zero gradient. A narrow SFT warm-start (teach loop discipline: explore→edit→test→iterate,
   commit-from-partial) is required before RLVR has signal — matching the program's prior lesson.

## Environment (the central artifact)

`scripts/coding_env.py` — `CodingEnv` for TRL `environment_factory`: `reset(**row)` copies a real
OSS repo + stubs a tested function + returns the task; methods `read_file/write_file/list_dir/run_bash`
mirror pi-coding-agent's tool interface (so training transfers to pi deployment); `get_reward()` is
engagement-gated (no-edit=0) with dense partial credit and full-pass=1.0. Pooled-reuse safe.

## Reproduce

`scripts/m0_loopclose.py` — the loop-closure gate smoke. `scripts/agentic_grpo_test.py` — the
end-to-end agentic GRPO integration on toolz stub-a-function tasks. `scripts/gen_repo_tasks.py` +
`scripts/loop_repo.py` — real-repo task generation + a pi-style harvest loop. Run under `.venv-vllm`
(has trl 1.8 + openenv + vllm 0.24 + peft). Serving/recipe details in `reports/report.md`.

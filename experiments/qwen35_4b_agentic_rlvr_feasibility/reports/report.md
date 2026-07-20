# Qwen35 4B Agentic RLVR Feasibility — Report

**In-progress (2026-07-19).** Establishes that single-GPU agentic RLVR (execution-reward GRPO)
physically works for Qwen3.5-4B, the exact recipe to fit it on one 24GB card, and that a narrow SFT
warm-start is a prerequisite for the raw base to produce a learning signal. Successor stages: harvest
(pi-coding-agent trajectories) → SFT warm-start → RLVR → transfer to a held-out real-coding split.

## Stack

Installed into `.venv-vllm` (no breakage to vllm 0.24 / transformers 5.13 / torch 2.11): `trl==1.8.0`,
`openenv==0.4.1`, `trackio`. TRL 1.8 provides `GRPOTrainer(environment_factory=...)` where an env
instance's non-underscore methods become the model's tools, TRL drives the multi-turn tool-calling
loop AND owns the token logprobs GRPO needs, `reset(**row)` receives the dataset row and returns the
task text, `get_reward()` is the terminal reward, and instances are pooled/reused. TRL ships Qwen3.5
chat templates + tool-call parse schemas (`qwen3_5_think.jinja`, `qwen3_5_schema`) so agentic
tool-calling with thinking is natively supported.

Why not pi rollouts directly for GRPO: pi-coding-agent drives the base headlessly and produces
faithful text trajectories (ideal for harvest/SFT and for eval), but records NO per-token logprobs
(`grep logprob` across pi dist + all trajectories = 0). GRPO needs logprobs, so RLVR generation is
TRL-driven through `CodingEnv` (whose tools mirror pi's interface, so it trains for pi deployment);
pi remains the harvest + eval scaffold.

## M0 GATE — the single-GPU GRPO colocate loop CLOSES

C49 (Confirmed) says vLLM 0.24 runtime-LoRA is a silent no-op for Qwen3.5-4B PEFT adapters, which
threatens TRL colocate's per-step LoRA→vLLM sync (it could train a frozen policy). Test: 20-step
GRPO on base Qwen3.5-4B, colocate vLLM, bf16 LoRA, a trivial MOVABLE reward, asserting the served
policy shifts. A "push-shorter" completion-length reward (base is verbose → starts 0.028) reached
**0.38 within 3 steps** — impossible if the policy were frozen. **Verdict: the loop closes; C49 does
NOT bite TRL colocate.** (The trend is noisy/oscillating → tune lr/KL for stability; not a blocker.)

### The recipe that fits two 4B copies on one 24GB card (all three required)

1. **enforce_eager**: monkeypatch `vllm.LLM.__init__` to inject `enforce_eager=True`. TRL has no
   config field for it, and Qwen3.5's hybrid GDN/Mamba+attention arch HANGS on torch.compile /
   CUDAGraph capture (also true for plain `vllm serve` — must pass `--enforce-eager`).
2. **bf16 model load**: `model_init_kwargs={"dtype":"bfloat16"}`. `bf16=True` alone only sets the
   training autocast; HF still loads the model in fp32 (16GB) → vLLM init OOM. bf16 load = 8.5GB.
3. **memory split**: `vllm_gpu_memory_utilization=0.55` + `vllm_enable_sleep_mode=True` + a small
   `vllm_max_model_length`. Below ~0.55 the KV cache computes NEGATIVE (weights + activation eat the
   budget); at 0.55 KV ≈ 1.7GB and peak ≈ 22.6GB. Sleep mode frees ~12.8GB between generation and the
   optimizer step (time-slices the two copies). `beta=0` skips the reference model.

## Agentic integration — RAN END-TO-END

`GRPOTrainer(environment_factory=lambda: CodingEnv())` on toolz stub-a-function tasks, thinking-on,
num_generations=2, max_completion_length=1536, max_tool_calling_iterations=6, vllm_max_model_length
4096: 3 GRPO steps at ~25s/step, fit 24GB. `tools/failure_frequency 0` — the pi-mirroring tools
execute correctly. **The whole agentic RLVR machine works on one 4090.** (num_gen 4 + completion 3072
+ len 8192 crashed "CUDA device not ready" = too-long multi-turn sequences in the logprob forward;
num_gen 8 OOMs — so the feasible group size on 24GB is ≈ 4.)

## The learning blocker → warm-start prerequisite

Across configs (± engagement scaffold, ± strong system prompt, num_generations up to the 24GB
ceiling of ~4): `reward_std = 0`, `frac_reward_zero_std = 1` → zero advantage → zero gradient
(loss 0). Logged completions show the base thinks briefly, emits 1–2 read/list tool calls (~120–150
tokens), then STOPS on turn 2 without ever writing a fix — the same non-engagement diagnosed on real
duet tasks (61% no-diff). Every rollout leaves the stub → identical reward → no variance. On 24GB the
group size can't be grown enough (num_gen 8 OOMs) to catch the base's rare (~10%) successes.

**Conclusion:** RLVR from the raw base is blocked by low engagement × small feasible group size. A
NARROW SFT warm-start — teach the read→edit→test→iterate loop discipline and commit-from-partial
(NOT success-only minimization, which deletes recovery per the program graveyard) — is the
prerequisite that makes the base engage → produce reward variance → RLVR learns. This is the
successor stage: harvest pi-coding-agent's own execution-verified completed trajectories (it engages
the base well), convert to multi-turn tool-calling SFT rows, warm-start, then RLVR from there.

## Operational constraint: HOST RAM (15GB), not just VRAM

This box has **15GB of system RAM**, and it is a binding constraint independent of the 24GB GPU.
Plain `from_pretrained` stages the whole 4B through CPU and leaves a resident copy: the SFT trainer
grew to ~9.2GB RSS and the kernel OOM-killer silently killed it 2-5 steps into training, twice, with
NO traceback in the log (`dmesg`: `Out of memory: Killed process (python3) anon-rss:9165056kB`).
Symptom to recognize: training dies mid-run, no Python error, adapter never saved.

FIX: `model_init_kwargs={"dtype": "bfloat16", "low_cpu_mem_usage": True}` — loads shard-by-shard
straight onto the GPU. Trainer RSS dropped 9.2GB -> 2.2GB and training ran clean. Also keep
`max_length` modest (6144) for multi-turn agentic rows. Long-running jobs should be launched with
`setsid nohup ... &` so they survive shell/agent teardown, and outputs written under
`large_artifacts/` (a crash wiped the /tmp scratchpad and cost ~4h of harvest).


## SFT warm-start result: engagement INSTALLED, variance now a difficulty problem (2026-07-19)

Harvested 89 completed+passing multi-turn trajectories (self-contained tasks, 93% yield in 18 min —
~30x more efficient than real-repo stub tasks at ~6%), converted to TRL conversational rows
(assistant turns carry the base's OWN harvested reasoning + tool_calls; tool results masked), and SFT'd
a LoRA with the Qwen3.5 THINK training template + assistant_only_loss (24 steps, loss 0.23 -> 0.055),
then merged to a composite.

Measured in the agentic GRPO env (verify_engagement.py, toolz stub tasks):

| metric | raw base | + warm-start |
|---|---|---|
| completions/mean_length | ~130 tok | **517** (4x) |
| tools/call_frequency | 1.5 | **3.38** (2.3x) |
| reward / reward_std | 0 / 0 | 0 / **0** |

**The warm-start installs ENGAGEMENT** — the model now runs the explore->edit->test loop instead of
quitting after 1-2 reads, exactly what it was trained to do, and it transfers from the self-contained
training tasks to real-repo tasks. **But reward_std is still 0**: it passes 0/8 toolz episodes, so
GRPO still has no within-group advantage. The blocker has CHANGED SHAPE: not engagement, but task
DIFFICULTY CALIBRATION. GRPO needs tasks the policy passes SOMETIMES (~30-70%); toolz stub tasks are
~0% for the warm-start and the self-contained tasks are ~100%. Next: calibrate per-task pass rate for
the WARM-STARTED policy across a task mix, select the 30-70% band, and run RLVR on that band.

### Additional single-4090 constraints found here
- TRL sleeps the colocate engine at **level=2**, which offloads the 8.5GB of weights to HOST RAM and
  reloads on every wake -> OOM-killed on a 15GB box. Monkeypatch `vllm.LLM.sleep` to force **level=1**
  (frees only KV, weights stay on GPU): same VRAM time-sharing, no host hit. With util 0.50 this runs.
- TRL refuses `assistant_only_loss` for "vision-language models", and Qwen3.5-4B is multimodal-capable,
  so the default `AutoProcessor` triggers that guard. Pass a **tokenizer** as `processing_class`
  (our data is pure text) -> `_is_vlm=False` -> masked multi-turn SFT works.
- Adapters trained against multimodal `Qwen/Qwen3.5-4B` carry a `language_model.` key segment that the
  vendored text-only merger rejects; use `merge_peft.py` (PEFT `merge_and_unload`) instead.

## Next Experiments

- Harvest pi trajectories on solvable real-repo tasks → multi-turn tool-calling SFT warm-start.
- RLVR (GRPO) from the merged warm-start; reward = FAIL_TO_PASS AND PASS_TO_PASS, no-network sandbox
  (reward-hacking guard); kill rule: RLVR must beat matched-compute sample-more on held-out pass-rate.
- OpenEnv packaging of `CodingEnv` (create_app) for portability + pi/harvest reuse; mine duet-eval
  scenarios into OpenEnv envs under a strict train/test firewall (never eval on trained scenarios).

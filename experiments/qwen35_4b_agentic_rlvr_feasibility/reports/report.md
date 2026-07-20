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

## Breaking the zero-gradient wall: it was THREE bugs, not task difficulty (2026-07-19)

The previous section concluded the blocker was "task DIFFICULTY CALIBRATION". That was wrong, and
the way it was wrong is the lesson. `reward_std=0` is consistent with both "the band is miscalibrated"
and "the harness is broken", and those were only distinguishable by tracing INDIVIDUAL rollouts —
GRPO logs the batch MEAN, which cannot tell "policy never engaged" from "policy engaged and failed".
Adding a per-rollout reward trace (`RLVR_REWARD_LOG`) resolved in one run what inference had not in
many. **Instrument the unit the metric aggregates over.**

### Bug 1 — tool strictness (the dominant one)

The policy emits a spurious `<parameter=content>None</parameter>` on EVERY tool call. TRL passes tool
args straight through, so `read_file(path=..., content=None)` raised TypeError and **86% of all tool
calls failed** (`tools/failure_frequency 0.8621`). The agent burned all 8-12 iterations on rejected
calls, never reached `write_file`, and every rollout scored exactly 0.0 → tied group → no advantage.

Verified NOT self-inflicted: the SFT rows carry 0% spurious args (462 tool calls audited) and the
training template renders them correctly, so the warm-start did not teach it — it is the policy's own
generation habit. Fix: `_tolerant` drops undeclared kwargs; `functools.wraps` sets `__wrapped__` so
`inspect.signature()` still resolves the ORIGINAL signature and tool JSON schemas generate correctly
(this is what the earlier `*a, **k` attempt got wrong → `DocstringParsingException`). Missing REQUIRED
args still raise. pi-coding-agent also tolerates extra args, so this makes the training env behave
like the deployment target rather than papering over a defect.

### Bug 2 — the LM head, not attention, dominates training memory

Qwen3.5-4B's vocab is **248320**, so logits are enormous. Measured at seq 4096, micro-batch 1,
gradient checkpointing ON:

| what | activations |
|---|---|
| full loss over all positions | 12.04 GiB |
| loss over last 128 positions only (head cost ≈ 0) | 2.60 GiB |

**The LM head alone is 9.44 GiB — 78% of all activation memory** (predicted 9.47 from
`seq × 248320 × 10 bytes`; body is only ~0.65 GiB/1k). This is what made 8k rollouts need 32.8 GiB on
a 24 GiB card. Fix: TRL `use_liger_kernel=True` → `LigerFusedLinearGRPOLoss`, a chunked loss that
never materializes the `[seq, 248320]` logits. Safe here only because our LoRA does not target
`lm_head` (TRL raises otherwise — Liger reads `lm_head.weight` directly and would silently train a
frozen head).

`flash-linear-attention` was installed for the gated-delta-net fast path and is numerically correct
(loss diff 0.0008 vs the torch fallback) but bought only 14% (3.5 → 3.0 GiB/1k). **It was the wrong
20%.** Chasing the loud warning (`Falling back to torch implementation`) instead of measuring first
cost real time. NOTE: PyPI `flash-linear-attention` is BROKEN — it ships only `fla/layers` and
`fla/models`, no `fla/ops`, yet still flips `is_flash_linear_attention_available()` to True, which
hard-breaks every Qwen3.5 load. Install from GitHub source or not at all.

### Bug 3 — truncation made the task physically unsolvable

At `model_len 3072` the prompt (mostly ~1200 tokens of tool schemas) left ~1100 tokens for
read+think+write+test: `clipped_ratio 1.0`, i.e. 100% of episodes truncated before `write_file`. No
policy could have scored above 0. At 8192 → `clipped_ratio 0.25`.

### Result (same tasks, same policy)

| metric | before | after |
|---|---|---|
| `frac_reward_zero_std` | 1.0 (no gradient) | **0.0** |
| `reward_std` | 0 | **0.075** |
| edited solution.py | 0/4 | 3/4 |
| completion length | 1076 | 2164 |
| `tools/failure_frequency` | 0.8621 | 0.4583 |

GRPO now has advantage signal. Remaining: no full passes yet (rewards cluster at 0.15 = edited but
tests fail), so current variance is about ENGAGEMENT rather than SOLUTION QUALITY.

### Additional single-4090 constraints found here

- **`expandable_segments` must stay OFF under vLLM colocate.** It makes torch's allocator use the CUDA
  virtual-memory driver APIs (`cuMemCreate`/`cuMemMap`) that vLLM's sleep-mode `CuMemAllocator` also
  uses on the same device. The collision surfaces as `RuntimeError: CUDA driver error: device not
  ready` **in backward** (torch tags driver-API failures via `C10_CUDA_DRIVER_CHECK`), and only once
  sequences are long enough to force segment growth — which is exactly why the short-completion M0
  gate passed and every long-rollout config crashed.
- Sleep **level 2** is now correct (it was ruled out earlier for OOM-ing a 15GB host, but that was
  before `low_cpu_mem_usage` cut trainer RSS 9.2 → 2.2 GiB). It frees 12.12 GiB between generation and
  the optimizer step. Full vLLM teardown would recover only ~1.5 GiB more and is not worth it.
- **Unloading either model does not enable long rollouts.** Trainer-alone peaks, whole card free:
  1024→12.22, 2048→15.70, 4096→22.66, 8192→**36.59 GiB**. The constraint was never coexistence.
- `pgrep -f rlvr_band` matches the agent's OWN shell wrapper and reports a dead run as "running".
  Check `nvidia-smi --query-compute-apps` instead.

## Next Experiments

- Harvest pi trajectories on solvable real-repo tasks → multi-turn tool-calling SFT warm-start.
- RLVR (GRPO) from the merged warm-start; reward = FAIL_TO_PASS AND PASS_TO_PASS, no-network sandbox
  (reward-hacking guard); kill rule: RLVR must beat matched-compute sample-more on held-out pass-rate.
- OpenEnv packaging of `CodingEnv` (create_app) for portability + pi/harvest reuse; mine duet-eval
  scenarios into OpenEnv envs under a strict train/test firewall (never eval on trained scenarios).

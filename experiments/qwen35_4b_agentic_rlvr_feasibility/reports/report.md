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

## Tool-call failures: the policy emits a UNION of all tools' parameters (2026-07-20)

After the three fixes above, `tools/failure_frequency` was still 0.46. Per-call tracing
(`RLVR_TOOL_LOG`) showed the policy emits a blend of every tool's parameter names — `path`,
`content`, `command` — on each call, and fails whenever the argument that tool actually requires is
absent from the blend:

| tool | what it emits | failure rate |
|---|---|---|
| `read_file` | path + spurious content | 0% (extras dropped) |
| `run_bash` | content + path, often no `command` | 55% (16/29) |
| `write_file` | path, sometimes no `content` | 45% (10/22) |

Two plausible hypotheses were tested and **refuted by measurement** before this one was found:
1. *"content is truncated mid-write so `</parameter>` never closes"* — doubling
   `max_completion_length` 4096→8192 left the failure rate identical at 41%.
2. *"the parser drops multi-line values"* — `transformers`' `_xml_inline` uses `re.DOTALL` and
   handles multi-line content correctly, verified directly against the real parser.

Fix: a missing REQUIRED argument returns an instructive error naming what is missing, what was
provided, and the exact accepted parameter list, instead of raising `TypeError`. The agent retries
inside the same episode instead of burning an iteration it needs to reach `write_file`. This is
recovery, not reward-hacking — a correct call is still required to make progress.

Cumulative effect on tool failures: **0.86 → 0.26 (tolerant dispatch) → ~0.00 (instructive recovery)**.

### Trainable context ceiling on a 24GB card

With Liger, `model_len 12288` runs typical episodes but OOMs in `create_causal_mask` on the longest
ones (crashed at step 3). **8192 is the proven ceiling**; at 8192 with `max_completion_length 6144`
the run is stable with `clipped_ratio 0` (no truncation at all) and `tools/failure_frequency 0`.

### Healthy-loop reference numbers

For comparison when this regresses: step ~260s at 4 rollouts × 10 tool iterations, `reward_std`
0.07–0.46, `frac_reward_zero_std` 0, rollout rewards distributed across {0.0, 0.15, 1.0} with real
passes appearing.

### Train/test firewall

RLVR trains on 20 of the 27 band tasks; **7 are held out** (`base_convert`, `dedup`, `flood_fill`,
`json_pointer`, `point_in_polygon`, `roman`, `stats2`) and written to `<out>_split.json` at the
TRAINING entry point, so the eval cannot silently include a trained-on scenario. `calibrate_trl.py
--tasks <split.json>` reads the `holdout` list directly.

## DEPLOYMENT TRUTH: the model is far stronger in pi than in our harness (2026-07-20)

Everything above measures the model in OUR tool loop. Driving the SAME merged warm-start through
pi-coding-agent itself, on the SAME 7 held-out tasks:

| harness | held-out mean pass rate |
|---|---|
| ours (`calibrate_trl.py`, TRL template + TRL parser) | 0.486 |
| **pi-coding-agent (deployment truth)** | **0.810** |

5 of 7 held-out tasks pass 100% of the time through pi (`stats2`, `roman`, `base_convert`, `dedup`,
`point_in_polygon`); engagement 19/20. **Our harness was underselling the model by ~1.7x.**

Two consequences, both serious:

1. **The 0.2–0.8 "difficulty band" is a harness artifact.** It was computed in our loop. In pi most
   of those tasks are saturated, so the band does not describe the deployment policy.
2. **The GRPO run was optimizing against that artifact.** It trains on tasks selected for being hard
   *in our harness*, and its reward is our harness's execution loop. Improvements there are not
   guaranteed to be improvements in the scaffold the model actually ships in. The 8-step partial run
   is preserved (`trl_grpo_partial_history.json`) but its band needs re-deriving from pi.

### Correction: our tools do NOT mirror pi's interface

Earlier sections of this report claim `CodingEnv`'s tools "mirror pi-coding-agent's interface".
**That is false**, and measuring through pi is what exposed it. pi's actual tools are:

    read, edit, write, bash

versus ours: `read_file, write_file, list_dir, run_bash`. Every name differs, and pi has an **`edit`**
tool (targeted string replacement) that we never trained on at all — in the first successful pi
episode the model solved the task using `read` ×2, **`edit` ×2**, `bash` ×3, i.e. its main
code-modification tool was one absent from training. Training installed tool habits for an interface
that does not exist downstream.

### The pi recipe (was never codified from the earlier probe)

    vllm serve <merged-composite> \
      --served-model-name qwen35-4b-pi8k --port 8420 \
      --enforce-eager --max-model-len 40960 --gpu-memory-utilization 0.90 \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml

    pi --provider kiln-local --model qwen35-4b-pi8k -p "<task>" --mode json --no-session < /dev/null

Each flag was required, and each was found by a failing run:
- **`--enable-auto-tool-choice --tool-call-parser qwen3_xml`** — without them vLLM returns
  `400 "auto" tool choice requires ...` and pi completes 0 tool calls. `qwen3_xml` matches Qwen3.5's
  `<tool_call><function=><parameter=>` format.
- **`--max-model-len 40960` + a pi model entry with `maxTokens 8192`** — pi sends its entry's
  `maxTokens` as `max_completion_tokens` on EVERY call, and vLLM rejects
  `prompt + max_completion_tokens > max_model_len`. With the stock 32768 entry the agent died the
  moment the conversation passed ~8k. (Added additively as `qwen35-4b-pi8k` in
  `~/.pi/agent/models.json`; original backed up to `models.json.bak-preRLVR`.)
- **stdin closed** — pi is interactive by default and otherwise hangs.
- pi **auto-retries 3x with backoff**, so transport failures look like slow episodes, not errors.

### Why RLVR cannot be driven by pi rollouts directly

pi's distribution contains **no logprob support**, and TRL's GRPO requires `per_token_logps` to form
the objective. Policy-gradient RL over pi rollouts is therefore not possible without patching pi to
surface logprobs. The executable form of "RLVR from execution reward through pi rollouts" is
execution-filtered policy improvement: pi generates → execution verifies → fit the verified winners →
repeat. That needs no logprobs and is what the goal separately asks for ("harvest pi-coding-agent's
own execution-verified successful trajectories").

## The pi-driven loop CLOSED — and RFT made the policy WORSE (2026-07-20)

The full cycle ran end to end: pi generated 78 episodes on the train split → execution verified them
(63 passed, 81%) → the passing trajectories were fitted on top of the policy that produced them →
merged → re-evaluated through pi on the stratified holdout. **Result: a large regression.**

| | held-out mean pass | episodes timing out (exit 124) | mean episode secs |
|---|---|---|---|
| baseline (merged warm-start) | **0.606** | 13/32 | 364 |
| pi-RFT composite | **0.121** | **28/33** | 572 |

Every held-out task got worse or stayed at zero. This is a real, execution-verified NEGATIVE result,
recorded as such.

**Mechanism: the RFT policy never stops.** The regression is not "wrong answers" — it is timeouts.
Baseline timed out on 13/32 episodes; RFT on 28/33, at exactly the 600s wall clock, having looped
tool calls the whole time. The 1.6x slowdown is the same fact seen from the side.

**Why fitting SUCCESSFUL trajectories taught non-termination.** The harvested trajectories are long
(mean 12 assistant turns, up to 31) and 92% of ALL assistant turns are tool calls; only the final
turn of each is a clean stop. With `assistant_only_loss` weighting every assistant span equally, the
model saw "call another tool" ~11x more than "stop and answer", and LoRA at lr 1e-4 on top of an
already-working policy over-fit the continuation behaviour into a loop-forever regime. Fitting
*verified successes* is not automatically safe when the successes are long and stop-behaviour is rare
per-token.

**What this rules in for the next iteration** (hypotheses, pre-registered so the next run is a test):
1. Down-weight or cap length: fit only the LAST k turns, or add explicit terminal-stop emphasis, so
   "stop when done" is not drowned out ~11:1.
2. Much lower LR / fewer steps: 1e-4 full-strength SFT is too aggressive for polishing a working
   policy; try 2e-5, or KL-anchor to the warm-start.
3. Reward SHORT successes: prefer trajectories that solved in few turns, since those carry the
   stop-early signal the long ones dilute.
4. This is exactly the gap RLVR (advantage-weighted, penalize the timeouts) would close if pi
   surfaced logprobs — the execution-filtered SFT surrogate has no negative signal for looping.

The good policy (merged warm-start, 0.606) is preserved and remains the deployment baseline;
`merged/pi_rft` is kept only as the evidence for this finding.

## Iteration: the short-success + low-LR fix CONFIRMS the mechanism (2026-07-20)

Re-ran the loop with two of the pre-registered fixes: keep only trajectories with ≤8 assistant turns
(21 of 46 survive) and drop LR 1e-4 → 2e-5. Same harvest, same holdout, same harness.

| | held-out mean pass | timeouts (exit 124) | mean episode secs |
|---|---|---|---|
| baseline (warm-start) | **0.606** | 13/32 | 364 |
| pi-RFT v1 (all 46, lr 1e-4) | 0.121 | 28/33 | 572 |
| **pi-RFT v2 (≤8 turns, lr 2e-5)** | **0.576** | 12/33 | 363 |

**This confirms the v1 diagnosis.** Recovering almost the entire regression (0.121 → 0.576) by exactly
the two changes predicted to fix non-termination — with timeouts falling 28/33 → 12/33 and speed
returning to baseline — proves v1's collapse WAS the long-trajectory/high-LR loop-forever pathology,
not something inherent to fitting pi trajectories.

**But v2 still does not beat baseline (0.576 vs 0.606).** It rescued two previously-unsolved tasks
(`case_convert`, `schema_lite`: 0.00 → 0.33) and lost ground on three (`expr_eval`, `flatten`,
`pretty_bytes`), for a net slightly-negative move that is within noise at k=3. So the honest status is:
**execution-filtered SFT on pi's own successes has been made SAFE but not yet NET-POSITIVE.**

Why this is expected, not a surprise: fitting successes reinforces what the policy already does right
and has NO signal to suppress the failures (the timeouts, the wrong answers). It cannot push a task
from 0 → 1 unless a success for that task was harvested, and the tasks that most need help are exactly
the ones with the fewest harvested successes. The missing ingredient is a NEGATIVE gradient on the bad
rollouts — which is what advantage-weighted RLVR provides and what execution-filtered SFT structurally
cannot. On this hardware that means either (a) patching pi to surface logprobs so GRPO can consume its
rollouts, or (b) a preference method (DPO) over pi's passing-vs-timing-out trajectory pairs, which
needs no logprobs and directly penalizes the loop-forever behaviour. (b) is the next bet.

## DPO on the loop-forever pathology REGRESSED the policy (2026-07-20)

The pre-registered bet (b): DPO supplies the negative gradient RFT structurally lacks. Built it end to
end and it is a **NEGATIVE result** — worse than the null RFT.

**How the rejected side was actually constructed, and why it changed.** The plan was real pass-vs-
timeout pairs. Harvesting the timeouts turned out to be infeasible as designed: pi_rft's timeout is
NOT a clean multi-turn loop but a **single non-terminating generation** (measured on one episode: 92
stream deltas, 1 `message_end`, content frozen at "let me read the test file" while `<think>` ran to
the wall). The logging proxy records an exchange on stream COMPLETION, so a stream that never
completes logs nothing → every harvested rejected had 0 turns. (Two real wiring bugs surfaced on the
way and are worth keeping: pi must be served as `--served-model-name qwen35-4b-pi8k`, NOT `Qwen3.5-4B`
— the stale `pi_episode.py` docstring is wrong and a 404 model-name mismatch silently yields empty
trajectories; and the `kiln-harvest{i}` providers are pinned to base-port 8431, so a harvester on any
other base-port gets "Connection error".)

So the rejected side became a **synthetic decision-point negative**: at the solved state (last tool
result = `ALL PASS`), chosen = the real terminal stop, rejected = one more redundant tool call. Pure
stop-vs-continue, 45 pairs over 24 train tasks, firewall-clean. DPO trained cleanly and learned it
emphatically: `rewards/accuracies` 0 → **1.0**, `rewards/margins` 0 → **0.565**, loss 0.693 → 0.451.

**Deployment result (pi, 11-task holdout, k=3, same conditions as the 0.606 baseline):**

| | holdout mean pass | Δ vs baseline |
|---|---|---|
| warm-start baseline | **0.606** | — |
| pi-RFT v2 (short+lowLR SFT) | 0.576 | −0.030 (null) |
| **pi-DPO (synthetic stop-signal)** | **0.515** | **−0.091 (regression)** |

Per task, DPO rescued two (`expr_eval` 0.33→0.67, `case_convert` 0.00→0.33) but damaged four that were
solid (`min_heap` 1.00→0.33, `rle`/`pretty_bytes` 1.00→0.67, `allocate` 0.33→0.00). The three
persistent zeros (`json_pointer`, `schema_lite`, and now `allocate`) never move.

**Two lessons, both about mis-targeting.**
1. **Wrong target.** The holdout failure is looping *before* the task is solved; the synthetic
   negative only teaches termination *after* it is solved (a state those episodes never reach). The
   signal cannot fire where the failure lives.
2. **Right method, wrong dose — same fragility as RFT-v1.** Learned to accuracy 1.0 / margin 0.57 on
   45 pairs, the "prefer stop over act" preference generalized into *premature stopping / degraded
   acting* and broke tasks the policy already solved. This is the third time a narrow policy edit on
   the working 0.606 warm-start has failed to beat it (RFT-v1 crashed, RFT-v2 null, DPO regressed).

**Converging conclusion:** on this curriculum the 0.606 warm-start behaves like a local optimum that
narrow SFT/preference edits perturb *downward*. The remaining holdout headroom is concentrated in a
few tasks the model solves 0% of the time in *both* policies — that reads as a CAPABILITY gap, not a
termination-discipline gap, and termination framing (the loop-forever lens) addressed the wrong bottleneck.
The 0.606 warm-start remains the deployment policy; `merged/pi_dpo` is kept only as this finding's evidence.
Caveat: k=3, so per-task rates are noisy — but a solid task falling to 0.33 (min_heap) is a real
2-of-3 flip, and the direction (down, via damage to working tasks) is consistent across four tasks.

## Best-of-8 capability map + elicitation SFT: distillation works on the task, regresses deployment (2026-07-20)

To decide whether the persistent zeros are latent (harvestable) or absent, ran best-of-8 through pi on
the zero tasks with the warm-start. It cleanly split them:

| task | pass@8 (full) | mean partial reward | verdict |
|---|---|---|---|
| bellman_ford | 0.88 | — | reliably solvable (the earlier 0.00 was unlucky k=2 sampling) |
| topo_lex | 0.25 | — | partially solvable — real single-shot headroom |
| allocate | 0.12 | — | solvable, holdout (transfer-only) |
| deep_merge | 0.00 | 0.41 (max 0.55) | **edits, passes ~half the tests, TIMES OUT — never closes** |
| semver | 0.00 | 0.33 (max 0.53) | partial, close, times out |
| schema_lite | 0.00 | 0.21 | partial, times out |
| glob_match/case_convert/patch_apply/json_pointer | 0.00 | 0.13–0.18 (max 0.32–0.49) | partial, times out |

**CORRECTION (2026-07-21).** An earlier version of this section called these seven tasks "ABSENT —
never makes productive progress (meanR 0.00)". That was a codification ERROR: the `meanR` value came
from `dict.get(task, 0)` on a field that was never written to the result JSON, so a missing key read as
0.00. The episode-level rewards (above) show the opposite — the model DOES engage: it edits
`solution.py` on 3–8 of 8 tries and passes SOME tests every time (partial reward 0.13–0.41, up to 0.55
on deep_merge), but never ALL of them, and **every episode times out (exit 124) — it loops instead of
terminating**. So this is not a hard capability wall; it is "close-but-can't-finish + a termination
failure," which is materially more tractable. Lesson: verify against episode-level data, not a summary
field that may be absent.

**Best-of-12 ceiling result (2026-07-21).** Re-ran these seven at k=12 (timeout 240). It settles the
ceiling question decisively:

| task | pass@12 | mean partial | max | edits | timeouts |
|---|---|---|---|---|---|
| schema_lite | **1/12** | 0.31 | **1.00** | 11/12 | 12/12 |
| deep_merge | 0/12 | 0.36 | 0.65 | 11/12 | 12/12 |
| json_pointer | 0/12 | 0.20 | 0.49 | 9/12 | 10/12 |
| semver | 0/12 | 0.29 | 0.38 | 11/12 | 12/12 |
| case_convert, glob_match | 0/12 | 0.16–0.19 | 0.32–0.35 | 9–11/12 | 11/12 |
| patch_apply | 0/12 | 0.03 | 0.24 | 2/12 | 12/12 |

Two decisive conclusions:
1. **The capability is NOT absent.** `schema_lite` fully closed once in 12 (maxR 1.00) and most tasks
   reach real partial credit (deep_merge 0.65). These are rare-and-hard, not impossible.
2. **The universal failure is TERMINATION.** Across all 83 episodes the model engages, gets partway,
   cannot close, and **loops to the wall — 96% timeout (80/83 exit 124)**. Even the single schema_lite
   full solve itself timed out (it closed the task, then kept going instead of stopping). The
   bottleneck is finishing + terminating, NOT missing skill.

This reframes the whole program: the deployment ceiling is a **termination / finishing** problem, not
a capability wall. It also retroactively explains why DPO's "stop after solving" failed — these tasks
are almost never *solved*, so that signal never fires; the need is "recognize you are stuck, finish or
stop cleanly." The natural non-training lift is **execution-selected best-of-n at deploy**: run N pi
rollouts, keep the highest-scoring by the tests. Since these tasks occasionally close (schema_lite)
and often reach high partial (deep_merge 0.65), selection captures the rare full solve and the best
partial — and it sidesteps the warm-start's proven edit-fragility entirely (no LoRA training).

**Elicitation SFT (STaR-style):** harvested 15 execution-verified best-of-n successes on the two
latent TRAIN tasks (bellman_ford 8/16, topo_lex 7/16), mixed them with the 46 existing diverse passes
(to preserve breadth), and SFT'd the warm-start at a gentle 2e-5 / 1 epoch.

- **On the TRAINED task, distillation worked:** topo_lex single-shot 0.25 → **0.75** (k=8). Fitting
  best-of-n successes does raise single-shot reliability on that task. (bellman_ford 0.88→0.50 is
  within noise — the harvest itself showed 0.50, so ~0.5 is its true rate and 0.88 was a high sample.)
- **On the holdout it REGRESSED, worse than DPO:** 0.606 → **0.485** (−0.121), damaging four
  previously-solid tasks (min_heap/pretty_bytes 1.00→0.33, base_convert/rle 1.00→0.67) while rescuing
  two — the identical damage signature as DPO, despite the breadth-preserving mix and gentle LR.

**The converging law (now four independent replications):** every LoRA edit of the 0.606 warm-start
perturbs deployment DOWNWARD — RFT-v1 crash (0.121), RFT-v2 null (0.576), DPO −0.091 (0.515),
elicit-SFT −0.121 (0.485). The warm-start is a robust local optimum; fitting new task-specific data
helps THAT task (topo_lex +0.50) but disrupts the general policy. The "improve the warm-start with
more LoRA training on this curriculum" line is falsified across SFT, preference (DPO), and STaR
distillation. **The original merged warm-start (0.606) is the deployment artifact.**

## CAPSTONE: execution-selected best-of-n beats the ceiling WITHOUT training (2026-07-21)

The whole arc resolves here. Four LoRA edits of the 0.606 warm-start each REGRESSED deployment
(−0.09 to −0.12). The ceiling probe showed the failure is termination — the model reaches partial
(occasionally full) solutions but loops instead of closing. Put together, these say the lift must come
from INFERENCE-time selection, not policy editing. Measured directly from the baseline's own 3
rollouts per holdout task (execution-selected = keep the run whose tests score highest; "solved" if
ANY of the 3 fully passes):

| | holdout mean pass |
|---|---|
| single-shot (deployment baseline) | 0.606 |
| **execution-selected best-of-3** | **0.727 (+0.121)** |

**+0.121 with zero training** — the exact mirror of what every policy edit *lost*. It sidesteps the
warm-start's edit-fragility entirely (no LoRA), and it is directly implied by the ceiling result: the
hard tasks occasionally close (schema_lite 1/12) or reach high partial (deep_merge 0.65), so keeping
the best of N captures those wins. best-of-3 is the FLOOR — best-of-8/12 would capture the rarer
closes (schema_lite at 1/12 needs ~8+ samples). The honest program-level answer: **since editing this
4B's policy regresses it, the deployable lift on real agentic coding comes from inference-time
execution selection over multiple pi rollouts, not from further training the policy.** (Requires a
verifier at deploy — here the tests; on a real repo, its own test suite or a generated check.)

## Verification

Every headline number in the two sections above was independently re-derived from the raw per-episode
data by an 8-way adversarial audit (each auditor recomputed one statistic from the `episodes` arrays,
forbidden from trusting any pre-aggregated field — the discipline that would have caught the `meanR`
bug). **All 8 matched, zero discrepancies:** baseline 0.606 / best-of-3 0.727 (8/11 tasks), RFT
0.121 & 0.576, DPO 0.515 (rates-only, episode-level unverifiable), elicit 0.485 + all four damage
drops, topo_lex 0.25→0.75 & bellman_ford 0.875→0.50, the partial-not-absent correction (all seven
tasks nonzero mean 0.13–0.41, none absent), and best-of-12 schema_lite 1/12 + 96% timeout.

## Next Experiments

- **Confirm the capstone at best-of-8/12** with the explicit selection protocol (run N pi rollouts per
  holdout task, keep the highest test-scoring) — best-of-3 from re-used baseline data already shows
  +0.121; a dedicated run should push higher by capturing the rare closes (schema_lite).
- **The incremental-LoRA-edit line is closed (4 replications).** If more capability is wanted, train a
  SINGLE better warm-start FROM BASE on a much larger, more diverse execution-verified pi harvest
  (include the now-known-solvable topo_lex/bellman_ford), rather than editing the current optimum —
  the "scale the curriculum 10-20x, train once" bet, not another adapter on top.
- Diverge from policy-editing this warm-start. Four narrow edits (RFT×2, DPO, elicit-SFT) have now
  failed to beat 0.606; stop iterating hyperparameters. The holdout headroom (`schema_lite`,
  `json_pointer`, and the train-side `deep_merge`/`semver`/etc.) is NOT absent capability — the model
  edits and passes SOME tests (partial reward up to 0.55) but never CLOSES and times out. So the
  intervention to test is one that helps it FINISH a partial solution and TERMINATE: more inference
  compute (best-of-n with execution selection, which the pending best-of-12 probe informs), a
  higher think/turn budget, or targeting the specific failing tests — not "installing a missing skill".
- If returning to a negative gradient: target loop-*before*-solve (the real failure), pair against it
  with a gentler dose (higher beta, ≤1 epoch, lower LR) so the working policy is not perturbed — the
  DPO regression was overshoot on a mis-targeted signal, not proof the method cannot help.
- Harvest pi trajectories on solvable real-repo tasks → multi-turn tool-calling SFT warm-start.
- RLVR (GRPO) from the merged warm-start; reward = FAIL_TO_PASS AND PASS_TO_PASS, no-network sandbox
  (reward-hacking guard); kill rule: RLVR must beat matched-compute sample-more on held-out pass-rate.
- OpenEnv packaging of `CodingEnv` (create_app) for portability + pi/harvest reuse; mine duet-eval
  scenarios into OpenEnv envs under a strict train/test firewall (never eval on trained scenarios).

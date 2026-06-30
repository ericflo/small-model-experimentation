# Qwen3.5-4B Thinking-Budget Scaling Experiment Log

## Scaffold

Created as a new experiment under the new `test_time_reasoning_budget` program.

## Environment (RTX 4090 / WSL, migrated box)

The box arrived without a working ML env (the old `vllm/env` venv was broken: built for
py3.10 but its `python` symlinked to system 3.12). System python lacked `ensurepip`, so
`python3 -m venv` failed. Resolution: installed `uv` (no sudo) and created a standard
project `.venv`. Stack: torch 2.12.1+cu130 (CUDA OK on the 4090), transformers 5.12.1
(natively supports `qwen3_5`), datasets 5.0.0, accelerate, bitsandbytes, matplotlib, pandas,
flash-linear-attention 0.5.1.

## Model

- **Qwen/Qwen3.5-4B** is the repo standard and is used here. (An earlier detour to the older
  Qwen3-4B was a mistake and was reverted — only the repo's current model is valid.)
- Qwen3.5-4B is `model_type: qwen3_5`, a **hybrid linear-attention + multimodal** model
  (`Qwen3_5ForCausalLM` for text). Loads via `AutoModelForCausalLM(trust_remote_code=True,
  dtype=bfloat16)`, ~8.4 GB VRAM. The cached copy on the box was tokenizer-only (~13 MB);
  full weights (~9.3 GB) were downloaded.
- **Think-token ids differ from Qwen3-4B**: verified empirically `<think>`=248068,
  `</think>`=248069 (vocab 248320). Using Qwen3-4B's ids (151667/151668) would silently break
  budget forcing.

## Harness

- `src/runtime.py`: prompts (system+user, `enable_thinking` toggle), batched generation,
  s1-style thinking-budget forcing (cap thinking at B tokens; if `</think>` not emitted, inject
  it and regenerate the answer), shuffled-thinking control, OOM-resilient batching
  (auto-subdivide on CUDA OOM).
- `src/tasks.py`: MBPP sanitized loader + sandboxed execution verifier (subprocess, rlimits,
  timeout) + code extraction. Verifier runs in a **separate torch-free process**
  (`scripts/verify_runs.py`) because forking candidate sandboxes from the CUDA process
  triggered MemoryError (a fork of an 8 GB+CUDA process under a 2 GB AS rlimit).
- `src/metrics.py`: unbiased pass@k + deployable (greedy, visible-test selector) vs oracle.
- `scripts/run.py`: GPU generation phase → writes generations.jsonl → spawns torch-free
  verification → summary.json. `analysis/analyze.py`: tables + scaling-curve figures.

## Smoke / sanity

- Smoke (5 tasks, no_think+think_512, k=2) validated the full path. Early shape: thinking
  lifted greedy 0.60→1.00 and pass@2 0.80→1.00 while the visible-selector stayed 0.60 (the C2
  oracle-vs-deployable gap this experiment is built to measure) — far too few tasks to conclude.
- Long-budget sanity (1024/2048/unbudgeted) confirmed no OOM and natural termination; easy
  tasks finish thinking in ~450–570 tokens, so forcing only bites at B≤512.

## Throughput finding (a documented lesson)

Qwen3.5-4B's linear-attention layers run in a **slow torch fallback**: the fast path needs
BOTH `flash-linear-attention` AND `causal-conv1d`. fla installed cleanly; `causal-conv1d`
needs CUDA-toolkit compilation (nvcc 13.2 is present at `/usr/local/cuda-13.2`, so it is
buildable — deferred as a throughput optimization for future program runs). Without it,
aggregate decode is ~368 tok/s at batch 32 (~11 tok/s/seq), GPU util ~34% (kernel-bound, not
batch-bound). First OOM came from batch 64 + no expandable_segments; fixed with conservative→
moderate batches, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and OOM auto-subdivision.

## Main run

Launched: 100 MBPP test tasks, k=8, budgets {no_think, 256, 512, 1024, 2048, unbudgeted},
deployable greedy + oracle pass@k, output `runs/main/`. Generation took ~8.6h (the linear-attention
fallback + batch-gating by the slowest sequence; `unbudgeted` alone was 3.6h). Controls
(shuffled-thinking at 512/2048) ran after, via `scripts/finish_experiment.sh`.

## Results (see reports/report.md)

Deployable greedy pass@1: no_think 0.76 → think_1024 **0.91** (+15pp) → 2048 0.86 → unbudgeted 0.84.
Oracle pass@8: 0.91 → 0.96. Deployable moved more than oracle; selection gap narrowed; paired
17 fail→pass vs 2 pass→fail at think_1024 (McNemar p≈0.001). Non-monotonic (overthinking).
Shuffle control: scrambled thinking reproduces much of the gain (shuffle_512 0.80, shuffle_2048
0.86 ≈ real 0.86) — a large share is compute/scaffold, not coherent reasoning.

All headline numbers were **independently recomputed from raw `verified.jsonl` and audited** by a
separate verification workflow (transcription clean; the audit's overclaim flags — exact-1024
optimum, argmax-over-budgets difficulty deltas, half/half content split — were folded into the
final report as hedges). A verification jitter (±1–2pp from timeout-sensitivity under GPU load)
motivated the verifier's retry-on-timeout; final numbers were verified with no competing GPU job.

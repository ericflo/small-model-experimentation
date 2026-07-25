#!/usr/bin/env bash
# Serve a policy for pi-coding-agent, with every flag that a failing run has already taught us.
#
#   scripts/serve.sh Qwen/Qwen3.5-4B                     # raw base (the missing control)
#   scripts/serve.sh <path-to-merged-warmstart>          # a trained arm
#
# Each flag is load-bearing (provenance: qwen35_4b_agentic_rlvr_feasibility/reports/report.md):
#   --served-model-name qwen35-4b-pi8k   pi's model entry name. Serving as "Qwen3.5-4B" instead gives
#                                        a 404 model-name mismatch that silently yields EMPTY
#                                        trajectories rather than an error -- 38 episodes were once
#                                        lost this way.
#   --enable-auto-tool-choice
#   --tool-call-parser qwen3_xml         without these vLLM returns 400 ("auto" tool choice requires
#                                        a parser) and pi completes ZERO tool calls. qwen3_xml matches
#                                        Qwen3.5's <tool_call><function=><parameter=> format.
#   --max-model-len 40960                pi sends its entry's maxTokens (8192) as
#                                        max_completion_tokens on EVERY call, and vLLM rejects
#                                        prompt + max_completion_tokens > max_model_len. At 16384 the
#                                        agent dies the moment the conversation passes ~8k.
#   --enforce-eager                      Qwen3.5's hybrid GDN/attention arch HANGS on torch.compile /
#                                        CUDAGraph capture (C61).
#   --gpu-memory-utilization 0.45        VRAM is mirrored into host RAM ~1:1 under WSL2, so VRAM is a
#                                        HOST memory cost: 8.5 GB weights + ~2.3 GB KV keeps vmmem in
#                                        a safe band (docs/wsl_stability.md).
#
# Run inside a cgroup scope: MemoryMax bounds RESIDENT memory only, so it is safe for CUDA (which
# reserves huge virtual address space) and still prevents a guest-side balloon from OOM-killing
# /init.scope and taking down all of WSL.
set -euo pipefail

MODEL="${1:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-8420}"
# util 0.45 (the prior cell's figure) does NOT fit a 40960 context on this box: gpu_memory_utilization
# is a fraction of TOTAL VRAM, ~2 GiB of which is already held by WSL/WDDM before vLLM starts. Budget:
# 8.8 GiB weights + ~2 GiB overhead + KV. At 0.45 only 0.39 GiB was left for KV against the 1.31 GiB a
# single max-length request needs, so the engine refused to start ("estimated maximum model length is
# 11088"). 0.62 leaves ~4 GiB of KV.
UTIL="${UTIL:-0.62}"
MAXLEN="${MAXLEN:-40960}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
LOGDIR="$ROOT/large_artifacts/qwen35_4b_realrepo_agentic_instrument/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/serve_$(basename "$MODEL" | tr '/' '_').log"

# `.venv-vllm/bin` MUST be on PATH: vLLM shells out to `ninja` while sizing the KV cache, and without
# it engine startup dies with FileNotFoundError: 'ninja' -- reported as the unhelpful
# "Engine core initialization failed", with the real cause 40 lines up the log.
export PATH="$ROOT/.venv-vllm/bin:$PATH"

echo "serving $MODEL on :$PORT (util $UTIL, maxlen $MAXLEN) -> $LOG"
GUARD_MEM="${GUARD_MEM:-12G}" GUARD_HIGH="${GUARD_HIGH:-11G}" GUARD_SWAP="${GUARD_SWAP:-1G}" \
nohup "$HERE/guard.sh" "$ROOT/.venv-vllm/bin/vllm" serve "$MODEL" \
  --served-model-name qwen35-4b-pi8k \
  --port "$PORT" \
  --enforce-eager \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" \
  --max-num-seqs 4 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  > "$LOG" 2>&1 &

echo "waiting for readiness (a served endpoint that never came up has silently emptied runs before)"
for i in $(seq 1 180); do
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "READY after ${i}0s"
    curl -s "http://localhost:$PORT/v1/models" | head -c 400; echo
    exit 0
  fi
  if grep -qE "Error|Traceback|CUDA out of memory|ValueError" "$LOG" 2>/dev/null; then
    echo "FAILED to start -- last lines:"; tail -20 "$LOG"; exit 1
  fi
  sleep 10
done
echo "TIMEOUT waiting for :$PORT"; tail -20 "$LOG"; exit 1

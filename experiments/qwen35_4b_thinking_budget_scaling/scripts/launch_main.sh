#!/usr/bin/env bash
# Launch the main thinking-budget sweep detached, with OOM-safe settings.
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_thinking_budget_scaling
pkill -9 -f "run.py --tasks 100" 2>/dev/null || true
pkill -9 -f "verify_runs.py" 2>/dev/null || true
sleep 3
rm -rf runs/main runs/longcheck
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup ../../.venv/bin/python scripts/run.py --tasks 100 --k 8 \
  --budgets no_think,256,512,1024,2048,unbudgeted --out runs/main \
  > runs/main_console.log 2>&1 &
echo "launched main run, PID $!"

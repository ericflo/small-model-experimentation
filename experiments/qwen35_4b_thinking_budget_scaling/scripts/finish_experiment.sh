#!/usr/bin/env bash
# Autonomous completion: wait for main sweep -> analyze -> run shuffled-thinking
# controls -> analyze. Designed to be launched harness-tracked so the agent is
# notified when the ENTIRE experiment (main + controls + analysis) is done.
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_thinking_budget_scaling
PY=../../.venv/bin/python
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. wait for the main 6-condition sweep
DEADLINE=$(( $(date +%s) + 8*3600 ))
while [ ! -f runs/main/summary.json ]; do
  if ! pgrep -f "run.py --tasks 100 --k 8 --budgets" >/dev/null && [ ! -f runs/main/summary.json ]; then
    echo "MAIN RUN DIED"; tail -25 runs/main_console.log; exit 3
  fi
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "TIMEOUT (main)"; exit 4; }
  sleep 30
done
echo "=== MAIN DONE — analyzing ==="
$PY analysis/analyze.py --tag main 2>&1 | grep -vE "warn|Warning"

# 2. controls: shuffled-thinking at 512 and 2048 (content-vs-compute test)
echo "=== LAUNCHING CONTROLS (shuffled thinking) ==="
$PY scripts/run.py --tasks 100 --k 8 --only-controls --out runs/controls > runs/controls_console.log 2>&1
echo "=== CONTROLS DONE — analyzing ==="
$PY analysis/analyze.py --tag controls 2>&1 | grep -vE "warn|Warning"

echo "=== FULL_EXPERIMENT_DONE ==="

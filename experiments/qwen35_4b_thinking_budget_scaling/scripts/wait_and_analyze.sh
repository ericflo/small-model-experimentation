#!/usr/bin/env bash
# Block until the main sweep finishes (summary.json appears), then run analysis.
# Run this harness-tracked so completion notifies the agent with results in hand.
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_thinking_budget_scaling
SUMMARY=runs/main/summary.json
DEADLINE=$(( $(date +%s) + 6*3600 ))   # 6h safety cap
while [ ! -f "$SUMMARY" ]; do
  if ! pgrep -f "run.py --tasks 100" >/dev/null && [ ! -f "$SUMMARY" ]; then
    echo "RUN DIED before producing summary.json"; tail -25 runs/main_console.log; exit 3
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo "TIMEOUT waiting for summary"; exit 4; fi
  sleep 30
done
echo "=== summary.json present; running analysis ==="
HF_HUB_OFFLINE=1 ../../.venv/bin/python analysis/analyze.py --tag main 2>&1 | grep -vE "warn|Warning"
echo "=== ANALYSIS_DONE ==="

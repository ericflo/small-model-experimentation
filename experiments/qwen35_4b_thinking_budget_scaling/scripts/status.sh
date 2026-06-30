#!/usr/bin/env bash
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_thinking_budget_scaling
if pgrep -f "run.py --tasks 100" >/dev/null; then
  echo "STATE: RUNNING (pid $(pgrep -f 'run.py --tasks 100' | head -1))"
else
  echo "STATE: NOT RUNNING"
fi
echo "GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
echo "--- progress ---"
grep -E "\[gen\]|conditions=|model loaded|verifying|generation done|OOM|out of memory|Traceback|Error" runs/main_console.log 2>/dev/null | tail -10
echo "--- gens written: $(wc -l < runs/main/generations.jsonl 2>/dev/null || echo 0) ---"
if [ -f runs/main/summary.json ]; then echo "SUMMARY EXISTS"; fi

#!/usr/bin/env bash
# Wait for the 2048 recovery to finish, then verify + curve. Harness-tracked so completion
# notifies the agent with the full result in hand.
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_overthinking_content_ladder
PY=../../.venv/bin/python
DEADLINE=$(( $(date +%s) + 3*3600 ))
while pgrep -f add_2048.py >/dev/null; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "TIMEOUT waiting for recovery"; exit 4; }
  sleep 30
done
echo "=== recovery finished; records=$(wc -l < data/records.jsonl) ==="
if [ "$(wc -l < data/records.jsonl)" -lt 10000 ]; then
  echo "RECOVERY INCOMPLETE"; tail -20 runs/add_2048.log; exit 3
fi
echo "=== VERIFY ==="
HF_HUB_OFFLINE=1 $PY scripts/verify.py 2>&1 | grep -vE "warn"
echo "=== CURVE ==="
$PY analysis/curve.py 2>&1 | grep -vE "warn"
echo "=== FINISH_DONE ==="

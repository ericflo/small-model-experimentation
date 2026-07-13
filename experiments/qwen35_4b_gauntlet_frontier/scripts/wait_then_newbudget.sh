#!/usr/bin/env bash
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation
LOG=experiments/qwen35_4b_gauntlet_frontier/runs
# 1) wait for the reinforcement job to finish
until grep -q 'reinforce complete' "$LOG/cr_reinforce_2048.log" 2>/dev/null; do sleep 20; done
echo "reinforce done; waiting for GPU to drain..."
# 2) wait for GPU compute apps to clear (no stray vLLM holding memory)
for i in $(seq 1 60); do
  APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  [ "${APPS:-0}" -eq 0 ] && break
  sleep 10
done
sleep 5
echo "GPU drained; launching new-budget eval"
# 3) run the new-canonical (8192) eval; first quick event smoke-tests 65536 ctx
bash experiments/qwen35_4b_gauntlet_frontier/scripts/newbudget_eval.sh

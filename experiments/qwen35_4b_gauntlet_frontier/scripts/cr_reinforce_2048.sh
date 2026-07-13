#!/usr/bin/env bash
# Strengthen the 2048 point of the compute-response study to n=6: this is the
# budget at which the medium delta first clears +0.32, so it decides whether the
# medium wall is budget-tunable past the goal bar. Fresh seeds, paired.
set -euo pipefail
cd /home/ericflo/Development/small-model-experimentation
MERGED=large_artifacts/qwen35_4b_gauntlet_frontier/merged/apex_cr
BENCH=experiments/qwen35_4b_gauntlet_frontier/scripts/bench.py
for seed in 54007 54008 54009 54010; do
  echo "=========== medium budget=2048 seed=$seed ==========="
  .venv/bin/python "$BENCH" --tier medium --seed "$seed" \
    --arms base merged --merged "$MERGED" --think-budget 2048 \
    --note "compute-response medium tb=2048 reinforce"
done
echo "=========== reinforce complete ==========="

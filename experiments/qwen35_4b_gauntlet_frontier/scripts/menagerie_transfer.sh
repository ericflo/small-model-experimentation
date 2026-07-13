#!/usr/bin/env bash
# Menagerie retain-delta test for the induction/exploration install: does the
# gym exploration gain TRANSFER to the held-out benchmark at the maxed 8192
# budget? Paired base-vs-induction1 on quick+medium, fresh seeds, tier default
# 8192. Fast read of the combined install (includes the net-negative glyphgate
# traces, so this is a conservative lower bound on the exploration transfer).
set -euo pipefail
cd /home/ericflo/Development/small-model-experimentation
MERGED=large_artifacts/qwen35_4b_gauntlet_frontier/merged/induction1
BENCH=experiments/qwen35_4b_gauntlet_frontier/scripts/bench.py
for cfg in "quick 56001" "quick 56002" "medium 56003" "medium 56004"; do
  set -- $cfg
  echo "===== $1 @8192 seed=$2 ====="
  .venv/bin/python "$BENCH" --tier "$1" --seed "$2" --arms base merged --merged "$MERGED" \
    --note "induction1 transfer 8192 tier=$1"
done
echo "===== menagerie transfer complete ====="

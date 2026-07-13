#!/usr/bin/env bash
# Tighten the exploration-only MEDIUM retain-delta to n=6: the n=2 read (+0.277,
# one seed +0.332) straddles the +0.32 line and needs more seeds before any
# claim (medium per-event sd ~0.05).
set -euo pipefail
ROOT=/home/ericflo/Development/small-model-experimentation
cd $ROOT
MERGED=large_artifacts/qwen35_4b_gauntlet_frontier/merged/exploration1
for seed in 57005 57006 57007 57008; do
  echo "--- medium seed=$seed ---"
  .venv/bin/python experiments/qwen35_4b_gauntlet_frontier/scripts/bench.py \
    --tier medium --seed "$seed" --arms base merged --merged "$MERGED" \
    --note "exploration-only 8192 tier=medium tighten"
done
echo "===== expl medium tighten complete ====="

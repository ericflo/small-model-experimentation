#!/usr/bin/env bash
# Diagnostic: is there a RETAINED induction gap at the maxed 8192 budget? Runs
# base vs the apex broad-install merged on glyphgate (active induction) atoms,
# greedy@1, per level, tb=8192, episodes skipped for speed. If apex >> base at
# 8192, induction is where retained install value lives (unlike efficiency,
# which C55 showed compresses); a focused induction install is then worth it.
set -euo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_gauntlet_frontier
VP=../../.venv-vllm/bin/python
MERGED=/home/ericflo/Development/small-model-experimentation/large_artifacts/qwen35_4b_gauntlet_frontier/merged/apex_cr
echo "===== BASE @8192 glyphgate ====="
$VP scripts/eval_gym.py --config configs/frontier.yaml --tag diag_ind_base8192 \
  --families glyphgate --think-budget 8192 --episode-levels
echo "===== APEX @8192 glyphgate ====="
$VP scripts/eval_gym.py --config configs/frontier.yaml --tag diag_ind_apex8192 \
  --families glyphgate --think-budget 8192 --episode-levels --merged "$MERGED"
echo "===== induction diag complete ====="

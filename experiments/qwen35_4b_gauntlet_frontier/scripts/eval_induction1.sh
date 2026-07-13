#!/usr/bin/env bash
# Merge the induction1 adapter, then gym-eval at 8192 on glyphgate (induction)
# and burrowmaze (exploration). The gate: does composed-rule induction
# (glyphgate L4-L6, base=0.0) lift above zero? Also runs base on burrowmaze so
# the exploration axis has its own base reference.
set -euo pipefail
ROOT=/home/ericflo/Development/small-model-experimentation
cd $ROOT/experiments/qwen35_4b_gauntlet_frontier
VP=$ROOT/.venv/bin/python
VVP=$ROOT/.venv-vllm/bin/python
MERGED=$ROOT/large_artifacts/qwen35_4b_gauntlet_frontier/merged/induction1
echo "===== MERGE induction1 ====="
$VP scripts/merge_adapter.py --adapter $ROOT/large_artifacts/qwen35_4b_gauntlet_frontier/adapters/induction1 --out $MERGED
echo "===== EVAL induction1 @8192 (glyphgate+burrowmaze) ====="
$VVP scripts/eval_gym.py --config configs/frontier.yaml --tag ind1_8192 \
  --families glyphgate burrowmaze --think-budget 8192 --episode-levels --merged "$MERGED"
echo "===== EVAL base @8192 (burrowmaze reference) ====="
$VVP scripts/eval_gym.py --config configs/frontier.yaml --tag diag_expl_base8192 \
  --families burrowmaze --think-budget 8192 --episode-levels
echo "===== eval_induction1 complete ====="

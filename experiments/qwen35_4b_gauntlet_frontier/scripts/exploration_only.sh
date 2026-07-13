#!/usr/bin/env bash
# Clean exploration-only install: isolate the installable weak axis (burrowmaze,
# no net-negative glyphgate). Definitive ceiling for exploration installability
# and its menagerie retain-delta at 8192 (the combined install's +0.190 medium
# is a lower bound). train -> merge -> gym eval -> menagerie retain-delta.
set -euo pipefail
ROOT=/home/ericflo/Development/small-model-experimentation
cd $ROOT/experiments/qwen35_4b_gauntlet_frontier
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VP=$ROOT/.venv/bin/python
VVP=$ROOT/.venv-vllm/bin/python
ADAPT=$ROOT/large_artifacts/qwen35_4b_gauntlet_frontier/adapters/exploration1
MERGED=$ROOT/large_artifacts/qwen35_4b_gauntlet_frontier/merged/exploration1

echo "===== TRAIN exploration1 ====="
$VP scripts/train_think.py --train data/sft_exploration.jsonl --out $ADAPT \
  --epochs 2.0 --lr 2e-4 --rank 32 --alpha 64 \
  --batch-size 1 --grad-accum 16 --max-length 3072 --w-think 0.2 --seed 42

echo "===== MERGE exploration1 ====="
$VP scripts/merge_adapter.py --adapter $ADAPT --out $MERGED

echo "===== GYM EVAL burrowmaze @8192 ====="
$VVP scripts/eval_gym.py --config configs/frontier.yaml --tag expl1_8192 \
  --families burrowmaze --think-budget 8192 --episode-levels --merged "$MERGED"

echo "===== MENAGERIE retain-delta @8192 ====="
for cfg in "quick 57001" "quick 57002" "medium 57003" "medium 57004"; do
  set -- $cfg
  echo "--- $1 seed=$2 ---"
  $VP scripts/bench.py --tier "$1" --seed "$2" --arms base merged --merged "$MERGED" \
    --note "exploration-only 8192 tier=$1"
done
echo "===== exploration_only complete ====="

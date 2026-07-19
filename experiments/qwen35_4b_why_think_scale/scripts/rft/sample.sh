#!/bin/bash
# Usage: sample.sh OFFSET COUNT K TAG
set -uo pipefail
ROOT=/home/ericflo/Development/small-model-experimentation
SP=/tmp/claude-1000/-home-ericflo-Development-small-model-experimentation/877cfc7b-ff96-4334-b8b5-d31dd4c686fa/scratchpad
RD=$SP/rft
OFF=$1; CNT=$2; K=$3; TAG=$4
RUNNER=$ROOT/experiments/qwen35_4b_coding_fitness_harness/src/vllm_runner.py
BASE=$ROOT/large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ENABLE_V1_MULTIPROCESSING=0 \
       HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
mkdir -p $RD/work
python3 $RD/build_problems.py --offset $OFF --count $CNT --out $RD/work/${TAG}_problems.jsonl --meta $RD/work/${TAG}_meta.json
echo "=== sampling K=$K temp 0.8 ==="
$ROOT/.venv-vllm/bin/python -B $RUNNER \
  --input $RD/work/${TAG}_problems.jsonl --output $RD/work/${TAG}_gen.jsonl --metadata $RD/work/${TAG}_gen.meta.json \
  --thinking budget --thinking-budget 8192 --n $K --temperature 0.8 --top-p 0.95 \
  --max-tokens 9216 --answer-max-tokens 1024 --seed 0 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 24 --max-num-batched-tokens 8192 \
  --model-override $BASE 2>&1 | tail -6
echo "=== filtering ==="
python3 $RD/filter_build.py --gen $RD/work/${TAG}_gen.jsonl --meta $RD/work/${TAG}_meta.json \
  --problems $RD/work/${TAG}_problems.jsonl --out $RD/work/rft_${TAG}.jsonl --max-per-problem 1

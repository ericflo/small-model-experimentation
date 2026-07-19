#!/bin/bash
set -uo pipefail
ROOT=/home/ericflo/Development/small-model-experimentation
SP=/tmp/claude-1000/-home-ericflo-Development-small-model-experimentation/877cfc7b-ff96-4334-b8b5-d31dd4c686fa/scratchpad
BASE=$ROOT/large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized
TRAINER=$ROOT/experiments/qwen35_4b_why_think_scale/scripts/train_think.py
MERGER=$ROOT/experiments/qwen35_4b_why_think_scale/scripts/merge_adapter.py
EVAL=$ROOT/experiments/qwen35_4b_coding_fitness_harness/scripts/eval_pass1.py
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
AD=$ROOT/large_artifacts/_abl/cleanfree_adapter
MG=$ROOT/large_artifacts/_abl/cleanfree_merged
rm -rf $AD $MG
echo "=== [cleanfree] TRAIN (clean code, w_think=0 w_close=0) ==="
$ROOT/.venv/bin/python -B $TRAINER --train $SP/abl_nowhy_5000.jsonl --out $AD \
  --epochs 1.0 --lr 1e-05 --rank 32 --alpha 64 --batch-size 1 --grad-accum 8 \
  --max-length 4096 --w-think 0.0 --w-close 0.0 --seed 95201 --model-path $BASE 2>&1 | tail -3
echo "=== [cleanfree] MERGE ==="
$ROOT/.venv/bin/python -B $MERGER --adapter $AD --out $MG --base-model $BASE 2>&1 | tail -1
$ROOT/.venv-vllm/bin/python $EVAL --dataset humaneval --n 164 --model-override $MG --out $SP/abl_cleanfree_he.json > /dev/null 2>&1
$ROOT/.venv-vllm/bin/python $EVAL --dataset mbpp --n 200 --model-override $MG --out $SP/abl_cleanfree_mbpp.json > /dev/null 2>&1
python3 -c "
import json
he=json.load(open('$SP/abl_cleanfree_he.json')); mb=json.load(open('$SP/abl_cleanfree_mbpp.json'))
print(f'[cleanfree] HE {he[\"passed\"]}/164 ({round((he[\"pass_at_1\"]-0.8963)*164):+d})  MBPP {mb[\"passed\"]}/200 ({round((mb[\"pass_at_1\"]-0.7550)*200):+d})')
"
rm -rf $AD $MG
echo "=== CLEANFREE DONE ==="

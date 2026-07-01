#!/usr/bin/env bash
# M3 end-to-end: collect self-solutions -> QLoRA-SFT -> eval frozen vs trained on held-out fresh tasks.
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_neurosymbolic_repl_substrate
PY=../../.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
flt() { grep -avE "it/s\]$|Loading checkpoint|Loading weights|Fetching|FutureWarning|_check_is_size|triton|make_block|UserWarning|warnings.warn"; }
echo "=== COLLECT $(date +%T) ==="
$PY scripts/collect_solutions.py --train-depths 1 2 3 --per-depth 150 --k 6 \
    --eval-depths 1 2 3 4 5 --eval-per-depth 15 --eval-seed 303 2>&1 | flt | grep -aE "train pool|collected|depth|solved|wrote"
echo "train pairs: $(wc -l < data/train.jsonl)"
echo "=== TRAIN $(date +%T) ==="
$PY scripts/train_lora.py --epochs 2 2>&1 | flt | grep -aE "training on|trainable|'loss'|train_loss|saved|Error|Traceback"
echo "=== EVAL FROZEN $(date +%T) ==="
$PY scripts/eval_lora.py --eval-tasks data/eval_tasks.jsonl --k 5 --out runs/eval_frozen.json 2>&1 | flt | grep -aE "frozen|nothink|think_|wrote"
echo "=== EVAL TRAINED $(date +%T) ==="
$PY scripts/eval_lora.py --eval-tasks data/eval_tasks.jsonl --k 5 --adapter runs/lora_adapter --out runs/eval_trained.json 2>&1 | flt | grep -aE "trained|nothink|think_|wrote"
echo "=== M3_DONE $(date +%T) ==="

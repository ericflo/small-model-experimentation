#!/usr/bin/env bash
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_neurosymbolic_repl_substrate
PY=../../.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
flt() { grep -avE "it/s\]$|Loading|Fetching|FutureWarning|_check_is_size|triton|make_block|UserWarning|warnings.warn"; }
echo "=== CONFIRM EVAL FROZEN $(date +%T) ==="
$PY scripts/eval_lora.py --eval-tasks data/eval_tasks_big.jsonl --k 5 --out runs/eval_frozen_big.json 2>&1 | flt | grep -aE "frozen|nothink|think_|wrote"
echo "=== CONFIRM EVAL TRAINED $(date +%T) ==="
$PY scripts/eval_lora.py --eval-tasks data/eval_tasks_big.jsonl --k 5 --adapter runs/lora_adapter --out runs/eval_trained_big.json 2>&1 | flt | grep -aE "trained|nothink|think_|wrote"
echo "=== CONFIRM_DONE $(date +%T) ==="

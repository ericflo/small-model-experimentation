#!/usr/bin/env bash
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_simulation_keystone_repair
PY=../../.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
flt(){ grep -avE "it/s\]$|Loading|Fetching|FutureWarning|_check_is_size|triton|make_block|UserWarning|warnings.warn"; }
echo "=== TRAIN SIM $(date +%T) ==="
$PY scripts/train_lora.py --train data/train_sim.jsonl --out runs/adapter_sim --epochs 2 2>&1 | flt | grep -aE "training on|train_loss|saved"
echo "=== TRAIN PROD $(date +%T) ==="
$PY scripts/train_lora.py --train data/train_prod.jsonl --out runs/adapter_prod --epochs 2 2>&1 | flt | grep -aE "training on|train_loss|saved"
echo "=== SIMBENCH SIM $(date +%T) ==="
$PY scripts/run_simbench.py --tasks-file data/simbench_tasks.jsonl --adapter runs/adapter_sim --out runs/simbench_sim.json 2>&1 | flt | grep -aE "done|d[0-9]k|wrote"
echo "=== SIMBENCH PROD $(date +%T) ==="
$PY scripts/run_simbench.py --tasks-file data/simbench_tasks.jsonl --adapter runs/adapter_prod --out runs/simbench_prod.json 2>&1 | flt | grep -aE "done|d[0-9]k|wrote"
echo "=== LADDER BASE $(date +%T) ==="
$PY scripts/run_ladder.py --out runs/ladder_base.json 2>&1 | flt | grep -aE "ladder tasks|done|OVERALL|d[0-9]k|wrote"
echo "=== LADDER SIM $(date +%T) ==="
$PY scripts/run_ladder.py --adapter runs/adapter_sim --out runs/ladder_sim.json 2>&1 | flt | grep -aE "done|OVERALL|d[0-9]k|wrote"
echo "=== LADDER PROD $(date +%T) ==="
$PY scripts/run_ladder.py --adapter runs/adapter_prod --out runs/ladder_prod.json 2>&1 | flt | grep -aE "done|OVERALL|d[0-9]k|wrote"
echo "=== PHASE12_DONE $(date +%T) ==="

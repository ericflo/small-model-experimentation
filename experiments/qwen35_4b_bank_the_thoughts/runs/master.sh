#!/bin/bash
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_bank_the_thoughts
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../.venv/bin/python
F='it/s\]$|it/s,|Loading checkpoint|Fetching|UserWarning|warnings.warn|attention mask|pad token|Setting|generation flags|torch._check|warmup_ratio'
echo "=== HARVEST $(date +%H:%M:%S) ==="
$PY scripts/harvest_thoughts.py --adapter /tmp/claude-1000/-home-ericflo-Development-small-model-experimentation/023b1a84-6a82-4a18-85df-570f97a29549/scratchpad/scaling_adapters/banked_1280_adapter --pool 500 --budget 2048 2>&1 | grep -avE "$F" || exit 1
$PY scripts/build_train.py 2>&1 | grep -avE "$F" || exit 1
for X in A T Tcorrupt; do
  echo "=== TRAIN $X $(date +%H:%M:%S) ==="
  $PY scripts/train_lora_think.py --train data/train_$X.jsonl --out runs/adapter_$X --epochs 3 2>&1 | grep -avE "$F" || exit 1
done
D="--eval-file data/eval_frozen_d3.jsonl --K 16 --n-per-depth 80 --depths 3"
echo "=== DEPLOY base_nt $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag deploy_base_nt $D 2>&1 | grep -avE "$F"
echo "=== DEPLOY A_nt $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag deploy_A_nt --adapter runs/adapter_A $D 2>&1 | grep -avE "$F"
echo "=== DEPLOY T_th $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag deploy_T_th --adapter runs/adapter_T --think $D 2>&1 | grep -avE "$F"
echo "=== DEPLOY Tcorrupt_th $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag deploy_Tcorrupt_th --adapter runs/adapter_Tcorrupt --think $D 2>&1 | grep -avE "$F"
echo "=== DEPLOY T_nt $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag deploy_T_nt --adapter runs/adapter_T $D 2>&1 | grep -avE "$F"
for X in base A T; do
  echo "=== STEP1 $X $(date +%H:%M:%S) ==="
  AD=""; [ "$X" != "base" ] && AD="--adapter runs/adapter_$X"
  $PY scripts/run_thinking.py --tag s1_$X $AD --n 60 --budgets 0 2048 --steps 1 2>&1 | grep -avE "$F"
done
echo "=== ALLDONE $(date +%H:%M:%S) ==="

#!/bin/bash
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_bank_the_thoughts
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../.venv/bin/python
F='it/s\]$|it/s,|Loading checkpoint|Fetching|UserWarning|warnings.warn|attention mask|pad token|Setting|generation flags|torch._check|warmup_ratio'
$PY scripts/build_train.py 2>&1 | grep -avE "$F" || exit 1
for X in A T Tcorrupt; do
  echo "=== TRAIN $X $(date +%H:%M:%S) ==="
  $PY scripts/train_lora_think.py --train data/train_$X.jsonl --out runs/adapter_$X --epochs 3 2>&1 | grep -avE "$F" || exit 1
done
# DEPLOYABILITY (does thought-banking deploy depth-3 better? coverage@16=multi-sampling, greedy@1=single-shot), n=80
D="--eval-file data/eval_frozen_d3.jsonl --K 16 --n-per-depth 80 --depths 3"
echo "=== DEPLOY base_nt $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag deploy_base_nt $D 2>&1 | grep -avE "$F"
echo "=== DEPLOY A_nt $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag deploy_A_nt --adapter runs/adapter_A $D 2>&1 | grep -avE "$F"
echo "=== DEPLOY T_th $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag deploy_T_th --adapter runs/adapter_T --think $D 2>&1 | grep -avE "$F"
echo "=== DEPLOY Tcorrupt_th $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag deploy_Tcorrupt_th --adapter runs/adapter_Tcorrupt --think $D 2>&1 | grep -avE "$F"
echo "=== DEPLOY T_nt $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag deploy_T_nt --adapter runs/adapter_T $D 2>&1 | grep -avE "$F"
# PRIMARY: step-1 planning ranking (rationalization-robust), n=60, no-think & think
for X in base A T Tcorrupt; do
  echo "=== STEP1 $X $(date +%H:%M:%S) ==="
  AD=""; [ "$X" != "base" ] && AD="--adapter runs/adapter_$X"
  $PY scripts/run_thinking.py --tag s1_$X $AD --n 60 --budgets 0 2048 --steps 1 2>&1 | grep -avE "$F"
done
echo "=== ALLDONE $(date +%H:%M:%S) ==="

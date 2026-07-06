#!/bin/bash
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_bank_the_thoughts
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../.venv/bin/python
F='it/s\]$|it/s,|Loading checkpoint|Fetching|UserWarning|warnings.warn|attention mask|pad token|Setting|generation flags|torch._check|warmup_ratio'
echo "=== HARVEST_P2 $(date +%H:%M:%S) ==="
$PY scripts/harvest_phase2.py --adapter /tmp/claude-1000/-home-ericflo-Development-small-model-experimentation/023b1a84-6a82-4a18-85df-570f97a29549/scratchpad/scaling_adapters/banked_1280_adapter --pool 500 --budget 2048 2>&1 | grep -avE "$F" || exit 1
$PY scripts/build_phase2.py 2>&1 | grep -avE "$F" || exit 1
for X in Aself Tself Tsynth Tselfcorrupt; do
  echo "=== TRAIN $X $(date +%H:%M:%S) ==="
  $PY scripts/train_lora_think.py --train data/train_$X.jsonl --out runs/adapter_$X --epochs 3 2>&1 | grep -avE "$F" || exit 1
done
D="--eval-file data/eval_frozen_d3.jsonl --K 16 --n-per-depth 80 --depths 3"
echo "=== DEPLOY Aself_nt $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag p2_Aself_nt --adapter runs/adapter_Aself $D 2>&1 | grep -avE "$F"
echo "=== DEPLOY Tself_th $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag p2_Tself_th --adapter runs/adapter_Tself --think $D 2>&1 | grep -avE "$F"
echo "=== DEPLOY Tsynth_th $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag p2_Tsynth_th --adapter runs/adapter_Tsynth --think $D 2>&1 | grep -avE "$F"
echo "=== DEPLOY Tselfcorrupt_th $(date +%H:%M:%S) ==="; $PY scripts/eval_ladder.py --tag p2_Tselfcorrupt_th --adapter runs/adapter_Tselfcorrupt --think $D 2>&1 | grep -avE "$F"
echo "=== ALLDONE $(date +%H:%M:%S) ==="

#!/usr/bin/env bash
# Expert-iteration self-training flywheel: each round, solve a FIXED train pool with the CURRENT model,
# accumulate verified (prompt->code) pairs, retrain a fresh LoRA from base, eval on the fixed held-out set.
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_neurosymbolic_repl_substrate
PY=../../.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
flt(){ grep -avE "it/s\]$|Loading|Fetching|FutureWarning|_check_is_size|triton|make_block|UserWarning|warnings.warn"; }
ROUNDS=${ROUNDS:-3}; DEPTHS=${DEPTHS:-1 2 3 4}; PERDEPTH=${PERDEPTH:-90}; K=${K:-6}; EPOCHS=${EPOCHS:-2}
DATA=${DATA:-data/train_ei.jsonl}; HELD=${HELD:-data/eval_tasks_big.jsonl}; TAG=${TAG:-ei}
POOL="--train-depths $DEPTHS --per-depth $PERDEPTH --k $K --train-seed 202 --skip-eval-gen"
rm -f "$DATA"
prev=""
for r in $(seq 1 "$ROUNDS"); do
  echo "===== ROUND $r COLLECT (generator=${prev:-frozen}) $(date +%T) ====="
  adap=""; [ -n "$prev" ] && adap="--adapter $prev"
  $PY scripts/collect_solutions.py $POOL $adap --out "$DATA" --append 2>&1 | flt | grep -aE "generator|solved|pairs|total"
  echo "===== ROUND $r TRAIN $(date +%T) ====="
  $PY scripts/train_lora.py --train "$DATA" --out "runs/${TAG}_adapter_$r" --epochs "$EPOCHS" 2>&1 | flt | grep -aE "training on|train_loss|saved"
  echo "===== ROUND $r EVAL $(date +%T) ====="
  $PY scripts/eval_lora.py --adapter "runs/${TAG}_adapter_$r" --eval-tasks "$HELD" --k 5 --out "runs/${TAG}_eval_$r.json" 2>&1 | flt | grep -aE "trained|think_|wrote"
  [ -n "$prev" ] && rm -rf "$prev"   # keep only the current adapter
  prev="runs/${TAG}_adapter_$r"
done
echo "===== ${TAG}_DONE $(date +%T) ====="

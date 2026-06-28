#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUDGETS="1,2,4,8,max"
VERIFIER_ADAPTER="/workspace/large_artifacts/qwen35_4b_real_sample_verify_commit/models/verifier_sft_lora"
STOP_ADAPTER="/workspace/large_artifacts/qwen35_4b_real_sample_verify_commit/models/stop_sft_lora"

for dataset in mbpp humaneval; do
  if [[ "$dataset" == "mbpp" ]]; then
    records="data/mbpp_eval_records.jsonl"
  else
    records="data/humaneval_eval_records.jsonl"
  fi
  python scripts/eval_commit.py --records "$records" --policy first_visible --name first_visible --out "reports/eval/${dataset}_first_visible.json" --budgets "$BUDGETS"
  python scripts/eval_commit.py --records "$records" --policy shortest_visible --name shortest_visible --out "reports/eval/${dataset}_shortest_visible.json" --budgets "$BUDGETS"
  python scripts/eval_commit.py --records "$records" --policy public_signature_majority --name public_signature_majority --out "reports/eval/${dataset}_public_signature_majority.json" --budgets "$BUDGETS"
  python scripts/eval_commit.py --records "$records" --policy oracle_coverage --name oracle_coverage --out "reports/eval/${dataset}_oracle_coverage.json" --budgets "$BUDGETS"
  python scripts/eval_commit.py --records "$records" --policy base_verifier --name base_verifier --out "reports/eval/${dataset}_base_verifier.json" --budgets "$BUDGETS"
  python scripts/eval_commit.py --records "$records" --policy sft_verifier --name sft_verifier --out "reports/eval/${dataset}_sft_verifier.json" --budgets "$BUDGETS" --adapter-dir "$VERIFIER_ADAPTER"
done

python scripts/eval_commit.py --records data/mbpp_train_records.jsonl --policy sft_verifier --name sft_verifier --out reports/eval/mbpp_train_sft_verifier.json --budgets "$BUDGETS" --adapter-dir "$VERIFIER_ADAPTER"
python scripts/build_stop_examples.py --scores reports/eval/mbpp_train_sft_verifier.json --budgets "$BUDGETS"
python scripts/tune_threshold.py --scores reports/eval/mbpp_train_sft_verifier.json --budgets "$BUDGETS" --out reports/threshold_tuning.json
THRESHOLD="$(python - <<'PY'
import json
print(json.load(open("reports/threshold_tuning.json"))["best"]["threshold"])
PY
)"

for dataset in mbpp humaneval; do
  if [[ "$dataset" == "mbpp" ]]; then
    records="data/mbpp_eval_records.jsonl"
  else
    records="data/humaneval_eval_records.jsonl"
  fi
  scores="reports/eval/${dataset}_sft_verifier.json"
  python scripts/eval_adaptive_budget.py --records "$records" --scores "$scores" --mode threshold --threshold "$THRESHOLD" --name threshold_sft_score --out "reports/eval/${dataset}_threshold_sft_score.json" --budgets "$BUDGETS"
  python scripts/eval_adaptive_budget.py --records "$records" --scores "$scores" --mode oracle_stop --name oracle_stop --out "reports/eval/${dataset}_oracle_stop.json" --budgets "$BUDGETS"
  python scripts/eval_adaptive_budget.py --records "$records" --scores "$scores" --mode sft_stop --name sft_stop_controller --out "reports/eval/${dataset}_sft_stop_controller.json" --budgets "$BUDGETS" --stop-adapter-dir "$STOP_ADAPTER"
done


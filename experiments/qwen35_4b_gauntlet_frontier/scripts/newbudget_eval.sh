#!/usr/bin/env bash
# New-canonical-budget evaluation (tiers now think@8192, max_model_len 65536).
# The first quick event doubles as the 65536-context vLLM smoke test: if the
# engine fails to init at that context on the 4090, it aborts here within ~1min
# of load, before wasting a full run. Paired base-vs-merged at the tier's now-
# default 8192 budget (NO --think-budget override -> uses the tier default).
# This measures the conjunction under the redefined benchmark.
set -euo pipefail
cd /home/ericflo/Development/small-model-experimentation
MERGED=large_artifacts/qwen35_4b_gauntlet_frontier/merged/apex_cr
BENCH=experiments/qwen35_4b_gauntlet_frontier/scripts/bench.py

run() { # tier seed
  echo "=========== $1 @8192 seed=$2 ==========="
  .venv/bin/python "$BENCH" --tier "$1" --seed "$2" \
    --arms base merged --merged "$MERGED" \
    --note "new-canonical 8192 tier=$1"
}

# quick first (cheapest; smoke-tests the 65536 engine), then medium
run quick 55001
run quick 55002
run medium 55003
run medium 55004
echo "=========== newbudget eval complete ==========="

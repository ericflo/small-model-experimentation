#!/usr/bin/env bash
# Compute-response study: does the merged medium-specialist's advantage over
# base GROW with the deployed think budget? This is the decisive test of C54's
# load-bearing claim that the medium wall is a SERIAL-COMPUTE limit (procedure
# known, cannot execute within the deployed budget) rather than a capability
# gap. Medium items at escalating budgets, paired base-vs-merged, fresh seeds.
# If delta climbs toward/past +0.32 with budget -> serial-compute confirmed and
# the wall is budget-tunable. If flat -> C54 is wrong; the wall is capability.
set -euo pipefail
cd /home/ericflo/Development/small-model-experimentation
MERGED=large_artifacts/qwen35_4b_gauntlet_frontier/merged/apex_cr
BENCH=experiments/qwen35_4b_gauntlet_frontier/scripts/bench.py

run() { # budget seed
  echo "=========== medium budget=$1 seed=$2 ==========="
  .venv/bin/python "$BENCH" --tier medium --seed "$2" \
    --arms base merged --merged "$MERGED" --think-budget "$1" \
    --note "compute-response medium tb=$1"
}

# canonical anchor + two escalation points, n=2 paired seeds each
run 1024 54001
run 1024 54002
run 2048 54003
run 2048 54004
run 4096 54005
run 4096 54006
echo "=========== sweep complete ==========="

#!/usr/bin/env bash
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation
until grep -qE 'saved adapter|Error|Traceback' experiments/qwen35_4b_gauntlet_frontier/runs/train_curric.log 2>/dev/null; do sleep 20; done
if ! grep -q 'saved adapter' experiments/qwen35_4b_gauntlet_frontier/runs/train_curric.log; then echo "TRAIN FAILED"; tail -20 experiments/qwen35_4b_gauntlet_frontier/runs/train_curric.log; exit 1; fi
echo "===== TRAIN DONE; eval BASE glyphgate ====="
.venv/bin/python experiments/qwen35_4b_gauntlet_frontier/scripts/eval_glyphgate_hf.py --n 15 --budget 3072
echo "===== eval CURRICULUM adapter glyphgate ====="
.venv/bin/python experiments/qwen35_4b_gauntlet_frontier/scripts/eval_glyphgate_hf.py --adapter large_artifacts/qwen35_4b_gauntlet_frontier/adapters/curric_induction --n 15 --budget 3072
echo "===== curric eval complete ====="

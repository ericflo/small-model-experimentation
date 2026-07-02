#!/usr/bin/env bash
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_context_composition
PY=../../.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
flt(){ grep -avE "it/s\]$|Loading|Fetching|FutureWarning|_check_is_size|triton|make_block|UserWarning|warnings.warn"; }
echo "=== BASE conds $(date +%T) ==="
$PY scripts/run_context.py --conds afc_plain afc_orch afc_icl ident_orch --out runs/ctx_base.json 2>&1 | flt | grep -aE "tasks|acc|wrote"
echo "=== SIM conds $(date +%T) ==="
$PY scripts/run_context.py --adapter runs/adapter_sim --conds afc_plain afc_orch ident_orch --out runs/ctx_sim.json 2>&1 | flt | grep -aE "tasks|acc|wrote"
echo "=== CTX_DONE $(date +%T) ==="

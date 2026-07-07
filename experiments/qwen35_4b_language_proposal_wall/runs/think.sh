cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_language_proposal_wall
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../../.venv/bin/python
F='it/s\]$|it/s,|Loading checkpoint|Fetching|UserWarning|warnings.warn|attention mask|pad token|Setting|generation flags|torch._check|Loading weights'
echo "=== ling THINK (induction) $(date +%H:%M:%S) ==="; $PY scripts/eval_proposal.py --render ling --n-per-depth 40 --depths 1 2 3 --n-ent 16 --think 2>&1 | grep -avE "$F"
echo "=== app THINK (application) $(date +%H:%M:%S) ==="; $PY scripts/eval_proposal.py --render app --n-per-depth 40 --depths 1 2 3 --n-ent 16 --think 2>&1 | grep -avE "$F"
echo "=== ALLDONE $(date +%H:%M:%S) ==="

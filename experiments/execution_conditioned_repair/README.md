# Execution-Conditioned Repair LoRA Experiment

This workspace implements the experiment brief in `/root/.codex/attachments/67be2ef6-a2f4-424d-a411-f0479b10c076/pasted-text-1.txt`.

Core question:

> Does single-adapter QLoRA SFT on `buggy repo + wrong patch + execution trace -> corrective diff` improve executable repair over ordinary final-patch SFT?

The code is deliberately organized around the brief's required deliverables:

- `scripts/build_repair_dataset.py`
- `scripts/sample_wrong_patches.py`
- `scripts/compute_corrective_diffs.py`
- `scripts/train_repair_lora.py`
- `scripts/eval_repair_synthetic.py`
- `scripts/eval_repair_swebench.py`
- `scripts/run_trace_ablation.py`

Current environment note: Docker can be installed but nested container execution is blocked by kernel namespace/mount restrictions in this runner. SWE-smith and SWE-bench scripts are present, but official Docker-backed evaluation is not considered verified until `scripts/eval_repair_swebench.py --preflight-only` passes.

Recommended first pass:

```bash
python scripts/build_repair_dataset.py --output-dir data
python scripts/compute_corrective_diffs.py --input data/repair_train.jsonl --output data/repair_train.checked.jsonl
python scripts/train_repair_lora.py --train data/repair_train.jsonl --eval data/repair_val_synth.jsonl --mode final_patch --output-dir models/final_patch_sft_lora
python scripts/train_repair_lora.py --train data/repair_train.jsonl --eval data/repair_val_synth.jsonl --mode no_trace --output-dir models/failure_conditioned_no_trace_lora
python scripts/train_repair_lora.py --train data/repair_train.jsonl --eval data/repair_val_synth.jsonl --mode trace --output-dir models/failure_conditioned_trace_lora
python scripts/eval_repair_synthetic.py --data data/repair_val_synth.jsonl --output reports/frozen_second_attempt_results.json --prompt-mode trace
python scripts/run_trace_ablation.py --data data/repair_val_synth.jsonl --adapter models/failure_conditioned_trace_lora --output reports/trace_ablation_results.json
python scripts/make_report.py --output reports/transfer_gap_report.md
```

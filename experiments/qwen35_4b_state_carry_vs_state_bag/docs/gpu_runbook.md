# Ada GPU Runbook

Run from the repository root. Use one GPU process at a time and `set -o pipefail` in any shell wrapper.

## 0. Read Before Running

1. Root `AGENTS.md`.
2. `docs/compute_environment.md`.
3. This experiment's `docs/research_handoff.md`.
4. `reports/preregistration.md`.
5. `reports/design_review.md`.
6. `reports/implementation_review.md`.

Never read or import `benchmarks/` contents. Do not substitute a smaller or more convenient model.

## 1. Rebuild the Pinned Training Environment

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --torch-backend=cu129 requirements-training.lock.txt
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=8 \
  uv pip install --python .venv/bin/python --no-build-isolation \
  --no-binary causal-conv1d causal-conv1d==1.6.2.post1
uv pip check --python .venv/bin/python
```

Verify the device and fast paths:

```bash
.venv/bin/python - <<'PY'
import torch, transformers, peft
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
assert torch.cuda.is_available()
p = torch.cuda.get_device_properties(0)
assert p.total_memory / 2**30 >= 44
assert is_causal_conv1d_available() and is_flash_linear_attention_available()
print(p.name, p.total_memory / 2**30, torch.__version__, transformers.__version__, peft.__version__)
PY
```

Expected core versions are recorded in `docs/compute_environment.md`. Stop on divergence until compatibility is understood.

## 2. CPU Contracts and Deterministic Data

```bash
python3 experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py --smoke
python3 -m unittest discover -s experiments/qwen35_4b_state_carry_vs_state_bag/tests -v
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
  --stage prepare-data
```

Inspect `experiments/qwen35_4b_state_carry_vs_state_bag/data/generated/manifest.json`. It must report zero structural duplicates and zero benchmark files read.

## 3. G0 Live Model Smoke

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
  --stage model-smoke
```

Inspect `runs/model_smoke/receipt.json`. Require:

- `MODEL_SMOKE_PASS`;
- exact model/revision/backend;
- K=1 error ≤ `1e-5`;
- state slot count 8;
- Carry/Bag parameter equality;
- matching initial-value and cumulative training-compute receipts for every later Carry/Bag seed pair;
- nonzero LoRA, state, and sufficiency gradients;
- no OOM and credible peak VRAM.

Do not train if any receipt is absent. Fix mechanics without changing the scientific contract, add a regression test, rerun G0, and record the fix in `experiment_log.md`.

## 4. G1 Paired Pilot

Use the same default config and seed; `--pilot` changes only the registered step/evaluation limits.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
  --stage train --pilot --arm carry --seed 7411 \
  --output large_artifacts/qwen35_4b_state_carry_vs_state_bag/pilot_carry_seed7411

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
  --stage train --pilot --arm bag --seed 7411 \
  --output large_artifacts/qwen35_4b_state_carry_vs_state_bag/pilot_bag_seed7411
```

Evaluate the fixed 300-step checkpoints:

```bash
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
  --stage evaluate --pilot --arm carry --seed 7411 \
  --checkpoint large_artifacts/qwen35_4b_state_carry_vs_state_bag/pilot_carry_seed7411/checkpoint_000300 \
  --output experiments/qwen35_4b_state_carry_vs_state_bag/runs/pilot_carry_seed7411

.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
  --stage evaluate --pilot --arm bag --seed 7411 \
  --checkpoint large_artifacts/qwen35_4b_state_carry_vs_state_bag/pilot_bag_seed7411/checkpoint_000300 \
  --output experiments/qwen35_4b_state_carry_vs_state_bag/runs/pilot_bag_seed7411
```

Run analysis, but remember the expected label is under-replicated:

```bash
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py --stage analyze
```

Apply G1 exactly. Do not try another seed after a miss. Diagnose whether the failure is optimization, state formation, coda use, or collapse.

## 5. G2 Full Continuous Carry/Bag

If G1 promotes, train all six fixed runs from scratch:

```bash
for seed in 7411 7412 7413; do
  for arm in carry bag; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
      --stage train --arm "$arm" --seed "$seed" \
      --output "large_artifacts/qwen35_4b_state_carry_vs_state_bag/${arm}_seed${seed}"
  done
done
```

Evaluate only fixed final checkpoints:

```bash
for seed in 7411 7412 7413; do
  for arm in carry bag; do
    .venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
      --stage evaluate --arm "$arm" --seed "$seed" \
      --checkpoint "large_artifacts/qwen35_4b_state_carry_vs_state_bag/${arm}_seed${seed}/checkpoint_001500" \
      --output "experiments/qwen35_4b_state_carry_vs_state_bag/runs/full_${arm}_seed${seed}"
  done
done
```

Then analyze:

```bash
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py --stage analyze
```

## 6. G3 Inference Edge Cut

For every trained Carry checkpoint, evaluate the exact same weights in Bag mode. Use separate output directories; checkpoint metadata preserves `train_arm=carry`, while rows record `eval_mode=bag`.

```bash
for seed in 7411 7412 7413; do
  .venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
    --stage evaluate --arm bag --seed "$seed" \
    --checkpoint "large_artifacts/qwen35_4b_state_carry_vs_state_bag/carry_seed${seed}/checkpoint_001500" \
    --output "experiments/qwen35_4b_state_carry_vs_state_bag/runs/edge_cut_seed${seed}"
done
```

Donor swaps are already included in Carry evaluation summaries. Inspect `counterfactual_swaps.jsonl`; aggregate damage without donor following is not a causal pass.

## 7. G4 Mixed Semantic Echo, Only If Triggered

Use `configs/mixed_echo.yaml` without modifying `default.yaml`. Run model smoke under that config, then paired Carry/Bag pilot and full multiseed only if the preregistered interface signature exists.

```bash
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
  --config experiments/qwen35_4b_state_carry_vs_state_bag/configs/mixed_echo.yaml \
  --stage model-smoke \
  --output experiments/qwen35_4b_state_carry_vs_state_bag/runs/model_smoke_mixed/receipt.json
```

Every later mixed command must pass the same `--config`. Analyze mixed and continuous configs separately; config hashes prevent accidental pooling.

## 8. G5 Explicit-CoT and Sample-More

Only after a mechanistic pass:

```bash
for seed in 7411 7412 7413; do
  .venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
    --stage text-baseline --seed "$seed" \
    --output "large_artifacts/qwen35_4b_state_carry_vs_state_bag/text_seed${seed}"

  .venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
    --stage sample-more --seed "$seed" \
    --checkpoint "large_artifacts/qwen35_4b_state_carry_vs_state_bag/text_seed${seed}" \
    --output "experiments/qwen35_4b_state_carry_vs_state_bag/runs/sample_more_seed${seed}"
done
```

The current text baseline saves its final adapter at the run root. Verify every sample row has `sample_layer_token_budget <= recurrent_layer_token_budget`; any violation invalidates the comparator. A deployable verdict also requires all three seed-matched comparisons and a positive paired lower bound against oracle `pass@N`.

## 9. Finalize Evidence

After a terminal result:

1. Update `reports/report.md` without deleting setup history.
2. Update `reports/artifact_manifest.yaml` with every external checkpoint and hash.
3. Preserve negative/stopped rows.
4. Update program evidence/backlog and synthesis only if strategy changed.
5. Add native chart data and the practitioner brief.
6. Run `make check`.
7. Commit, push, and inspect `gh run list`.

## Recovery

- Never overwrite an existing run directory; the harness refuses.
- If OOM occurs, stop concurrent processes and follow the CUDA recovery section in `docs/compute_environment.md`.
- Do not lower K, sequence content, state slots, or layer count in a result-bearing retry.
- A mechanics-only code fix requires a new smoke receipt but not a new scientific experiment.
- A design change requires a successor experiment directory.

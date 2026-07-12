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
It must also contain the current critical-source digest, distinct pilot/confirmation seeds,
balanced node/checksum counts by cell, and the dedicated `pilot_validation`, `pilot_depth`,
`pilot_joint`, and `pilot_counterfactual` splits. The historical committed CPU receipt is
superseded; do not reuse it.

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
- the expected shared-wrapper trainable parameter identity;
- nonzero finite LoRA, state, step, and sufficiency gradients for both Carry and Bag;
- successful worst-format K=12 evaluation forward;
- matching config/source/training-lock receipts; and
- no OOM plus credible K=4-backward/K=12-forward timing and peak VRAM.

Later Carry/Bag initialization, ordered-row, and cumulative training-compute equality cannot exist at
G0; inspect those immediately after each pair completes and again in analysis. Do not train if any
G0 receipt is absent. Fix mechanics without changing the scientific contract, add a regression test,
regenerate source-bound data, write a new smoke path, and record the fix in `experiment_log.md`.

## 4. G1 Paired Pilot

Use the same default config and the pilot-only seed 7401; `--pilot` selects the registered 300-step
phase and dedicated pilot splits. The CLI rejects confirmatory seeds in pilot mode.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
  --stage train --pilot --arm carry --seed 7401 \
  --output large_artifacts/qwen35_4b_state_carry_vs_state_bag/pilot_carry_seed7401

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
  --stage train --pilot --arm bag --seed 7401 \
  --output large_artifacts/qwen35_4b_state_carry_vs_state_bag/pilot_bag_seed7401
```

Evaluate the fixed 300-step checkpoints:

```bash
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
  --stage evaluate --pilot --arm carry --seed 7401 \
  --checkpoint large_artifacts/qwen35_4b_state_carry_vs_state_bag/pilot_carry_seed7401/checkpoint_000300 \
  --output experiments/qwen35_4b_state_carry_vs_state_bag/runs/pilot_carry_seed7401

.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
  --stage evaluate --pilot --arm bag --seed 7401 \
  --checkpoint large_artifacts/qwen35_4b_state_carry_vs_state_bag/pilot_bag_seed7401/checkpoint_000300 \
  --output experiments/qwen35_4b_state_carry_vs_state_bag/runs/pilot_bag_seed7401
```

Run analysis and require its machine promotion decision:

```bash
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py --stage analyze
```

Require `analysis/summary.json` to say `PILOT_PROMOTION_READY`; full training refuses any other
status. Do not try another seed after `PILOT_MECHANISM_MISS`. Diagnose whether the failure is
optimization, state formation, coda use, or collapse without touching confirmation data.

## 5. G2 Full Continuous Carry/Bag

If G1 promotes, train all six fixed runs from scratch:

```bash
set -euo pipefail
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
set -euo pipefail
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
set -euo pipefail
for seed in 7411 7412 7413; do
  .venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
    --stage evaluate --arm bag --seed "$seed" \
    --checkpoint "large_artifacts/qwen35_4b_state_carry_vs_state_bag/carry_seed${seed}/checkpoint_001500" \
    --output "experiments/qwen35_4b_state_carry_vs_state_bag/runs/edge_cut_seed${seed}"
done

.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
  --stage analyze
```

The harness recognizes this exact same-checkpoint edge cut and evaluates only the full primary
matched-depth cells used by the gate. Donor swaps are included in Carry evaluation summaries in both
directions. Inspect the hashed `counterfactual_swaps.jsonl`; require geometry equality, the full 1,024
directed rows per seed, positive pre/post donor following, and donor following rather than generic
recipient damage.

## 7. G4 Explicit-CoT and Sample-More

Only after a mechanistic pass:

```bash
set -euo pipefail
for seed in 7411 7412 7413; do
  .venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
    --stage text-baseline --seed "$seed" \
    --output "large_artifacts/qwen35_4b_state_carry_vs_state_bag/text_seed${seed}"

  .venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
    --stage sample-more --seed "$seed" \
    --checkpoint "large_artifacts/qwen35_4b_state_carry_vs_state_bag/text_seed${seed}" \
    --output "experiments/qwen35_4b_state_carry_vs_state_bag/runs/sample_more_seed${seed}"
done

.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py \
  --stage analyze
```

The current text baseline saves its final adapter at the run root. The sampler uses the frozen
depth-aware allowance and stores raw token IDs/text. Verify every row has
`sample_layer_token_budget <= recurrent_layer_token_budget`; require exactly 3,200 common IDs per
seed, Carry answer-mode ≥95%, explicit-CoT parse ≥95%, and cap contact ≤5%. Any violation makes the
deployment comparator interface-invalid. A deployable verdict also requires all three seed-matched
comparisons and a positive crossed task×seed lower bound against oracle `pass@N`.

## 8. Resource and interruption policy

The frozen full plan is intentionally large: six recurrent trainings consume 144,000 microforwards
plus periodic validation; the original evaluation geometry is roughly 123,000 item-level forwards,
and G4 adds 72,000 text-training microforwards plus 9,600 generation calls. Edge-cut primary-only
mode removes most of its former redundant evaluation. Use the G0 and pilot receipts to project wall
time before G2; record the projection and free durable storage. Do not infer throughput from an older
L40 label—the runtime receipt is authoritative for the actual ≥44 GiB Ada device.

Training is deliberately non-resumable because exact optimizer/RNG/data-cursor state is not stored.
On interruption, preserve the partial attempt, record the failure, and restart from step zero in a
fresh attempt directory. Never evaluate a partial/intermediate checkpoint. Evaluation and generation
partial files are likewise preserved, not appended into a new attempt.

## 9. Finalize Evidence

After a terminal result:

1. Update `reports/report.md` without deleting setup history.
2. Update `reports/artifact_manifest.yaml` with every external checkpoint and hash.
3. Preserve negative/stopped rows.
4. Update program evidence/backlog and synthesis only if strategy changed.
5. Add native chart data and the practitioner brief.
6. Run `make check`.
7. Commit, push, and inspect `gh run list`.

If a valid terminal LoRA outcome fails to establish deep state formation, this is not the terminal end
of the research goal. Create **and execute** the fresh zero-initialized full-rank extra-R-delta
successor specified in preregistration section 10. Mechanics/data failures and mathematically
infeasible gates instead require repair or design review. If state is strongly readable but causally
unused, execute the controlled interface successor; a sample-more-only loss requires neither capacity
follow-up. Keep Qwen/Qwen3.5-4B, the ordinary K=1 path, pilot firewall, Carry/Bag equality, crossed
analysis, and causal gates fixed. Do not retrofit either follow-up into this result-bearing directory.

## Recovery

- Never overwrite an existing run directory; the harness refuses.
- Never approximate-resume training; restart at step zero in a new attempt directory and preserve the partial one.
- If OOM occurs, stop concurrent processes and follow the CUDA recovery section in `docs/compute_environment.md`.
- Do not lower K, sequence content, state slots, layer count, sample allowance, or validity gates in a result-bearing retry.
- A mechanics-only code fix requires a new smoke receipt but not a new scientific experiment.
- A design change requires a successor experiment directory.

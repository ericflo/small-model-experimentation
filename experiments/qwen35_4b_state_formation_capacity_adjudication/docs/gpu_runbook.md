# GPU Runbook

**Execution status:** not started. This file defines the required phase order; it is not evidence that
the current implementation satisfies the contract. Do not run a model-bearing command until the
preregistration, adversarial design review, implementation review, tests, and immutable design receipt
are complete.

Use one exclusive RTX 6000 Ada process and the pinned Transformers training environment. Never run a
second GPU job concurrently.

```bash
cd /workspace/small-model-experimentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
RUN=experiments/qwen35_4b_state_formation_capacity_adjudication/scripts/run.py
EXP=experiments/qwen35_4b_state_formation_capacity_adjudication
```

Every command must exit successfully without a pipe hiding its status. The CLI contract below must be
present in `--help` before execution; if an option is absent, implementation is incomplete—do not
approximate it with the copied predecessor interface.

## 1. Environment and repository preflight

```bash
nvidia-smi
uv pip check --python "$PY"
$PY - <<'PY'
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)
assert is_causal_conv1d_available()
assert is_flash_linear_attention_available()
PY
$PY -B -m unittest discover -s "$EXP/tests" -v
$PY "$RUN" --stage cpu-smoke
git diff --check
```

Confirm exactly one physical `NVIDIA RTX 6000 Ada Generation` (compute capability 8.9) is visible,
has at least 44 GiB free, has no competing process, and matches the exact training lock. Its UUID and
stable runtime/setup receipt must remain identical across every reached arm and evaluation. Confirm
no benchmark access. CPU smoke is setup-only.

## 2. Freeze design and generate fresh data

After the reviewed design and implementation are committed and pushed, require every registered
design/source/test/lock input to be tracked and clean at that HEAD, then create the canonical
immutable receipt:

```bash
$PY "$RUN" --stage design-boundary \
  --output "$EXP/reports/design_receipt.json"
```

It must bind the exact scientific-design bytes of `idea_intake.md`, `reports/preregistration.md`,
`reports/design_review.md`, `docs/architecture.md`, `docs/gpu_runbook.md`,
`docs/research_handoff.md`, and `configs/default.yaml`. It records source/test/implementation-review
and lock provenance at the clean HEAD but freezes scientific design only. Every downstream artifact
separately binds the current source/test and lock digests, so a mechanical implementation repair
requires regeneration of prior data/init/setup/result artifacts without rewriting the design
receipt. It deliberately does not bind the mutable post-result README or report.

Inspect it first: require exact status `DESIGN_FROZEN`, `scientific_evidence: false`,
`benchmark_files_read: 0`, and a valid canonical frozen-file manifest identity.

```bash
$PY "$RUN" --stage prepare-data --output "$EXP/data/generated"
```

Inspect the tracked manifest. It must contain seven frozen result splits, seeds 73301–73307, and exact
row counts. Trigger validation has 256 rows at each depth 1–4; sealed `contrast_validation` has 768
rows at depths 2–4, 256/depth; every depth-5–12 split has 128 rows/depth. Require zero structural
overlap, canonical decompressed hashes, frozen gzip metadata, critical-source/config/lock identities,
and `benchmark_files_read: 0`.

Confirm the validator is tiered. Before `STAGE_B_CONTRAST_AUTHORIZED`, later stages may verify each
sealed contrast's manifest identity, compressed-byte SHA-256/size, and gzip header only. They must
not decompress or canonical-reopen any of the three sealed payloads. The authorized evaluator must
append its Stage-B-receipt/checkpoint/cell identity to the access ledger before first decompression;
subsequent authorized calls may see only entries bound to that same receipt and frozen cell set.
Confirm ledger writes hold the separate stable `.lock` inode across an atomic,
temporary-file-and-parent-directory-fsynced ledger replacement.

The 48-row positive-control corpus uses separate setup seed 73991 and a separate generation receipt.
It has exactly two rows in each depth 2/3/4 × query kind × training family × training template cell
(24 cells total); it is not part of the result manifest or any scientific metric.

Do not open `contrast_validation`, `contrast_depth`, or `contrast_joint` through a model stage,
generic data validator, evaluator, or analyzer. Initial generation and compressed-file verification
do not spend their outcome firewall; post-generation decompression does.

## 3. Create common initialization bundles

```bash
for seed in 7411 7412 7413; do
  $PY "$RUN" --stage prepare-init --seed "$seed" \
    --output "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/initialization_seed${seed}.pt"
done
```

For each seed, inspect both the external `.pt.json` sidecar and its byte-identical tracked mirror at
`runs/setup/initialization_seed${seed}.json`. They must bind the external file hash,
canonical tensor-value digest, every tensor name/shape/dtype/hash, seed, bundle version, and frozen
identities. Reopen each tensor bundle and reproduce the receipt before continuing.

## 4. LoRA joint mechanics and setup positive control

G0 and the positive control are capacity/seed-specific and objective-independent. Run them for every
result seed; conditional LoRA state-only training reuses the same seed-matched receipts.

```bash
for seed in 7411 7412 7413; do
  $PY "$RUN" --stage model-smoke --capacity lora --seed "$seed" \
    --initialization-bundle "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/initialization_seed${seed}.pt" \
    --output "$EXP/runs/setup/g0_lora_seed${seed}.json"
  $PY "$RUN" --stage positive-control --capacity lora --seed "$seed" \
    --initialization-bundle "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/initialization_seed${seed}.pt" \
    --model-smoke-receipt "$EXP/runs/setup/g0_lora_seed${seed}.json" \
    --output "$EXP/runs/setup/positive_control_lora_seed${seed}.json"
done
```

Inspect, do not merely read the terminal strings. Require:

- pinned model/layers and ordered 62-target manifest;
- 16,232,448 LoRA parameters, rank 32, alpha 64, dropout 0.05;
- deterministic copied-tensor custom-hook output and A/B-gradient parity to the pinned PEFT `Linear`
  reference at alpha/r = 2, without loading a second Qwen model, in both frozen regimes: exact
  FP32/dropout-off/autocast-off at `atol=1e-6`, `rtol=1e-5`, and live-like
  bf16-autocast/dropout-0.05 at `atol=2e-3`, `rtol=1e-2`; for the latter, require the same device RNG
  reset immediately before each forward, the corresponding first native-dropout mask position, and
  the custom one-call/one-cycle realized-mask receipt;
- exact common-state equality under deliberately shifted construction RNG;
- zero learned output initially, exact K=1 parity, and exactly 186 K=4/682 K=12 calls;
- matched dropout schedule and realized probe masks;
- an inventory proving every non-adaptation dropout-like module is absent, disabled, or probability
  zero, with no unreceipted RNG consumer in the controlled forward;
- finite two-step gradients in adaptation/common modules and none in base;
- a finite live K=4 joint backward/optimizer probe, including answer loss and nonzero aggregate,
  adaptation, initializer, step, damping, and state-head gradients with no base gradient;
- separate common/adaptation clipping receipts;
- a timed 10-step estimate, peak VRAM, and no nonfinite values; and
- a fresh setup-only K=12 row from seed 73992, structurally disjoint from all result rows, while G0
  opens only the training payload and no trigger/contrast payload; at least 4 GiB free VRAM after G0;
- fixed-256-update positive-control terminal joint accuracy at least 0.95 plus oracle analyzer
  accuracy at least 0.99.

Any failure stops for repair. It does not authorize full rank.

## 5. Stage A: three LoRA joint fixed-final runs

Run one process at a time. Use explicit fresh output directories; never overwrite or resume an
attempt.

```bash
for seed in 7411 7412 7413; do
  $PY "$RUN" --stage train --capacity lora --objective joint --seed "$seed" \
    --initialization-bundle "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/initialization_seed${seed}.pt" \
    --model-smoke-receipt "$EXP/runs/setup/g0_lora_seed${seed}.json" \
    --positive-control-receipt "$EXP/runs/setup/positive_control_lora_seed${seed}.json" \
    --output "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/lora_joint_seed${seed}"
done
```

After each launch, verify GPU memory and observed step time against the timed smoke. Only step 1500
may be saved. Check that row-order, dropout schedule/probe, common-init, every-step groupwise
preclip/applied-scale receipt, separately logged adaptation/common-state learning rates, complete
optimizer-state receipt, and metrics hashes are embedded in checkpoint metadata and mirrored under
`runs/training/<capacity>_<objective>_seed<seed>/`. Require analysis to recompute the registered
learning-rate schedule independently for both groups at every exact step.

After all three checkpoints are sealed, evaluate each on the trigger set. One invocation writes both
intact and adaptation-disabled rows and summaries:

```bash
for seed in 7411 7412 7413; do
  $PY "$RUN" --stage evaluate-state --capacity lora --objective joint --seed "$seed" \
    --eval-set trigger \
    --checkpoint "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/lora_joint_seed${seed}/checkpoint_001500" \
    --output "$EXP/runs/lora_joint_seed${seed}_trigger"
done
$PY "$RUN" --stage analyze --phase lora_joint \
  --output "$EXP/analysis/lora_joint_trigger.json"
```

Recompute cell counts and hashes independently. Depth 1 is reported but cannot pass an adapter claim.
The analyzer must expose every seed×depth terminal joint cell, refuse pooled substitution, and report
the intact-versus-disabled adaptation status. `ADAPTATION_DISABLED_REVERSAL` means intact misses while
disabled passes; preserve it as adapter-interference evidence, but keep the branch determined by the
intact formation status.

- `LORA_DOES_NOT_PREVENT_STATE_FORMATION`: stop; later arms are prohibited.
- `LORA_JOINT_MISS_CONTROLS_REQUIRED`: continue to Stage B; the exact receipt is mandatory.
- incomplete, mismatch, or setup failure: repair; do not continue.

Do not score the contrast splits at this stage.

## 6. Stage B setup: LoRA state-only and direct full-shape joint

Only an exact `LORA_JOINT_MISS_CONTROLS_REQUIRED` receipt licenses these commands.

LoRA state-only reuses the already passed LoRA seed receipts. Full rank needs its own seed-matched G0
and state-path positive control, both bound to the LoRA miss authorization:

```bash
for seed in 7411 7412 7413; do
  $PY "$RUN" --stage model-smoke --capacity fullrank --seed "$seed" \
    --initialization-bundle "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/initialization_seed${seed}.pt" \
    --authorization-receipt "$EXP/analysis/lora_joint_trigger.json" \
    --output "$EXP/runs/setup/g0_fullrank_seed${seed}.json"
  $PY "$RUN" --stage positive-control --capacity fullrank --seed "$seed" \
    --initialization-bundle "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/initialization_seed${seed}.pt" \
    --model-smoke-receipt "$EXP/runs/setup/g0_fullrank_seed${seed}.json" \
    --authorization-receipt "$EXP/analysis/lora_joint_trigger.json" \
    --output "$EXP/runs/setup/positive_control_fullrank_seed${seed}.json"
done
```

Full-rank G0 must additionally prove exactly 892,272,640 zero-initialized FP32 delta parameters,
finite shape-matched Adam moments for all 62 deltas, exact K=1 before and after a real optimizer step,
finite K=12, memory headroom, and destructive checkpoint restoration. OOM is a feasibility stop, not
a LoRA result. Do not reduce targets, precision, or steps.

## 7. Stage B training and trigger evaluation

```bash
for seed in 7411 7412 7413; do
  $PY "$RUN" --stage train --capacity lora --objective state_only --seed "$seed" \
    --initialization-bundle "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/initialization_seed${seed}.pt" \
    --authorization-receipt "$EXP/analysis/lora_joint_trigger.json" \
    --model-smoke-receipt "$EXP/runs/setup/g0_lora_seed${seed}.json" \
    --positive-control-receipt "$EXP/runs/setup/positive_control_lora_seed${seed}.json" \
    --output "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/lora_state_only_seed${seed}"

  $PY "$RUN" --stage train --capacity fullrank --objective joint --seed "$seed" \
    --initialization-bundle "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/initialization_seed${seed}.pt" \
    --authorization-receipt "$EXP/analysis/lora_joint_trigger.json" \
    --model-smoke-receipt "$EXP/runs/setup/g0_fullrank_seed${seed}.json" \
    --positive-control-receipt "$EXP/runs/setup/positive_control_fullrank_seed${seed}.json" \
    --output "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/fullrank_joint_seed${seed}"
done
```

Evaluate both new arm families on the trigger set. Each invocation produces intact and disabled
outputs:

```bash
for seed in 7411 7412 7413; do
  $PY "$RUN" --stage evaluate-state --capacity lora --objective state_only --seed "$seed" \
    --eval-set trigger \
    --checkpoint "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/lora_state_only_seed${seed}/checkpoint_001500" \
    --authorization-receipt "$EXP/analysis/lora_joint_trigger.json" \
    --output "$EXP/runs/lora_state_only_seed${seed}_trigger"
  $PY "$RUN" --stage evaluate-state --capacity fullrank --objective joint --seed "$seed" \
    --eval-set trigger \
    --checkpoint "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/fullrank_joint_seed${seed}/checkpoint_001500" \
    --authorization-receipt "$EXP/analysis/lora_joint_trigger.json" \
    --output "$EXP/runs/fullrank_joint_seed${seed}_trigger"
done
$PY "$RUN" --stage analyze --phase lora_control \
  --authorization-receipt "$EXP/analysis/lora_joint_trigger.json" \
  --output "$EXP/analysis/lora_control.json"
```

Require complete trigger evaluation before touching sealed contrasts. Then create the dedicated
Stage-B seal; the earlier LoRA-miss receipt alone is deliberately insufficient:

```bash
$PY "$RUN" --stage analyze --phase stage_b_seal \
  --authorization-receipt "$EXP/analysis/lora_joint_trigger.json" \
  --output "$EXP/analysis/stage_b_seal.json"
```

This receipt must reopen and bind the three LoRA-joint, three LoRA-state-only, and three
full-rank-joint step-1500 checkpoint identities; their complete intact/disabled trigger-evaluation
identities; the LoRA-control analysis; and all source/config/lock/data/init/order/dropout lineage. It
must also prove from the experiment access ledger and absent contrast-evaluation/output paths that
none of the three contrast payloads has previously been opened for scoring. This phase may inspect the
frozen contrast manifest identities, but must not read contrast rows or labels.

Inspect its exact status before doing anything else:

- `FULLRANK_STATE_ONLY_REQUIRED`: at least one direct-full-shape joint trigger cell missed. Skip every
  contrast command; the sealed rows remain unopened, and this receipt directly licenses Stage C.
- `STAGE_B_CONTRAST_AUTHORIZED`: every direct-full-shape joint trigger cell passed. Only this status
  licenses the contrast commands below.
- `CONTRAST_FIREWALL_NOT_READY`: a cell, lineage, or access proof is invalid or incomplete. Stop for
  inspection; it licenses neither contrasts nor Stage C. Missing preaccess lineage may be repaired
  while the ledger is demonstrably empty. An actual or unexplained premature-open event burns these
  contrast rows permanently: preserve it and create a fresh preregistered successor rather than
  deleting the ledger or regenerating this experiment's splits.

Only after `STAGE_B_CONTRAST_AUTHORIZED`, evaluate **LoRA joint and full-rank joint** on the three
sealed splits in the same invocation order and with exact row hashes. The loop below remains exactly
six capacity×seed evaluation jobs; each call writes intact and disabled rows for
`contrast_validation`, `contrast_depth`, and `contrast_joint`:

```bash
for capacity in lora fullrank; do
  for seed in 7411 7412 7413; do
    $PY "$RUN" --stage evaluate-state --capacity "$capacity" --objective joint --seed "$seed" \
      --eval-set contrast \
      --checkpoint "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/${capacity}_joint_seed${seed}/checkpoint_001500" \
      --authorization-receipt "$EXP/analysis/stage_b_seal.json" \
      --output "$EXP/runs/${capacity}_joint_seed${seed}_contrast"
  done
done
$PY "$RUN" --stage analyze --phase fullrank_joint \
  --authorization-receipt "$EXP/analysis/stage_b_seal.json" \
  --output "$EXP/analysis/fullrank_joint.json"
```

Before interpreting the receipt, confirm the analyzer reopened the Stage-B-bound full-rank-joint,
LoRA-joint, and LoRA-state-only trigger evaluations, recomputed all three formation summaries, and
required the two recomputed LoRA summaries to equal the values cached by Stage B.

Inspect absolute cell gates, intact-minus-disabled status separately for LoRA and full rank,
full-rank-minus-LoRA paired effects, every seed sign, every depth sign, and all three crossed lower
bounds. After the analyzer establishes complete, valid evidence, the first scientific terminal check
is LoRA intact formation across the complete sealed trained-depth, depth-extrapolation, and
joint-shift matrix.

- `BRANCH_EVIDENCE_INCOMPLETE` or invalid evidence: repair; no scientific branch is licensed.
- `LORA_TRIGGER_MISS_NOT_REPLICATED_ON_SEALED_CONTRAST`: LoRA intact passes every required cell on all
  three sealed splits; stop immediately and prohibit rescue, rank-limit, and Stage-C claims regardless
  of the full-rank score.
- `FULLRANK_STATE_ONLY_REQUIRED`: LoRA does not fully pass and at least one full-rank sealed absolute
  cell failed; continue to Stage C before applying the rescue-replication guard.
- `LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_ON_SEALED_CONTRAST`: full rank passes every trigger
  and sealed absolute cell, but at least one LoRA category that failed on trigger passes in its
  corresponding sealed domain. Stop and prohibit rescue/rank-limit claims; additional LoRA failures
  in other sealed categories do not substitute.
- `DIRECT_FULLSHAPE_RECIPE_RESCUE`: capacity question answered; stop.
- `DIRECT_FULLSHAPE_RECIPE_PASS_CONTRAST_UNCERTAIN`: every trigger and sealed full-rank absolute cell
  passed, but adaptation-dependence or paired contrast is uncertain; report uncertainty and stop.

Inspect `lora_trigger_failure_replication`: trigger `trained`, `depth`, and `joint` failures must recur
in `contrast_validation`, `contrast_depth`, and `contrast_joint`, respectively. Extra sealed failures
are allowed. Only a full-rank absolute pass reaches this guard; the all-cell LoRA pass and full-rank
miss branches above retain priority.

The analyzer must also expose `lora_sealed_contrast_adaptation` and the full-rank sealed adaptation
status. If either is `ADAPTATION_DISABLED_REVERSAL`, report that intact failed while disabled passed;
it never satisfies adaptation dependence. In particular, a sealed LoRA reversal remains visible even
though the nonreplication stop above is defined by LoRA **intact** formation.

## 8. Stage C: conditional direct full-shape state-only

Only an exact `FULLRANK_STATE_ONLY_REQUIRED` receipt licenses this arm. It comes from
`analysis/stage_b_seal.json` when the trigger already missed, or from `analysis/fullrank_joint.json`
when the trigger passed but a sealed absolute cell missed. Set the path to the receipt that actually
emitted the status; never choose it by hand from a score:

```bash
# Trigger miss path. For a sealed-cell miss, use analysis/fullrank_joint.json instead.
FULLRANK_MISS_RECEIPT="$EXP/analysis/stage_b_seal.json"
```

The full-rank state-only control reuses each seed's already passed full-rank G0 and positive-control
receipt; those setup gates test the state path and backend, not the answer-loss weight.

```bash
for seed in 7411 7412 7413; do
  $PY "$RUN" --stage train --capacity fullrank --objective state_only --seed "$seed" \
    --initialization-bundle "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/initialization_seed${seed}.pt" \
    --authorization-receipt "$FULLRANK_MISS_RECEIPT" \
    --model-smoke-receipt "$EXP/runs/setup/g0_fullrank_seed${seed}.json" \
    --positive-control-receipt "$EXP/runs/setup/positive_control_fullrank_seed${seed}.json" \
    --output "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/fullrank_state_only_seed${seed}"
done
```

Evaluate both modes through the single trigger invocation, then run:

```bash
for seed in 7411 7412 7413; do
  $PY "$RUN" --stage evaluate-state --capacity fullrank --objective state_only --seed "$seed" \
    --eval-set trigger \
    --checkpoint "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/fullrank_state_only_seed${seed}/checkpoint_001500" \
    --authorization-receipt "$FULLRANK_MISS_RECEIPT" \
    --output "$EXP/runs/fullrank_state_only_seed${seed}_trigger"
done
$PY "$RUN" --stage analyze --phase fullrank_control \
  --authorization-receipt "$FULLRANK_MISS_RECEIPT" \
  --output "$EXP/analysis/summary.json"
```

The final status must be one of the registered state-only patterns or repair-required. Never infer a
missing conditional arm as a miss. The terminal analyzer must reopen the Stage-B-bound
LoRA-state-only and full-rank-joint trigger predecessors and reject any evaluation-lineage mismatch.
It recomputes formation from the current LoRA- and full-rank-state-only trigger rows and proves a
Stage-C matching receipt: the three full-rank state-only runs match both predecessors on shared
initialization, data, row order, dropout schedule/probes, and exact full-rank
G0/positive-control reuse.
It separately reports each state-only arm's adaptation diagnostic; preserve
`ADAPTATION_DISABLED_REVERSAL` when disabled passes the required trigger matrix but intact does not,
without changing the intact-based four-pattern terminal taxonomy.

## 9. Final audit and reporting

Before editing results prose:

1. Reopen every upstream receipt and verify its file SHA and canonical identity.
2. Recompute every checkpoint, metrics, evaluation-row, and summary hash.
3. Confirm all result evaluations used only step 1500 and exact K=semantic depth.
4. Confirm common initialization, row-order, dropout schedule/probes, and split hashes pair exactly.
5. Confirm all seed×depth cells and unique-task counts; never call model×task rows independent.
6. Confirm prohibited branches have no model artifacts.
7. Update `reports/artifact_manifest.yaml` with realized external paths and exact runtime hashes.
8. Update README, report, experiment log, program evidence/backlog, and shared knowledge only to the
   degree licensed by the terminal result.
9. Run `make check`, then inspect CI after publication.

There is no Bag, causal swap, edge cut, textual-state, vLLM, benchmark, or sample-more command in this
runbook.

## 10. Failure and recovery

- OOM/resource failure: preserve logs and receipt, verify exclusive GPU use, and follow
  `docs/compute_environment.md`; do not alter scientific geometry.
- Nonfinite loss, wrong target/call count, common-init mismatch, dropout mismatch, stale lineage,
  missing row, or failed positive control: preserve the attempt and repair mechanics. Regenerate all
  downstream artifacts whose bound source/config changed.
- Interrupted result training: do not resume. Move the incomplete canonical directory, unchanged,
  under `large_artifacts/qwen35_4b_state_formation_capacity_adjudication/failed_attempts/` with a
  content-addressed attempt manifest, then restart step zero at the same canonical result path. Use
  the source-bound helper, passing the tracked training companion too if it exists:

  ```bash
  $PY "$EXP/scripts/archive_failed_attempt.py" \
    --path <canonical-incomplete-directory> \
    [--path <canonical-tracked-companion>]
  ```

  Commit the generated `runs/failures/*.json` receipt before retry.
- Interrupted authorized contrast evaluation: run the same helper on the one incomplete canonical
  evaluation directory. It refuses a directory containing completed `summary.json`; completed
  evaluations cannot be archived or replayed as failures. Inspect and commit the tracked
  `FAILED_ATTEMPT_ARCHIVED` receipt, which must be byte-identical to the external archive receipt and
  content-validate the preserved incomplete tree.
- Then rerun only the identical capacity/objective/seed, fixed checkpoint, Stage-B receipt, and
  canonical output path. The runner requires the newly recreated canonical output to be empty and
  discovers exactly one valid archive receipt not already bound to that cell. Before decompression it
  appends that receipt's path/hash/identity/archive lineage to the existing ledger event, recomputes
  the event and ledger identities, and atomically replaces the ledger while holding its separate
  stable lock inode; both temporary file and parent directory are fsynced. Initial access rejects an
  archive that predates its event. Zero or multiple new receipts fail closed.
  Every later retry requires one further newly archived attempt; an already-bound receipt can never
  license another replay. Never delete or edit the ledger or archive history.
- No seed shopping, intermediate checkpoint evaluation, threshold repair, or partial-arm pooling.

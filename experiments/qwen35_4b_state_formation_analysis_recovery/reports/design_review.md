# Adversarial Design Review

**Decision:** `RECOVERY_GO` for smoke and, only after the complete three-seed LoRA trigger matrix,
the exact registered v11 analysis phases.

**Review date:** 2026-07-13

## Scope

This experiment may repair an operational consumer boundary only. It may not alter the producer's
model, data, checkpoints, objectives, metrics, thresholds, bootstrap, branch taxonomy, sealed-data
firewall, or scientific analysis. It loads no model and creates no new result arm.

## Observed failure

The source-v11 analyzer constructs its external run directory by joining its experiment root to the
frozen config value `../../large_artifacts/qwen35_4b_state_formation_capacity_adjudication`.
`Path` preserves those components in the raw absolute spelling. `_canonical_expected_path` compares
that spelling with `abspath(raw)` and rejects it before calling the later repository-relative,
no-symlink validator. The independent seed-7411 reopen stopped at that guard before reading result
rows. Therefore this is neither a LoRA result nor evidence that any producer receipt is invalid.

## Adversarial findings and resolutions

### 1. Editing v11 would make the completed evidence self-inconsistent

Every setup, training, checkpoint, and evaluation receipt binds the v11 source-contract digest.
Changing `analysis.py` in place would create a new source digest that cannot validly consume those
receipts without regeneration.

Resolution: leave the entire producer byte-identical. Load it under an isolated package name and pin
source-contract version/digest, reviewed implementation digest, `analysis.py`, config file, resolved
config, model, revision, backend, experiment root, and repository root before installing a seam.

### 2. General `resolve()` would weaken a deliberate anti-alias boundary

A permissive replacement could admit unrelated `..`, symlink, traversal, or outside-repository
paths and detach analysis from canonical receipts.

Resolution: recognize only the exact raw registered external prefix. Descendant suffixes may contain
no empty, `.`, `..`, NUL, or backslash component. Normalize once, prove containment under both the
repository and exact canonical external prefix, then call v11's unchanged `canonical_repo_path`.
Every nonmatching input delegates to the original helper. Tests require unrelated aliases and a
registered-prefix traversal to fail.

### 3. Copying analyzer code could silently diverge

Even a mechanical copy risks changing scientific functions or later drifting from the frozen source.

Resolution: import and call the original `analyze_phase` function. No summarizer is copied. The exact
analysis file hash is pinned, and the original source snapshot remains held during analysis. The one
runtime replacement is restored in `finally` and rejected if any other writer changes it.

### 4. A recovery output at a new location would break branch lineage

Later producer stages accept only canonical analysis paths such as
`analysis/lora_joint_trigger.json`. A receipt under this successor cannot authorize Stage B.

Resolution: the exact v11 analyzer writes the original canonical output. This experiment writes a
separate sidecar binding output path, byte hash, receipt identity, source/config identities, frozen
smoke, and seam. This is an operational repair of the original analysis, not a scientific follow-up
result appended to it.

### 5. A crash after producer output but before the sidecar needs an auditable resume

Blindly adopting any pre-existing output would make provenance ambiguous; refusing it would strand a
valid no-overwrite v11 receipt.

Resolution: publish an immutable source-bound `RECOVERY_ANALYSIS_STARTED` receipt before analysis.
On resume, an existing output is accepted only under that attempt and only after strict JSON,
self-identity, phase, and complete v11 identity validation; the sidecar records the resume.

### 6. Smoke itself could peek at the partial LoRA outcome

Seed 7411 evaluation exists, so a convenience loader smoke could expose its scientific values before
the recovery design is frozen and before the three-seed barrier.

Resolution: smoke tests paths and source identities only. It does not call `_load_evaluation`, parse
an evaluation summary, open result JSONL, touch `benchmarks/`, or decompress sealed contrast. Its
receipt states zero result, benchmark, and sealed-row access. The actual analysis command remains
procedurally delayed until seeds 7412 and 7413 are evaluated.

### 7. Mutable recovery code could change after its smoke

Git provenance alone does not stop a local edit between smoke and analysis.

Resolution: the smoke hashes the config, adversarial review, runner, recovery module, package marker,
and tests. Every analysis phase recomputes and requires that exact recovery source contract before it
creates a STARTED receipt or opens producer evidence.

### 8. The wrapper could claim a different scientific identity

Changing or enriching the original receipt would no longer be the exact preregistered analyzer and
might break downstream validators.

Resolution: the producer receipt is byte-for-byte what v11 `analyze_phase` emits and retains its
v11 identity. Recovery provenance lives only in the successor sidecar. The wrapper exposes the
producer status/verdict/next stage but defines no new classification.

## Required controls

The recovery is authorized only while all of these pass:

1. The unmodified helper reproduces the exact lexical-canonical rejection.
2. Both helpers agree on the canonical external prefix.
3. The seam maps the registered raw prefix and an existing clean descendant to that same canonical
   tree.
4. Unrelated lexical aliases and traversal descendants remain rejected.
5. The original function object is restored after the context.
6. All producer identities equal the frozen v11 pins.
7. Smoke records zero result rows, benchmark paths, sealed rows, and scientific analysis calls.
8. The complete test suite and repository `make check` pass before publication.

## Residual limitations

- The sidecar proves which recovery wrapper invoked the exact v11 analyzer; the original receipt
  intentionally names v11 because its scientific logic is v11.
- This does not retroactively prove seed 7411's outcome, establish LoRA formation, or authorize full
  rank. Only the complete original analysis does.
- The prefix seam is specific to this producer and must not become a general repository utility.

## Final authorization

`RECOVERY_GO` applies to the non-result smoke now. Scientific analysis remains `NO_GO` until all
three preregistered LoRA trigger evaluation directories are complete and published. Any source pin
mismatch, failed scope control, or second analyzer defect retracts authorization and requires a new
review; it cannot be relabeled as a LoRA miss.

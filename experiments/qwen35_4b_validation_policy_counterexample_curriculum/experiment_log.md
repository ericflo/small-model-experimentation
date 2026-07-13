# Experiment log

## 2026-07-13 — intake and scaffold

- Attached the experiment to `agentic_breadth_installation`; direct predecessor
  is `qwen35_4b_transaction_invariant_recovery_curriculum`.
- Copied the predecessor's local looping harness, merged-checkpoint trainer,
  vLLM runner, locality audit, and conditional transition bank into this
  self-contained directory.
- Pinned the learned transaction parent (`1cf5fb...41ba3`), C54 apex anchor
  (`c93316...608d5`), prior primary bank (`9c196d...9315`), and prior bank
  receipt (`8c2c33...e63e`).

## 2026-07-13 — adversarial preflight repairs

- The first task generator had only five content variants and nominally fresh
  splits overlapped. Replaced it before preregistration with high-entropy
  procedural values/resources and three distinct public data representations.
- Made the reused atomic-reservation sentinel high-entropy and changed its
  partial state to the exact near-correct residual: copy, whole-request check,
  atomic update, and `False` rejection are already correct; only negative
  handling is absent.
- The copied locality builder reproduced the predecessor's content. Replaced
  every stem and prefix; all 48 new hashes are disjoint from prior locality
  blocks.
- Narrowed the treatment from full seven-row policy-task blocks to a single
  replaced `diagnosis_to_changed_patch` row in each of 24 otherwise frozen
  predecessor blocks. This removes direct whole-solution reteaching while
  retaining every conditional transition through 312 unchanged rows.
- Added public-content digests excluding task ID/split, cross-process manifest
  stability, within/cross-block uniqueness, sentinel separation, verification
  and commit gates, frozen Menagerie seeds, aggregate-only benchmark storage,
  and an immutable design-file receipt.

## 2026-07-13 — CPU preflight

- Unit suite: 21/21 passed.
- Harness smoke: passed across ten policy families; initial/partial fail,
  oracle pass, firewall clean, locality fresh.
- Full deterministic banks built: candidate/control 336 rows each; 24 injected
  candidate revisions; all transition and operator counts matched; 38,248
  weighted action tokens per operator per epoch; no overlength row.
- Bank SHA-256: candidate `940da9...305a`; control `524240...c10e`; receipt
  `45ce6e...d6e8`.
- No model output existed during any of these corrections or checks.

## Next recorded event

## 2026-07-13 — immutable design boundary

- Rebased the design onto current upstream `main`, resolved generated catalog
  conflicts by regeneration, and pushed directly to `main` at `e0b19f5d`.
- Wrote `runs/preregistration_receipt.json` over 17 design-critical files. The
  receipt records `model_output_precedes_lock: false`; every GPU/model mode now
  fails closed on file-digest or ancestry drift.

## Next recorded event

Commit and push the immutable receipt, then run GPU smoke. Only a passing smoke
authorizes the full staged pipeline.

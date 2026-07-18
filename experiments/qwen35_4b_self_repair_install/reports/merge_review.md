# Merge Review

Scoped to merging the self-repair adapter into an explicit composite.

- Adapter trained clean on the authenticated base_reserialized composite
  (504 rows, loss 1.151, ~7 min, adapter_complete). Published hashes
  filled into PUBLISHED_ARM_HASHES from the committed training receipt
  (adapter_config 70236058…, adapter_weights fe84c983…, log d6b469d2…,
  receipt 90b33c8e…).
- The merge uses the vendored merger (merge_adapter.py, sha cb9af8b4…)
  with --base-model at base_reserialized (tree 26d8ee48…, weights
  b654e033…); no runtime-LoRA path; the merger refuses without receipts.
  Output: large_artifacts/qwen35_4b_self_repair_install/merged/self_repair.

**Verdict:** `PASS_CONTROL_MERGE`.

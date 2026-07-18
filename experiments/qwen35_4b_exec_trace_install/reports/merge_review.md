# Merge Review

Scoped to merging the execution-trace adapter into an explicit composite.

- The adapter trained clean on the authenticated base_reserialized
  composite (400 rows, loss 0.696, ~6 min, adapter_complete). Published
  hashes filled into PUBLISHED_ARM_HASHES from the committed training
  receipt (adapter_config f816a44a…, adapter_weights 09353583…, log
  96d83112…, receipt 35bfa1c1…).
- The merge uses the vendored merger (merge_adapter.py, sha cb9af8b4…)
  with --base-model at the base_reserialized composite (tree 26d8ee48…,
  weights b654e033…); no runtime-LoRA path; the merger refuses without
  receipts. Output: large_artifacts/qwen35_4b_exec_trace_install/merged/
  exec_trace.

**Verdict:** `PASS_CONTROL_MERGE`.

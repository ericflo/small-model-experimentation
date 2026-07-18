# Merge Review

Scoped to merging the repair_why_stack adapter into an explicit composite.

- Adapter trained clean on base_reserialized at the 4-epoch recipe (1008
  rows, converged final-step loss ~0.02; adapter_complete). Published
  hashes filled into PUBLISHED_ARM_HASHES (adapter_config 1de29d76…,
  weights 9b009581…, log 4b2726ee…, receipt 8e9fc0e2…).
- The merge uses the vendored merger (cb9af8b4…) with --base-model at
  base_reserialized (tree 26d8ee48…, weights b654e033…); no runtime-LoRA
  path; refuses without receipts. Output:
  large_artifacts/qwen35_4b_repair_why_stack/merged/repair_why_stack.

**Verdict:** `PASS_CONTROL_MERGE`.

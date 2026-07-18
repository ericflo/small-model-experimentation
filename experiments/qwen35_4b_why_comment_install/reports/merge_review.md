# Merge Review

Scoped to merging the why_comment adapter into an explicit composite.

- Adapter trained clean on base_reserialized at the fixed 4-epoch recipe
  (504 rows, converged: final-step loss ~0.05, avg 1.528 pulled up by the
  ~10.3 start; adapter_complete). The epoch-1 recipe was corrected to 4
  epochs pre-measurement (undertraining, recorded). Published hashes
  filled into PUBLISHED_ARM_HASHES (adapter_config 3f08132f…, weights
  c513de04…, log bebf146b…, receipt 3645f21b…).
- The merge uses the vendored merger (cb9af8b4…) with --base-model at
  base_reserialized (tree 26d8ee48…, weights b654e033…); no runtime-LoRA
  path; refuses without receipts. Output:
  large_artifacts/qwen35_4b_why_comment_install/merged/why_comment.

**Verdict:** `PASS_CONTROL_MERGE`.

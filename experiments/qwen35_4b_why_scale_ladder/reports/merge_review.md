# Merge Review

Scoped to merging each trained scale-ladder rung adapter into an explicit
composite (identical merger for every rung; the rung's published hashes
are pinned into PUBLISHED_RUNG_HASHES from its committed training receipt
before the merge).

- Each rung adapter trained clean on the authenticated base_reserialized
  composite from its diverse WHY corpus (sha-pinned in ladder_manifest.json).
- The merge uses the vendored merger (merge_adapter.py, sha cb9af8b4...)
  with --base-model at base_reserialized (tree 26d8ee48..., weights
  b654e033...); no runtime-LoRA path; refuses without receipts. Output:
  large_artifacts/qwen35_4b_why_scale_ladder/merged/why_scale_<rows>.

**Verdict:** `PASS_CONTROL_MERGE`.

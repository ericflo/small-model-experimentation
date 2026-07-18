# Merge Review
Each trained rung adapter merges into an explicit composite via the vendored
merger (cb9af8b4...) with --base-model base_reserialized (tree 26d8ee48...,
weights b654e033...); published hashes pinned per rung from its committed
training receipt; no runtime-LoRA path; refuses without receipts.
**Verdict:** `PASS_CONTROL_MERGE`.

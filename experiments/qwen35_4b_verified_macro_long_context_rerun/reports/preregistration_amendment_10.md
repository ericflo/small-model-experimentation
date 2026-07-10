# Preregistration amendment 10: external scientific-smoke artifact durability

Date: 2026-07-10. Frozen after the complete base-smoke think@32,768 vLLM call returned its
runner-native rows atomically, but before those rows were classified under amendments 7 and 9 and
before any decoded output, parser result, task score, correctness field, or oracle field was
inspected. No think@49,152 generation had begun. This amendment changes artifact location,
transactionality, and validation only; it does not change generation or any scientific decision.

## Motivation

Scientific-smoke rows are much larger than the train-only interface rows. The completed 16,384 base
arm alone is 30,753,957 bytes. Linear projection from that artifact puts a single maximally used
61,440-token arm above GitHub's 100 MiB hard file limit, even before a physical selected-tier copy.
The repository validator scans untracked files too, so merely declining to stage a large JSONL does
not make an in-repository scientific cache safe. The existing promotion helper would also duplicate
the selected raw tier.

Moving only JSONL is not valid: cache validation binds the runner metadata and prompt preflight as
well. The scientific cache therefore needs one canonical external namespace whose integrity and
selection remain auditable from small tracked artifacts.

## Frozen storage protocol

1. Train-only calibration and heldout-interface artifacts remain under the experiment's tracked
   `runs/` tree. Scientific matrix arms and non-scored termination probes are canonical under the
   default root
   `/workspace/large_artifacts/qwen35_4b_verified_macro_long_context_rerun/scientific_smoke_v1/`.
   `QWEN35_MACRO_SCIENTIFIC_ARTIFACT_ROOT` may point to a byte-identical copy on another host. The
   resolved root must be absolute, path-contained, and free of symlink components.
2. Preserve the existing flat runner names under `smoke_tiers/think_<budget>/` or
   `smoke_budget_probes/think_<budget>/`: `ARM.preflight.json`, `ARM.jsonl`, and `ARM.meta.json`.
   A completed bundle adds `ARM.receipt.json` as its last-written atomic commit marker.
3. A preflight by itself is the only valid incomplete state. Rows or metadata without the other
   members and a valid receipt fail closed and require explicit quarantine; they are never silently
   resumed, promoted, scored, or overwritten.
4. Each receipt binds byte size and SHA-256 for the preflight, rows, and metadata; ordered record
   ids; input-record and rendered-prompt hashes; prompt-token counts; task order; K; arm; role;
   budget; complete-matrix versus probe mode; model id and revision; experiment-local runner hash;
   and exact sampling and engine identities. Receipt validation also checks JSONL row order, prompt
   identity, contiguous sample order, request count, and completion count without parsing or grading
   model answers.
5. The tracked `analysis/scientific_smoke_artifact_catalog.json` is a deterministic inventory of
   every receipt and permitted preflight-only state. It records only external-root-relative paths,
   per-file bytes and hashes, and a tree hash defined as SHA-256 over sorted
   `relative-path NUL file-sha256 newline` records. Unknown files, traversal, missing files,
   corruption, symlinks, partial bundles, and receipt/catalog drift all fail closed. Its protocol
   binding hashes the exact config, tasks, demonstrations, analyzer, run orchestration, domain,
   harness, storage validator, and vLLM runner sources. It binds the `base` and
   `designed_ceiling` library payloads by content rather than hashing the whole library container,
   so later train-only Qwen-library additions cannot retroactively invalidate frozen smoke bytes.
6. When a complete K=12 tier passes, the catalog stores a logical arm-to-entry pointer plus the
   byte size and SHA-256 of `analysis/smoke_budget_selection.json`. The dependency is deliberately
   one-way: the catalog hashes the selection file, while the selection file does not hash the
   catalog. No `runs/smoke/` raw copy or other physical promotion is created. A termination probe
   can never be a selected entry.

## Migration and resume boundary

The already-returned 32,768 rows remain valid because storage path and orchestration-source changes
are not part of the frozen vLLM sample identity. Migration must occur only after the old process has
exited. It copies the local scientific tree into a staging root, excludes the temporary higher-rung
guard, validates every complete triplet against its existing prompt/sampling/engine/runner identity,
writes receipts, builds and independently verifies the catalog and tree hash, then atomically
installs the external root. Local scientific files may be removed only after the external copy
passes the same exact cache validation. If an old process made a selected-tier copy, it must first
be proven byte-identical to its tier source and then discarded rather than cataloged.

Preflight-only files retain their exact bytes and remain resumable. A current-code smoke restart
must reuse exact-valid external rows while recomputing termination metadata under amendments 7 and
9. Artifact migration cannot authorize a higher rung, alter the ladder, or reinterpret an
inadequate row.

## Unchanged scientific boundary

No model, revision, vLLM runner, prompt, task, library, arm, sample count, seed, prompt order, batch
shape, context limit, thinking allowance, answer allowance, loop detector, termination threshold,
selector, hidden grader, matched-compute baseline, or confirmatory criterion changes. The
train-only proposal and full-run sharding protocols are unchanged. Missing external artifacts are
an operational/reproducibility failure, never evidence for or against verified macro invention.

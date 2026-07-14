# Adversarial Design Review

## Verdict

`GO` only for the exact-prefix branch-execution wrapper and failure archival transaction described
here. No producer source edit or direct invocation is allowed after the frozen smoke.

## Threats reviewed

1. **A broad path normalization could legalize traversal or aliases.** The seam recognizes only the
   producer's exact raw `../../large_artifacts/qwen35_4b_state_formation_capacity_adjudication`
   prefix. Every suffix component must be nonempty and not `.` or `..`; the normalized result must
   remain under the canonical registered prefix, then passes v11's original repository/no-symlink
   validator. Every nonmatching path delegates to the original helper.
2. **A wrapper could bypass the producer CLI's canonical-path or source-snapshot gates.** It imports
   and calls the exact pinned producer `scripts/run.py`; that CLI still validates every argument,
   canonical output/input, phase authorization, loaded source contract, lock, and reviewed source
   snapshot. The wrapper changes only the analyzer module's one path helper.
3. **A second module copy could leave the real validator unpatched.** The wrapper rejects any prior
   foreign `src` binding, imports the producer CLI first, and patches the exact `src.analysis` module
   used by `src.gate_receipts` and `src.gpu_runner`. Smoke calls the real downstream
   `_authorization_for` path.
4. **The wrapper could silently broaden Stage B.** It accepts only model-smoke, positive-control,
   train, and evaluate-state; requires an authorization receipt; and forwards the exact arguments to
   v11, whose purpose-specific policy accepts only the canonical registered receipt.
5. **The first failure could be erased to enable retry.** Archive requires byte-identical,
   inode-distinct canonical/mirror receipts with the pinned SHA and self-identity. Retirement requires
   a Git commit containing the archive copy, archive receipt, and both producer originals, then
   writes a STARTED marker before verified unlink and a terminal retirement receipt afterward.
6. **Crash windows could cause duplicate GPU work.** Every wrapper invocation publishes an immutable
   STARTED receipt before calling v11. If the producer output exists after a wrapper crash, rerun
   validates and adopts it into a COMPLETE receipt rather than launching a duplicate. Producer v11's
   own run lock and attempt recovery remain authoritative.
7. **Recovery could change scientific interpretation.** It defines no metric or threshold and never
   edits producer receipts. The immutable producer result remains the sole evidence and authorization
   source. Recovery outputs are operational lineage only.
8. **Smoke could contaminate sealed contrasts.** It revalidates only the already-open trigger-side
   LoRA-miss receipt. It loads no model, opens no benchmark path, and does not request Stage-B or
   contrast authorization.
9. **The original visibility deviation could motivate adaptation.** All recovery behavior is fixed
   from path/error/receipt identities. It makes no choice based on per-cell values and cannot alter
   seed, checkpoint, objective, or the mandatory Stage-B matrix.

## Required pre-retry evidence

- Focused tests pass.
- Recovery smoke returns `BRANCH_RECOVERY_SMOKE_PASS` under the frozen recovery source contract.
- The exact failed G0 pair and its recovery archive are committed and both workflows are green.
- A retirement receipt removes only the two producer failure paths; that retirement is committed and
  both workflows are green.
- The producer analysis receipt, producer source, and recovery smoke rehash immediately before retry.

## Stop conditions

- Any source/config/receipt/hash mismatch.
- Any alias/traversal control failure or helper-restoration failure.
- Any model load during smoke/archive/retirement.
- Any failure to prove the archive commit contains all required bytes.
- Any producer CLI request outside its registered canonical policy.

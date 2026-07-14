# Adversarial Design Review

## Verdict

`GO` only for the byte/status-aware downstream handoff defined here, after its focused tests and
no-model smoke pass and the complete checkpoint is pushed with both workflows green.

## Defect reproduced

The first recovery's frozen `invoke_producer` rejects whenever either former failure pathname
exists. That was correct immediately after archival but becomes overbroad after a successful retry:
producer v11 must write the valid G0 to the same canonical setup pathname. The observed positive-
control attempt stopped on this guard before STARTED publication, producer import, model load, or
scientific work.

## Threats reviewed

1. **A handoff could erase or relabel the original failure.** The exact failure bytes remain in the
   first recovery archive at pinned SHA-256. The terminal retirement receipt is rehashed and
   identity-checked on every invocation. The retired mirror must remain absent, and exact failure
   bytes at the canonical G0 slot are explicitly rejected.
2. **Any nonfailure file could be mistaken for success.** The canonical occupant must match the
   exact successful G0 file SHA-256 and receipt identity; status, phase, capacity, seed, producer
   source, model identity, positive-control authorization, no-training authorization, no benchmark,
   no contrast, and no training/evaluation fields are also checked.
3. **A detached G0 could be spliced into place.** The exact first-recovery STARTED and COMPLETE
   files are pinned by file hash and receipt identity. Completion must bind the successful G0 hash
   and identity and the exact STARTED hash.
4. **Fixing the first recovery in place would rewrite history.** This is a separate experiment. It
   imports the exact frozen first-recovery source, whose file hash, source contract, and smoke are
   pinned, and leaves every file in that experiment unchanged.
5. **Copied orchestration could bypass producer policy.** Argument shapes use the first wrapper's
   exact validator. Producer context/source/config/model checks, stage-specific output paths and
   statuses, the exact producer CLI, source snapshot, run lock, authorization, setup/control, train,
   and evaluation gates remain active. Only the first wrapper's pathname-only outer guard is not
   called.
6. **The path repair could broaden.** Model stages use the first recovery's exact frozen
   `installed_path_seam`; only `src.analysis._canonical_expected_path` changes temporarily, and its
   context manager proves restoration. No new path normalization is defined here.
7. **A crash could duplicate expensive work or adopt unrelated output.** Handoff STARTED is
   immutably published before producer entry. A preexisting producer output without that STARTED
   receipt is rejected. On rerun, an existing COMPLETE must equal a fresh completion derived from
   the producer output; an existing output with STARTED but no COMPLETE is validated and adopted
   without rerunning the producer.
8. **The new wrapper could expand the experimental matrix.** Allowed stages and capacity/objective/
   eval-set combinations are inherited from the first recovery's frozen validator. The producer's
   receipt-specific authorization is still decisive.
9. **Smoke could create model or hidden-label evidence.** Smoke only validates hashes, identities,
   statuses, safe path occupancy, and reproduces the first wrapper's outer rejection. That rejection
   occurs before producer context/model work. Smoke records zero model, training/evaluation,
   benchmark, contrast, or interpretation activity.
10. **The prior operator-blinding deviation could influence repair.** All handoff behavior is fixed
    from paths, exact receipt bytes, and already-mandated branch structure. No per-cell result or
    sealed metric informs the repair.

## Required pre-use evidence

- Focused tests cover exact lineage, failed-byte rejection, changed-G0 rejection, mirror rejection,
  the original false rejection, and inherited invocation-shape rejection.
- Smoke returns `BRANCH_HANDOFF_RECOVERY_SMOKE_PASS` under its frozen source contract.
- The successful G0 and both first-recovery invocation receipts are committed.
- This entire checkpoint passes Validate Repository and Publish Research Site on `main`.

## Stop conditions

- Any producer, authorization, first-recovery, retirement, failure archive, G0, or handoff hash/
  identity mismatch.
- Reappearance of the retired mirror or failed bytes at the canonical slot.
- Any output predating its handoff STARTED receipt.
- Any request outside the inherited registered stage matrix.
- Any model load, benchmark access, contrast access, or scientific work during smoke.

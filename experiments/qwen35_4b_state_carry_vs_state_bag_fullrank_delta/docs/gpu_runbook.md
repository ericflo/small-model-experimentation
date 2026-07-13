# GPU Runbook

No GPU stage has been run in the implementation handoff; proceed through the
gates below. Use the repository compute-environment recovery rules,
one exclusive GPU process, and the pinned root environment.

1. Run CPU tests and `--stage cpu-smoke`.
2. Run `--stage prepare-data`; inspect `manifest.json` for
   `PARENT_DATA_PARITY_PASS`. Parent artifacts need not be imported; when present
   they are compared directly in addition to the frozen row hashes.
3. Confirm the parent terminal analysis receipt remains the valid, complete LoRA
   deep-state miss. Never edit or overwrite it.
4. Run `--stage model-smoke --trigger-receipt <parent-summary>`. Treat it as
   setup-only. Inspect exact targets/parameters, both K=1 counts, K=4/K=12 counts,
   gradient groups, Adam state bytes, reserved peak/headroom, post-step parity,
   and checkpoint recurrent-logit error.
5. If G0 OOMs or fails mechanics, stop and preserve the receipt/log. Reserved
   headroom is recorded as a diagnostic, with no post-hoc threshold. Do not
   change precision, target set, optimizer, model, or parameterization here.
6. If G0 passes, train/evaluate pilot Carry and Bag at seed 7401 using distinct
   output directories. Training consumes the exact G0 receipt and embeds its
   path, artifact hash, identity hash, status, and phase in every checkpoint;
   checkpoint loading and analysis validate that lineage. Analyze the complete
   paired pilot.
7. Only `PILOT_PROMOTION_READY` licenses fixed-final seeds 7411–7413. A valid
   `PILOT_STATE_FORMATION_MISS` is terminal and directly answers the held-fixed
   LoRA-rank concern. `PILOT_INCOMPLETE` and `PILOT_PROMOTION_BLOCKED` also stop
   the run but do not support that capacity conclusion.
8. Full training likewise requires the exact G1 `PILOT_PROMOTION_READY` receipt
   and embeds both G0 and G1 lineage in every checkpoint.
9. After complete full Carry/Bag evaluation, run the same-checkpoint edge cut and
   swaps through the normal evaluator/analyzer path. Pass an explicit unique
   `--output` for the Carry-checkpoint/Bag-mode edge cut because the ordinary
   Bag evaluation owns the default `full_bag_seed*` path. Never select
   intermediate checkpoints.
10. Stop after G3. There is no sample-more CLI stage or deployment verdict here.

All commands are intentionally non-monolithic. The CLI and receipts enforce the
phase boundaries; consult `scripts/run.py --help` for paths and arguments.

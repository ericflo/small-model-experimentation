# Local Gate Adversarial Review

## Merge authorization

- Both arms trained clean under the frozen recipe (1,520 rows, zero
  encoder skips, 190 updates each; control loss 0.4334, candidate 0.54 —
  never capability evidence) with receipts bound to the exposure receipt
  and published pins filled fail-closed in train_trial.py.
- The merge uses only the pinned external merger (cb9af8b4…) with
  `--base-model` pointing at the authenticated hygiene_explore composite
  (tree recomputed against 9eb653d7… before merging); merge receipts pin
  adapter, base, merger, and output tree hashes; no runtime-LoRA path
  exists anywhere in the cell.

**Verdict:** `PASS_CONTROL_MERGE`.

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

## Local event authorization

- Merge receipts published for both arms (replay_ctl tree 8de87333…,
  feedloop_state tree 1dfa2ec1…) and self-pinned; the eval's trained-tree
  pins are filled fail-closed and the boundary re-authenticates parent and
  both merges before each engine run.
- The event: three arms × four frozen oracle-free inputs (axis holdout
  88,026; retention screens 88,027/88,028/88,030) in sequential
  authenticated engine runs; promotion logic and pooled_k3 bands are the
  frozen preregistration's, unit-pinned; no seed can be opened by this
  event.

**Verdict:** `PASS_LOCAL_EVENT`.

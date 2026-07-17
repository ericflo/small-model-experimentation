# Local Gate Adversarial Review

Backed by the three-lens adversarial workflow recorded in
experiment_log.md (zero MAJORs at design commit ffc78ec2; training
receipt and published pins committed this stage).

## Merge authorization

- The state-tracking adapter trained clean on the authenticated
  count_walk parent (160 rows, zero skips, loss 0.7637 — a clean single
  corpus loss level, never capability evidence). Published pins filled
  fail-closed and canonicalized by the normalized-hash pin (check_design
  --check verified unchanged after the fill).
- The merge uses the cell's own copied merger (cb9af8b4…) with
  --base-model at the count_walk composite (tree d5fdc55c…, weights
  ddd7bc4b… — full model.safetensors hash checked pre-merge); no
  runtime-LoRA path exists; the merger refuses without receipts.

**Verdict:** `PASS_CONTROL_MERGE`.

## Local event authorization

- Merge receipt published and self-pinned; the eval's trained-tree pin
  is filled fail-closed and the boundary re-authenticates the count_walk
  parent and the state_track merge before each run.
- The event: two arms (count_walk parent vs state_track candidate)
  across three pooled retention screens at fresh seeds 88063/88064/88065
  in sequential authenticated engine runs, plus the write-ahead
  local_events.jsonl ledger (opened before the first engine event;
  torn/discarded attempts refuse). Promotion logic is the frozen
  two-sided pooled_k3 band (correct ±15, parsed ±9, cap ±9 on integer
  screen sums vs parent); no axis kind exists at this stage; no seed can
  be opened by this event.

**Verdict:** `PASS_LOCAL_EVENT`.

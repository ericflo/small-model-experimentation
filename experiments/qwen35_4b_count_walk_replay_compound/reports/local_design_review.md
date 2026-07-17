# Local Gate Adversarial Review

Backed by the three-lens adversarial workflow recorded in
experiment_log.md (both MAJORs fixed and re-verified pre-freeze at
e61915e2; training receipt and published pins committed at 09089fb5).

## Merge authorization

- The stage-8 adapter trained clean on the authenticated count_walk
  parent (2,240 rows, zero skips, loss 1.209 — the clean chain's known
  loss-level property, never capability evidence). Published pins filled
  fail-closed and canonicalized by the normalized-hash pin (check_design
  --check verified unchanged after the fill).
- The merge uses the cell's own copied merger (cb9af8b4…) with
  --base-model at the count_walk composite (tree d5fdc55c…, weights
  ddd7bc4b… — full model.safetensors hash checked pre-merge per the
  review-added preflight); no runtime-LoRA path exists; the merger
  refuses without receipts.

**Verdict:** `PASS_CONTROL_MERGE`.

## Local event authorization

- Merge receipt published and self-pinned; the eval's trained-tree pin
  is filled fail-closed and the boundary re-authenticates the count_walk
  parent and the replay_compound merge before each run.
- The event: two arms (count_walk parent vs replay_compound candidate)
  across three pooled retention screens at fresh seeds 88060/88061/88062
  in sequential authenticated engine runs, plus a write-ahead
  local_events.jsonl ledger (opened before the first engine event;
  torn/discarded attempts refuse per the review-added guard). Promotion
  logic is the frozen two-sided pooled_k3 band (correct ±15, parsed ±9,
  cap ±9 on integer screen sums vs parent); no axis kind exists at this
  stage, so there is no fidelity readout; no seed can be opened by this
  event.

**Verdict:** `PASS_LOCAL_EVENT`.

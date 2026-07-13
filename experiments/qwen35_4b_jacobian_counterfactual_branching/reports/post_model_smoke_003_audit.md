# Post-Model-Smoke 003 Audit: Iterative Repair Near Miss

## Verdict

`LATTICE_REPAIR_REQUIRED`. Outcome-blind iterative geometry repair improves 51
of 60 rows to full validity but does not authorize mechanics.

## Results

- Layers 5, 6, and 7 pass 12/12 live non-J controls.
- Layer 4 passes 12/12 according to the repairer's per-row snapshot, but the
  independent recomputation finds maximum norm error `1.0338e-5`, just outside
  `1e-5`; this rounding-boundary discrepancy must be resolved by exact receipt.
- Layer 8 passes 3/12; maximum norm error is `1.3206e-5` and span projection
  `0.02096`.
- Iteration use by layer is 7, 370, 120, 168, and the full 512.
- All hooks, model/lens/prompt, finiteness, and firewalls remain valid. No branch
  probability, choice, target-selection, correct alias, or outcome was stored.

## Repair

The independent transport replication solved the same discontinuous bf16 tail
with exact pairs of one-ULP coordinate moves. Add its geometry-only pair search
after iterative correction for failing rows, bounded to 32 pairs per row. The
objective remains exactly max(norm-error/1e-5, span-fraction/0.01). No logits or
labels enter. Preserve this near-miss receipt and require another pushed hash
boundary before smoke 004.

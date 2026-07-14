# Adversarial Design Review

## Review 1 — 2026-07-14

**Reviewed commit:** `3eae868d182f4a02848f6415d8eaafdb87465336`

**Verdict:** **HOLD**

**Access:** zero tokenizer/model/GPU/benchmark/hidden/protected events; no review edits

The independent adversary reproduced the committed smoke and then tested the proposed
geometry. It found the original corpus impossible: the string family had only 122
eligible signatures for 192 required rows, and an uncaught state-explosion exception
terminated full construction. It also exhaustively checked the visible examples and
found that 14/30 smoke tasks admitted two to nine depth-three plan spellings.

Additional blockers were:

1. correct-versus-shuffled isolates task-specific auxiliary plan supervision, not a
   reflection-specific effect; a matched non-reflective plan-label arm is required;
2. the fixed `READY` seam is controlled branch transfer, not a literal interrupted
   action state, so claims must be scoped accordingly;
3. direct-control, retention, literal-reflection, and candidate-compute contracts were
   missing or contradictory;
4. tokenizer rendering, exact loss masks, LoRA/optimizer/batching parity, target-token
   accounting, and immutable receipts were not frozen;
5. qualification/confirmation power and seed-independent gates were not executable;
6. the conditional J stage lacked independent fit/confirmation/causal data and fixed
   readout/intervention controls, and therefore must not share behavioral evidence.

### Repair checkpoint status

The following defects are repaired in the working design and covered by executable CPU
tests:

- a finite exact-depth catalog is enumerated before allocation;
- globally behavior-equivalent and shallower-equivalent programs are excluded;
- seven visible demonstrations uniquely identify the exact ordered plan;
- full 216/144/144 construction succeeds with zero program or behavior collisions and
  complete per-split operation-position coverage;
- state explosions are rejected, not leaked;
- shuffled donor labels occupy separate supervision fields, preserve task truth, and
  are behaviorally wrong for the recipient;
- validation uses fail-closed exceptions rather than optimization-removable asserts;
- the construction entry point consumes the committed config and installs a Python
  audit-hook benchmark read firewall.

The verdict remains HOLD. No tokenizer/model/GPU/training/Jacobian stage may run until
the remaining design contracts are implemented, committed, and independently reviewed.

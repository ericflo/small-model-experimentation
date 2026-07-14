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
- full 216/72/144/144 construction succeeds with zero program or behavior collisions
  and complete per-split operation-position coverage;
- state explosions are rejected, not leaked;
- shuffled donor labels occupy separate supervision fields, preserve task truth, and
  are behaviorally wrong for the recipient;
- validation uses fail-closed exceptions rather than optimization-removable asserts;
- the construction entry point consumes the committed config and installs a Python
  audit-hook benchmark read firewall.

The verdict remains HOLD. No tokenizer/model/GPU/training/Jacobian stage may run until
the remaining design contracts are implemented, committed, and independently reviewed.

### Post-review implementation awaiting Review 2

The working repair adds the previously missing non-reflective correct-label arm,
real retention tasks, exact record and mask code, a frozen QLoRA recipe, within-step
token parity, executable paired gates, explicit literal-reflection accounting, and
two-seed confirmation rules. The J stage has been moved to a required separate future
experiment. These changes are not self-authorizing; Review 2 must inspect their exact
committed revision.

## Review 2 — 2026-07-14

**Reviewed commit:** `1cb3c351b1ca14b518abe7cbff02ac67e6134726`

**Verdict:** **HOLD** for training/evaluation; tokenizer implementation statically
safe, but a separate clean authorization signature is required.

The adversary reproduced full 216/72/144/144 plus 48 retention construction, verified
all 36 optimizer groups and 216 shuffled donors, passed the focused model-free tests,
and found the rendering/mask/trainer logic coherent and fail-closed. It confirmed the
J follow-up is genuinely result-separated.

It then demonstrated that the decision functions could pass a one-task, one-family,
one-depth artifact because they did not require the sealed task-ID sets. It also found
that generation metadata, merged trees, stage ancestry, retention promotion,
literal-reflection execution, adapter ON/OFF behavior, and adapter-tree lineage were
not mechanically enforced. These are blocking provenance defects even though the
statistical design itself is now specified.

During review, one overly broad repository search surfaced unrelated tracked run-log
lines containing hidden-output fields despite the benchmark exclusion. The reviewer
made zero tokenizer/model/GPU/benchmark calls and no edits, but this access mistake
means its tokenizer-safe assessment is not accepted as the required clean independent
authorization.

### Review 2 remediation implemented, pending independent review

- exact sealed task IDs, all three families, and both retention depths are now required;
- scoring binds input/label receipts, vLLM metadata, model/revision, runner, environment
  lock, clean Git state, sampling, seed, engine geometry, merged checkpoint, arm, and
  adapter seed;
- merge and vLLM load bind full artifact trees and exact source arm/seed lineage;
- screen, replication, confirmation, and final promotion use hash-bound stage receipts;
- retention has an executable promotion gate;
- literal reflection now has an exact branch constructor and token-matched base-prefix
  scorer;
- every merged adapter must pass a greedy token/logprob ON/OFF gate before scoring.
- runner metadata now hashes the generated JSONL itself, and every scoring path checks
  that binding before accepting rows.

The repaired model-free suite passes 45 focused tests and reconstructs all 624 sealed
depth-three plus retention tasks with no collision. No authorization changes occur
until these repairs are committed and a fresh clean reviewer returns its verdict.

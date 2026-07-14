# Adversarial design review

**State:** `PASS_DESIGN_FOR_MODEL_FREE_CONSTRUCTION_ONLY`

**Date:** 2026-07-14

**Model/GPU access:** none

**Parent sampled-output access:** none

**Hidden/gold/benchmark access:** none

This recovery successor is authorized for model-free construction only. Model,
tokenizer, GPU, calibration, mechanics generation, and hidden access remain
sealed.

## Known adversarial risks

1. A renamed parent task is not fresh; function fingerprints and rendered
   prompt-token sequences must be disjoint.
2. The parent returned 4,056 outputs. Even descriptive inspection could tune
   the successor, so raw bundles require an enforced read prohibition.
3. Allowing descendants during replay can accidentally weaken the initial
   generation gate. The two invariants require distinct typed APIs.
4. Authenticating transport before the descendant chain could accept foreign
   or gapped history. Full-chain authentication must precede replay.
5. A mocked visible-analysis test can reproduce the parent's false assurance.
   The lifecycle test must use the real composed functions.
6. Restart paths can resample after a crash or silently accept a partial
   successor. Every durable boundary needs a no-resample recovery test.
7. Ordinary Python equality admits Boolean/integer aliases. All durable
   schemas and plans require recursive exact JSON typing.
8. A selector can leak hidden labels through task admission, resource matching,
   pool construction, or tie-breaking. All selection inputs must be visibly
   reconstructible before the key exists.
9. Taskwise matched compute can be invalidated by backend mixing, noncanonical
   token accounting, or an insufficient direct pool. These cases stop rather
   than extrapolate.
10. Reusing the parent ciphertext/key or reading any benchmark would destroy
    the fresh boundary.

## Required independent review evidence

- Feasibility of excluding every parent function fingerprint while retaining
  the frozen 48/24 balance and mechanics strata.
- Exact parent administrative allowlist and a sampled-bundle denylist proven by
  path-audit mutation tests.
- Zero intersections for all declared identity, prompt-token, and seed domains.
- An unmocked complete lifecycle that would have failed on the parent code.
- Complete inventory-state, crash/restart, exact-type, and zero-call matrices.
- Fresh ciphertext/key generation and sealed hidden authorization.
- Same-model, same-revision, same-backend, same-arm, same-threshold proof.
- Exact-commit CI and lock release sequence with no unreviewed runtime files.

## Independent review result

The independent reviewer returned
`PASS_DESIGN_FOR_MODEL_FREE_CONSTRUCTION_ONLY` on exact pushed-green commit
`035c3f8ce14064dd8ee843a43f07970e47bf40f1`. It confirmed scientific
equivalence to the parent, genuine fresh-identity intent, the parent-sample
quarantine, distinct typed temporal APIs, unmocked lifecycle requirements,
matched-compute controls, the hidden boundary, and the release sequence.

Construction must fail closed unless it proves default-deny parent access;
zero parent intersections in function fingerprints, IDs, seed keys and derived
seeds, identity-free prompts, and prompt-token sequences; exact depth-three
balance/strata; a fresh ciphertext and key with no plaintext artifact or key
reread; and zero model/GPU/protected reads.

**Verdict:** `PASS_DESIGN_FOR_MODEL_FREE_CONSTRUCTION_ONLY`.

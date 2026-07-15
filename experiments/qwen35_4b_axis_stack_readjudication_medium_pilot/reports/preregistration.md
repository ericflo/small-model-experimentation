# Preregistration: Axis Stack Re-adjudication with Medium Pilot

Frozen before any model event. A failed gate is a preserved result, never
permission to change this contract inside this directory. Both predecessor
failures remain recorded and their seeds (78,144; 78,145) remain sealed —
this trial erases nothing.

## Frozen identities

- Experiment: `qwen35_4b_axis_stack_readjudication_medium_pilot`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Arms, all inherited published explicit composites (runtime LoRA forbidden):
  - `replay_parent`: tree `4c4f3561efbcafe1b9f777f4bd21bf4949ff89177f77946d0fa0f88cafafacd7`,
    weights `3df45004fcf42519ce28cdcfedcbb39b0907662f8ecfb8a87b13b416087d0072`.
  - `replay_squared`: tree `01108a985d2179561656141b2b824ee15a1d7a8a260da5d9e83387ebdc3a777d`,
    weights `e43b885c47ecc7046c3c741b48afd53dd1cb96d2d98426b714714d5ac271069e`.
  - Candidate `axis_on_replay`: tree `77e4858fe6ddade7a8446a0c561c3c18d07c338d4dea2f0b8193693fcca264ea`,
    weights `7ebcad397c820196fb2271fe4c608a62a578465152b48e3fcee2c8d3b46fd0e4`.
- Seeds: gate 88,016; conditional aggregate 78,146 (sealed until promotion).
  No training/merge/exposure seeds exist — this trial trains nothing.

## Why re-adjudication is legitimate here

(a) The prior gate failures stand as recorded; (b) the correction mechanism was
queued in the program backlog, in exact form, before this experiment opened,
from evidence in two independent receipts (protocol tied at the parent ceiling
8/8/8 in both experiments, silently converting 3-of-4 into 3-of-3); (c) the new
gate uses a fresh task seed — no graded item is reused; (d) the corrected bar
adds a fail-closed outcome (`GATE_UNDETECTABLE`) and leaves every other
condition unchanged; (e) the adversarial design review is instructed to attack
exactly this framing.

## Frozen gate (seed 88,016)

Two instruments as in the predecessors: 40 axis-holdout tasks (10 per kind) and
104 retention tasks (8 per original skill), fresh-seed generated, oracle-free
model input, zero canonical-message overlap against both frozen corpora files,
regenerated construction rows, prior local seeds 88,000–88,015, and all three
predecessor gate files. One event, three composites, pinned engine geometry.

Corrected promotion — `axis_on_replay` promotes iff ALL hold:

1. DETECTABILITY: an axis kind is detectable iff NEITHER control scores ≥ 9 of
   10 on it. Undetectable kinds are excluded from condition 3 and reported as
   `not_detectable`. If zero kinds are detectable the outcome is
   `GATE_UNDETECTABLE` and the gate fails closed.
2. Axis total strictly greater than BOTH controls.
3. Strict kind wins (candidate > max of both controls; ties fail the kind) on
   at least ⌈2/3 × detectable kinds⌉.
4. Retention non-inferiority, unchanged from the predecessors: correct ≥ each
   control − 5; cap contacts ≤ each control + 3; parsed ≥ each control − 3.
5. Feasible-route abstentions ≤ 4 of the 8 retention route rows.

No promotion permanently seals seed 78,146.

## Frozen conditional pilot — MEDIUM tier

One aggregate-only gateway event: tier `medium`, think budget 1,024, sealed
seed 78,146, four weight-recomputed composites (`base`
`b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db`, both
controls, the candidate), clean pushed `main` with the promotion receipt
committed, one-seed ledger, identical source inventory.

Pilot gates, all required: candidate aggregate strictly greater than base,
strictly greater than `replay_squared`, and strictly greater than
`replay_parent`.

The goal gate — every public family strictly above base — is recorded from the
same event; at the medium tier it has passed 8 of 92 historical events, so a
pass is demanding but achievable, and a fail is recorded as "not confirmed at
this event", never as evidence against the mechanism. If the pilot passes, the
universal claim still requires independent seeds and a same-backend
matched-compute sample-more baseline in successors.

## Mandatory checkpoint order

1. Model-free construction + design review — committed, pushed, green.
2. `local` (requires `PASS_LOCAL_EVENT` in `reports/local_design_review.md`);
3. conditional `benchmark` only on promotion.

## Interpretation limits

Single-seed events; no claims minted. Benchmark firewall: contents never read;
only gateway aggregates and public per-family scores consumed.

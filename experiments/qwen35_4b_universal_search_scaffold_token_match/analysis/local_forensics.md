# Fresh Local Paired Forensics

Seed 88,007 is the single preregistered 26-case event. These diagnostics use only
experiment-owned procedural completions after the formal promotion decision. They
do not expose or consume benchmark data.

## Outcome

| Arm | Correct | Parsed | Caps | Execute | Induct | Probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `close_xi_parent` | 18/26 | 23/26 | 3 | 1/2 | 1/2 | 2/2 |
| `replay_after_close` | 16/26 | 23/26 | 3 | 0/2 | 1/2 | 2/2 |
| `scaffold_after_close` | 16/26 | 23/26 | 3 | 0/2 | 0/2 | 0/2 |

The candidate fails accuracy, parse, cap, execute, and induct checks. It has zero
route abstentions, the only passing check. Promotion is empty and aggregate seed
78,137 remains sealed.

## Paired redistribution

Against the immediate parent, scaffold wins one `u_trace` and one `u_repair` case,
but loses one `u_execute`, one `u_induct`, and both `u_probe` cases (2 wins, 4 losses,
20 ties). Against active replay it wins one trace, repair, and state case but loses
one induction and both probe cases (3/3/20). The aggregate tie with replay is thus a
different error distribution, not installed target behavior.

Cap contacts are also redistributed rather than removed. Parent caps on trace,
optimize, and one execute case; replay caps on optimize, trace, and verify; scaffold
caps on optimize and both execute cases. Mean generated length increases from 434.2
tokens at the parent and 471.6 at replay to 520.5 at scaffold.

## Mechanism anatomy

- On both scaffold execute failures, the visible thought computes the correct final
  sequence but continues restating/checking until the 1,024-token cap, never emitting
  a parsed answer. Compact two-operation `COMMIT` lessons did not transfer to the
  natural-language three/four-operation interface.
- On induction, the candidate is wrong on both cases. One trace claims an invalid
  decomposition fits every probe; the other treats fixed-position overwrite as
  value replacement. Canonical `SET_i_value` training did not preserve the parent’s
  natural-language operation semantics.
- On both probe failures, the candidate performs lengthy simulations but either
  collapses all predictions or invents a tie where the executable generator certifies
  a unique maximum. Parent and replay solve both. The scaffold therefore harms the
  discrimination/simulation interface needed to choose search branches.
- No distinctive scaffold codes (`LEDGER`, `FIT_SECOND`, `REJECT_FIRST`,
  `APPLY_FIRST`, `EXECUTE_PAIR`, `ADVANCE_`, `SWAP_`, `SET_`, `ROTATE_IF_`) appear
  in candidate local completions. The training interface is not being reused
  verbatim at deployment.

## Design consequence

The result rejects the registered five-stage/two-branch package at this dose. The
next trial should not add more of these canonical-code lessons or another termination
weight. A result-separated successor should target the interface mismatch directly:
variable-depth natural-language procedures, explicit state-table execution,
independent hypothesis simulation/scoring, and an answer-only commit after a verified
final state. It still needs exact-token replay, fresh procedural seeds, and the same
absolute gate. This is a prospective direction, not a post-hoc rescue of this arm.

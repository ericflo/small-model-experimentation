# Post-Model-Smoke 005 Audit: Live Controls Pass

## Verdict

`LIVE_BRANCH_CONTROLS_PASS`. The outcome-blind native-prefix write path is
numerically valid and may support separately implemented label-free mechanics.
No continuation or correctness stage is yet authorized.

## Results

- Exact Qwen3.5-4B revision and lens hash pass.
- One 386-token prompt plus 32 live thought tokens fits at 9,115,230,720 peak
  allocated bytes.
- J and non-J arms each contain 12 finite branches and apply once at every layer
  4--8.
- All 60 non-J rows pass paired live geometry.
- Maximum paired norm relative error is `9.3881e-6` <= `1e-5`.
- Maximum complete-J-span projection is `0.00912094` <= `0.01`.
- Iterative correction uses 7/370/113/181/512 steps by layer.
- Lattice correction uses at most five pairs; layer-8 rows use 0--5.
- Pre-bf16 rank-11, zero-sum, exact-Gram branch construction remains locked by
  CPU smoke. Realized Gram/zero-sum diagnostics are reported but not substituted
  for the two preregistered live gates.

The receipt stores no branch probabilities, choices, supplied-target success,
correct alias, outcome, or confirmation access.

## Next authorization boundary

Implement mechanics to generate exactly one 512-token live prefix per each of
four label-sealed mechanics tasks, evaluate alpha 0.5/1/2 target controllability
and identical non-J controls, and choose only the smallest passing alpha. The
mechanics runner must discard every gold/hidden field before model load, write
no correctness, preserve all live numeric receipts, and remain unable to run
continuations. Commit/push and audit it separately.

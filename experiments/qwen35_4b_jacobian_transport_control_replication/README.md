# Qwen3.5-4B Jacobian Transport Control Replication

**Status:** finished

This experiment independently replicates the perfect-but-invalid result from
`qwen35_4b_context_local_jacobian_clamp`. It freezes that experiment's lens and
band, generates fresh mappings, and repairs the only failed measurement: random
controls must satisfy both perturbation norm and J-span orthogonality **after**
bf16 application.

## Research Program

- Primary: `interpretability_and_diagnostics`
- Conditional secondary: `structured_execution_and_compilers` and
  `test_time_reasoning_budget` only after a valid replication.
- Closest near-duplicate: `qwen35_4b_context_local_jacobian_clamp`, terminal
  `INVALID_CONTROL` despite 48/48 J transport because one of 96 random rows had
  norm error 1.155e-5 >1e-5.

## Question

Does the frozen early context-local J clamp again change a separately computed
lookup consequence on fresh mappings when every random control is valid after
quantization?

## Fixed mechanism

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Frozen prior 24-concept lens, SHA-256
  `e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`.
- Frozen band `[4,5,6,7,8]`, alpha one, selected-key token position.
- Fresh 24-item numeric-control calibration and 48-item untouched confirmation.
- Two independent random controls per item and prompt kind.
- Every random layer delta must have post-bf16 relative norm error <=1e-5 and
  post-bf16 J-span projection fraction <=0.01. Requested vectors are projected
  orthogonal before quantization.
- Same-subspace specificity: wrong-donor J must produce the wrong donor's own
  digit, not the registered target.
- No digit direction or consequence gradient can construct any intervention.

The frozen rules are in [`reports/preregistration.md`](reports/preregistration.md)
and the pre-run adversarial review is in
[`reports/design_review.md`](reports/design_review.md).

## Run

```bash
.venv/bin/python -m pytest experiments/qwen35_4b_jacobian_transport_control_replication/tests -q
.venv/bin/python experiments/qwen35_4b_jacobian_transport_control_replication/scripts/run.py --stage smoke
.venv/bin/python experiments/qwen35_4b_jacobian_transport_control_replication/scripts/run.py --stage model-smoke
.venv/bin/python experiments/qwen35_4b_jacobian_transport_control_replication/scripts/run.py --stage confirmation
```

Confirmation consumes the exact committed calibration artifacts; do not rerun
and overwrite calibration in this result-bearing checkout. The calibration
command is retained as a stage for independent clean-checkout reproduction.

## Results

The outcome-blind model smoke passes all 20/20 random layer deltas. Maximum
post-bf16 relative norm error is `9.0113e-6` and maximum realized J-span
projection fraction is `0.0098674`, both inside the frozen gates. Exact lattice
repair was needed for four layer-8 rows (three one-pair repairs and one two-pair
repair). Model, lens, token, position, length, and causal-suffix contracts pass.

Three preceding failed smoke receipts are preserved. The implementation history
and post-smoke adversarial audit explain the geometry-only repair.

The frozen numeric firewall subsequently passed all 480/480 calibration rows.
Maximum relative norm error was `9.8216e-6`, maximum realized J-span projection
fraction was `0.00999293`, and exact causal-suffix difference remained zero.
Thirty-seven rows used lattice repair (34 at layer 8), with at most three pairs.
The calibration artifacts contain only numeric geometry and explicitly record
that logits/outcomes were not written. These are still plumbing results, not
causal evidence. The hash-locked eight-arm confirmation runner and a second
pre-run adversarial implementation audit were completed before the single run.

Untouched confirmation returns `REPLICATED_J_TRANSPORT`:

- all-24 J: 48/48 target keys and 48/48 target mapped digits;
- full target donor: 48/48 and 48/48;
- source/target pair J: 48/48 and 46/48;
- wrong-donor J: 48/48 its own key and digit, 0/48 registered target;
- concept logit lens, random_a, and random_b: 0/48 target in both prompt kinds;
- both paired-bootstrap J-minus-random 95% intervals: `[1.0, 1.0]`; and
- 960/960 realized confirmation control-layer rows valid, with maximum relative
  norm error `9.9709e-6` and maximum J-span projection `0.0099970`.

This independently confirms that the early context-local J coordinates carry a
causally consumed concept state, not merely an imminent output direction.

## Scope

Even a valid replication is oracle causal-mechanism evidence. The target concept
and donor coordinates are supplied. It would license, but not itself constitute,
a new experiment on native thinking and a learned non-oracle controller. It is
not a capability gain and has not been compared with matched-compute sampling.

## Knowledgebase Update

- Remain unclaimed while the repository claim re-grade is open.
- Update program evidence only after the frozen terminal decision.

## Artifacts

All small data, the frozen lens copy, failed and passing smoke receipts,
calibration/confirmation controls, outcome rows, metrics, and reports are
committed. No training or adapter is used.

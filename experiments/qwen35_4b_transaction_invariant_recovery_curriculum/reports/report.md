# Transaction-invariant recovery curriculum report

## Verdict

**`TRANSACTION_DEV_FAIL`.** The fixed primary passed exact locality and produced
a large trained-family install, but gained only 1/64 over the parent and
equal-compute sample-more on unseen transaction families. Confirmation, broad
retention, and Menagerie remained sealed.

## Results

| Stage / arm | Parent | Replay-only | Sample-more | Transaction curriculum |
| --- | ---: | ---: | ---: | ---: |
| Apex-relative locality drift | — | — | — | **0.119** (pass ≤0.15) |
| Trained-family calibration | 51.7% | 38.3% | — | **81.7%** |
| Unseen transaction dev | 70.3% | 64.1% | 70.3% | **71.9%** |

Calibration passed every frozen gate: +30.0 points over the parent, +43.3
over matched replay-only, 100% for both changed-patch-within-two transitions,
and lower invalid-action and answer-cap rates. Locality was also clean: entropy
rose 0.011 nats and varentropy changed by only −0.0002.

Transfer did not meet the claim. Candidate minus parent was +1.56 points with a
paired 95% bootstrap interval [−3.13,+7.81]; candidate minus sample-more was
also +1.56 [−4.69,+7.81]. It beat replay-only by +7.81 [1.56,15.63] and did not
regress any family, but the registered bars were +10/+5/+5.

## Mechanism Forensics

The aggregate null hides a sharp partial success. On all 16 atomic-reservation
cases, the candidate's first changed patch:

- copied the capacity mapping;
- validated every resource in a request before subtracting;
- committed a request atomically and returned `False` when it did not fit.

That is the conjunction the predecessor never proposed. Every patch omitted
the separate negative-amount rule (`ValueError`). After the visible failure,
all 16 trajectories overcorrected by raising on *every* invalid or insufficient
request, destroying the required per-request `False` behavior. Final success
there remained 0/16. The curriculum shifted proposals, but it taught a generic
transaction template rather than verifier-faithful validation-policy
discrimination.

The three newly skinned transfer families were already near saturation:
candidate/parent was 100/100% on debits, 100/100% on membership moves, and
87.5/81.25% on document patches. The original sentinel therefore remained the
only headroom and the only shared failure.

## Operational Correction

The control-first analyzer caught mismatched procedural manifests before
candidate exposure. Python's randomized set rendering changed seat-test bytes
across processes and, on Ada, altered deterministic batch outputs. The invalid
controls were quarantined; every official child now freezes
`PYTHONHASHSEED=0`; corrected manifests match. This is codified in tests and
the design review.

## Interpretation

The promising next unit is not more transaction families or more dose. It is a
counterexample-conditioned validation-policy curriculum: start from the local
candidate, create near-correct public-failure states where exactly one policy
distinction is wrong (negative→raise, unavailable→False, malformed→None,
duplicate→reject), and teach the smallest changed patch that preserves the
other transaction invariants. Mix complete recovery replay and compare against
matched extra transaction/recovery dose.

This result supports proposal-structure installation but not transferable task
success or a general capability unlock. No benchmark seed was consumed.

## Artifacts

Compact metrics and hashes are in
[`result_receipt.json`](result_receipt.json). Detailed banks, checkpoints,
locality rows, valid trajectories, and quarantined invalid controls remain under
`large_artifacts/qwen35_4b_transaction_invariant_recovery_curriculum`.

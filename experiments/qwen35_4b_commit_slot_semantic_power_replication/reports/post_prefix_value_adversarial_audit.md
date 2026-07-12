# Post-Decision Adversarial Audit: Prefix J-Value

Completed after automatic `NO_PREFIX_J_VALUE` and before any post-decision
diagnostic. It cannot change the registered decision or open causal data.

## Verdict

Accept the run as a valid scientific negative for one shared, phase-invariant
120-coordinate J-value readout. Preserve midpoint 0.6083 versus endpoint 0.3958
as a phase-interaction lead only. Do not average, sign-flip, refit, or subset it
into a pass. `causal_confirmation` remains sealed.

## 1. The failure is an implementation invalidity

It is not. All 144 traces, 288 rows, five rank-24 dictionaries, exact contexts,
folds, finite features, and non-J projection controls passed. The within-task
shuffled null averaged 0.5061. The terminal label varied on 40 tasks.

## 2. The midpoint gate passed, so value passed

The primary frozen gate required shared AUC >=0.65, uncertainty above chance,
and incremental wins. Shared AUC was 0.5021; its lower bound was 0.4417. A
single conjunctive sub-gate cannot rescue the experiment.

## 3. Endpoint sign should be flipped post hoc

Sign was learned on training tasks under one frozen model. Flipping only one
fraction after seeing outcomes is a new phase-specific model, not correction.
It requires new preregistration and data.

## 4. Train separate midpoint and endpoint models now

That is allowed only as explicitly post-decision diagnosis on the already-open
value rows. Any resulting point estimate is selection evidence, not
confirmation. It cannot license causal work.

## 5. Pooling more pairs would restore power

Paths and fractions are correlated within 48 tasks. The task remains the fold,
macro, and bootstrap unit. Replacing task uncertainty with 288-row uncertainty
would be pseudo-replication.

## 6. J is still special because it beat gold-alias activity

The point difference was +0.0521, but its paired task lower bound was -0.0396.
J also lost to ordinary slot margin by -0.0427 and equal-width non-J residual
state by -0.0271. Specificity and incrementality both failed.

## 7. Non-J state validates a general certainty signal

Non-J AUC 0.5292 was also below any registered capability/value bar. It only
shows that J did not dominate a dimension/layer-matched generic readout.

## 8. The seam replication is contradicted

No. The seam asks whether ordered content changes the fixed answer choice. The
value stage asks whether one cross-task linear J readout ranks paths. Useful
thought content need not be linearly readable in this coordinate system.

## 9. Endpoint below chance proves anti-certainty

Not without independent replication. It may reflect a real phase-dependent
coordinate rotation, shared-model misspecification, alias/task imbalance, or
sampling error. The four fold AUCs were themselves unstable. Treat reversal as
a hypothesis generator.

## 10. Causal patching could adjudicate the reversal

The causal split was conditioned on a passing value axis and exact controls.
Patching a failed, adaptively reinterpreted axis would erase the firewall.
Do not open it.

## Allowed post-decision diagnostics

Using only the already-open value rows and labels:

- recompute task bootstrap for the registered midpoint slice;
- fit midpoint-only and endpoint-only OOF models under the same folds;
- cross-apply each phase's train-fold model to the other phase;
- compare coefficient alignment and feature/label stability by phase;
- repeat the same operations for margin and non-J features; and
- report alias/fold sensitivity.

These diagnostics must be labeled post hoc, deterministic, and unable to alter
`NO_PREFIX_J_VALUE`. If a midpoint-specific signal survives all direct/non-J
controls, the only licensed next step is a new fresh task split with frozen
phase-specific rules and independent confirmation.

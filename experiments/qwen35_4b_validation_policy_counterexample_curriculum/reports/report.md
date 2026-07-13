# Report: validation-policy counterexample curriculum

## Verdict

`CALIBRATION_INFEASIBLE`. The candidate was local, but the parent and matched
control were already perfect on the fresh trained-family instrument. The
registered effect bars could not be earned, so candidate behavior, transfer,
retention, and Menagerie remained sealed.

## What ran

The immutable design at `e0b19f5d` used the learned transaction checkpoint as
parent. Candidate and control each trained for 36 transition-stratified
optimizer steps over 336 rows with identical 152,992 total weighted action mass
per epoch and zero think loss. Candidate replaced only 24 post-failure revision
rows; 312 rows retained the prior bank. Control retained all 336 prior rows.

Candidate training loss was 0.013996 versus control 0.018505; merged delta-norm
sums were 2.908 and 2.957. These are diagnostics, not capability results. GPU
smoke passed 12/12 and exercised training, explicit composite merge, and the
real vLLM looping harness.

## Locality

Candidate versus C54 apex passed on all 48 fresh non-coding contexts:

- median centered non-target logit drift: 0.10944 (ceiling 0.15);
- mean entropy delta: +0.02142 (floor −0.05);
- mean varentropy delta: −0.01082 (diagnostic only).

This supports the narrow engineering point that one-transition, three-epoch
action-only replay can stay local. It does not show behavioral benefit.

## Controls-first calibration

The parent and matched control were generated before candidate behavior on 24
fresh train-skin repositories × two controlled recovery states:

| Arm | Success | Failed-test changed within two | Rejected valid changed within two | Invalid actions/turn |
| --- | ---: | ---: | ---: | ---: |
| learned transaction parent | 48/48 | 24/24 | 24/24 | 0.000 |
| extra-transaction control | 48/48 | 24/24 | 24/24 | 0.000 |

The theoretical candidate ceiling is 48/48. It cannot exceed parent by 15
points or control by 10 points, so the feasibility analyzer failed exactly
`success_vs_start` and `success_vs_control` and stopped.

## Mechanism forensics

This is not merely “the tasks were easy.” The task construction changed the
meaning of the predecessor's residual:

- the issue explicitly stated negative quantities must raise `ValueError`;
- the failed-test partial already copied state, validated existence/capacity,
  returned `False` for ordinary rejection, committed atomically, and preserved
  inputs;
- visible output directly identified the missing negative exception.

On every one of 48 parent trajectories, the first changed patch contained the
negative check, copied state, and the ordinary false-decision path. In rejected-
patch states the parent frequently wrote the complete correct program directly
from the initial source; in failed-test states it made the preserving revision
immediately. The original atomic-reservation failure had a more implicit
contract (nonnegative input domain without an explicit exception sentence) and
different proposal dynamics. Converting that failure into explicit instruction
converted it into an already-solved editing task.

## Interpretation

The counterexample mechanism remains untested, not disproven. The experiment
establishes a reusable ordering rule: failure forensics are not sufficient to
design a training substrate. Before capability-production spend, the exact
procedural task distribution must show replicated parent headroom under the
intended prompt and verifier feedback. This is the same portfolio principle
seen in specialist qualification, now at the curriculum-substrate level.

The next step is a no-training headroom tournament over multiple semantic
conflicts and public representations. It should distinguish explicit contract
following from inference under verifier evidence, then admit only axes with a
non-saturated parent band into a separately preregistered curriculum.

## Exposure ledger

- Candidate scientific calibration: not generated.
- Policy transfer development/confirmation: not generated.
- Broad recovery/normal retention: not generated.
- Menagerie: not invoked; seeds 71301/71302 unconsumed.
- Claim ledger: unchanged.

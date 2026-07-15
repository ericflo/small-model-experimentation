# Idea Intake: Axis-on-Replay Stack with Medium Pilot

## Program Fit

- Program: `agentic_breadth_installation`.
- Existing or new program: existing.
- Closest scorecard: Agentic Breadth Installation.
- Related-work discovery: `make related QUERY="replay compounding axis install medium tier family conversion stack"`.

## Prior Evidence

- Anchor 1: `qwen35_4b_goal_gap_axis_curriculum_target_match` — the axis atoms
  INSTALL (first local promotion in the line; +6/+10 on held-out axis tasks,
  retention byte-equal to parent) but under-convert at quick tier; its pilot's
  replay control posted the line's best recorded aggregate (0.5081) and flipped
  rites, while the axis candidate flipped warren.
- Anchor 2: replay continuation has compounded aggregate three consecutive
  times across lineages (0.4410→0.4851; parent 0.4644→replay 0.5081).
- Anchor 3 (goal-gap forensics): the all-families goal has been reached 8 times
  in 92 MEDIUM-tier events versus once in 65 quick events — family conversion
  is tier-dependent, and medium is the repository's confirm tier, which the
  goal itself names.
- Closest near-duplicate: the goal-gap axis experiment. This trial changes
  exactly three things: the parent (the 0.5081 replay composite), the control
  (a second replay round, which measures replay compounding inside the same
  event), and the pilot tier (medium).

## Novelty Claim

First trial to stack the proven axis-atom install on the strongest replay-
compounded parent, with a pilot at the tier where the goal is empirically
reachable — and the first to measure whether replay compounding continues or
saturates at round two, inside the same paired event.

## Mechanism

The axis install and replay compounding are independent effects with disjoint
family footprints at seed 78,144 (axis flipped warren; replay flipped rites);
stacking them on one model should union the footprints if the effects are
weight-compatible, and the medium tier's finer family granularity plus episode
items should convert installed multi-turn-flavored skills that quick-tier atoms
miss. The explanation is false if the stacked candidate loses its axis-holdout
wins (interference), regresses retention, or fails to beat the replay-squared
control at medium.

## Control Plan

- Baseline: the `replay_parent` composite (0.5081 at seed 78,144).
- Mechanism-falsifying control: `replay_squared` — a second exact-exposure
  replay round from the same parent, which simultaneously answers the queued
  replay-compounding question.
- Shift/robustness: fresh gate seed 88,015 for both instruments; conditional
  medium-tier pilot at sealed seed 78,145; independent-seed and sample-more
  confirmation remain mandatory before any universal claim.
- Hidden boundary: `benchmarks/` remains unread.

## Evidence Output

- Program evidence: stack-versus-interference reading, replay round-two
  compounding measurement, and medium-tier family conversion record.
- Shared synthesis: update the replay-compounding and conversion laws.
- Reusable artifact: none new; the inherited corpus and pipeline are reused.
- Stop condition: MILP infeasibility stops before training; gate failure seals
  seed 78,145; no bar or seed changes after any output.

## Decision

- Run experiment: proceed through model-free construction, two training events,
  two merges, one gate event, and one conditional medium pilot.
- Create program: no.
- Write synthesis only: no.
- Defer: every model stage until its prerequisite checkpoint is committed,
  pushed, and green.

Inherited corpus construction seed is 77,117 (byte-identical inheritance).
Fresh slot-match/training/gate/aggregate seeds are `55119/53/88015/78145` and
cannot change after their corresponding event.

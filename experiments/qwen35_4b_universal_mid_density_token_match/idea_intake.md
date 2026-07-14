# Idea Intake: Mid-Density Exact-Token Bridge

## Intake

- Date: 2026-07-13
- Program: `agentic_breadth_installation`
- Parent result: `qwen35_4b_universal_low_density_token_match`
- Closest near-duplicate: the parent exact-token 0/40/80 ladder.

## Idea

Test whether representative 160- or 240-row doses cross the abstract-procedure
installation threshold from the replay-refresh anchor while leaving substantially
more replay mass than the earlier 400-row mixture that passed locally but lost broad
capability.

## Why this is not a duplicate

The parent established that 40 and 80 rows are insufficient under exact token parity.
The replay-anchor predecessor established that 400 rows can pass a local gate from a
different starting point but then loses to replay continuation broadly. No experiment
has measured the 80-to-400 dose gap from the authenticated replay-refresh anchor with
row, slot, update, and forward-token parity.

## Feasibility result before freeze

The initial 0/160/240/320 proposal failed an outcome-free construction audit. A
proportional 320-row designed selection was shorter than even the 320 shortest replay
rows, so exact row-level substitution would require length-biasing the designed
curriculum. The registered experiment therefore stops at 240 rows. This is a design
constraint, not a model result; a length-aware 320-row arm would require a separate
intake and experiment.

## Decision

Proceed with a nested 0/160/240 ladder, fresh local seed 88,005, and conditional
aggregate-only seed 78,135. Stop before merge and benchmark if no arm passes the same
absolute local mechanism gate used by the parent line.

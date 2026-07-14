# Idea Intake: Low-Density Token-Matched Universal Curriculum

## Direction

Continue from the replay-refreshed policy with a nested 40/80-row dose of the
truth-audited abstract curriculum, while matching a replay-only continuation exactly
on training slots and forward-token exposure.

## Closest near-duplicate

`qwen35_4b_universal_replay_anchor` is the direct near-duplicate. Its 400-row designed
arm passed local gates but lost 0.0613 aggregate to replay-only. The replay control had
17.3% more forward tokens, so designed density and compute exposure were confounded.

## Why this is materially new

- Starts from the stronger replay-refreshed policy rather than C53 `blend`.
- Reduces designed density from 26.3% to prospectively frozen 2.6% and 5.3% doses.
- Uses position-aligned nested replacements: every non-replaced replay row is shared.
- Matches all arms exactly at 1,429,053 forward tokens, 1,520 rows, and 190 steps.
- Registers both doses before training and sends every locally eligible arm to the
  same fresh paired event, avoiding adaptive benchmark dose selection.

## Falsifiers

- Neither designed dose passes the frozen local safety/installability gate.
- Replay continuation beats both designed doses at exact compute.
- A candidate leaves any public family at or below base, fails to beat the inherited
  anchor, or fails to beat exact-token replay continuation.
- Any apparent pilot winner fails independent quick, medium, uncertainty, or
  matched-compute sampling confirmation.

## Contamination boundary

Only source corpora already frozen in the parent are copied. Benchmark family names
and aggregate scores are public metadata; no item, family source, transcript, raw
stream, verifier detail, or private result enters the curriculum or agent context.

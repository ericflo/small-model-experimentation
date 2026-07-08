# Benchmarks — held-out measurement instruments

This directory holds **evaluation suites that the experiment pipeline must never
train on**. They exist to answer one question the experiment corpus cannot answer
from inside its own substrates: *did a method install generalizable capability,
or did it just fit the training distribution?*

Much of the corpus's experimentation is deliberately whitebox — train on a
substrate, evaluate on held-out items from the same generator. That design can
only ever show distribution-fitting. The suites here are the blackbox
counterweight: bespoke, procedurally generated, disjoint from every public
benchmark and from every training substrate in `experiments/`. A method that
moves scores *here* moved something general.

## The firewall (absolute)

- Experiments may **run** these suites (via each suite's `run.py`) and record
  scores. That is the entire permitted surface.
- Experiments must never: import family modules, read family source or
  generated items, train on transcripts/items/scores-as-labels derived from
  these suites, or copy family content into training data. If benchmark
  content leaks into training, the instrument is destroyed — there is no way
  to un-leak it.
- Fresh instances are cheap: every run can use a new `--seed`, so there is
  never a reason to reuse (and thereby expose) a fixed item set.

## Suites

- [`menagerie/`](menagerie/) — tiered agentic capability suite (quick / medium /
  slow / deep). See its README and `CONTRACT.md`.

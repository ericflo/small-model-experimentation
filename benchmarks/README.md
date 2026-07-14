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

- Experiments may **run** these suites and record scores. Direct operator runs
  use each suite's `run.py`; experiment orchestration that needs a strict
  process boundary uses `scripts/run_benchmark_aggregate.py`, which invokes the
  suite privately and exposes only aggregate and per-family scores. Those CLIs
  are the entire permitted surface.
- Agents must not even read contents under benchmark suite subdirectories:
  family sources, generated items, transcripts, and result details contaminate
  the orchestrating agent's context and can leak into experiments it later
  writes. The only permitted interactions are invoking a suite's `run.py`, its
  `validate_suite.py` CLI, or the trusted aggregate gateway, and reading
  aggregate scores.
- Experiments must never: import family modules, read family source or
  generated items, train on transcripts/items/scores-as-labels derived from
  these suites, or copy family content into training data. If benchmark
  content leaks into training, the instrument is destroyed — there is no way
  to un-leak it.
- Directory names themselves, including suite and family names, are deliberately
  public metadata and are acceptable to see in `git status` or command output;
  the contents behind those names remain read-forbidden.
- Fresh instances are cheap: every run can use a new `--seed`, so there is
  never a reason to reuse (and thereby expose) a fixed item set.
- Raw suite stdout, stderr, and output files must not enter an experiment
  process. The aggregate gateway suppresses child streams, owns any raw output
  in a private temporary directory, and deletes it before returning a fixed
  aggregate-only schema. Infrastructure failures report only a generic exit
  status plus whether a private output existed and passed the aggregate schema;
  no score or raw log is echoed into agent context.

## Suites

- [`menagerie/`](menagerie/) — tiered agentic capability suite (quick / medium /
  slow / deep). See its README and `CONTRACT.md`.

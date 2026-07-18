# Coding-harness transfer measurement (2026-07-17)

**The measurement the whole program's value rests on: do the menagerie-aggregate
gains translate to real agentic coding? Answer at n=35: NO — statistically
indistinguishable from the raw base.**

## What was measured

The three current composites were run through a REAL agentic coding harness
(`duet-eval` driving the Pi coding agent against each model served locally),
graded on real filesystem outcomes (does the produced edit make the scenario's
tests pass), on 35 held-out `test`-split scenarios (gen4 clusters:
result/pyalgo/iter/fsm/num, difficulty 3–5). Config `bare` (library disabled =
raw model capability, the model-ranking control). Same 35 scenarios, same
backend/flags, temperature 0, per model.

- Arms: `base` (raw Qwen/Qwen3.5-4B), `count_walk` (the enumeration composite,
  lifecycle 27), `state_track` (the divergent-skill composite, lifecycle 30 —
  the program's nominal "best" by menagerie aggregate).

## Result

| model | passed | rate | 95% CI (Wilson) |
|---|---|---|---|
| base | 8/35 | 22.9% | [12.1%, 39.0%] |
| count_walk | 9/35 | 25.7% | [14.2%, 42.1%] |
| state_track | 8/35 | 22.9% | [12.1%, 39.0%] |

Paired McNemar (exact) vs base: count_walk p=1.00, state_track p=1.00 — NOT
significant. The three arms are statistically indistinguishable.

## The key nuance

10 of 35 scenarios are solved by DIFFERENT subsets across the three models
(count_walk uniquely solves several FSM tasks; base uniquely solves a couple of
algorithm/number tasks; state_track uniquely solves interval-merge and
union-find). So the LoRA training MOVED capability around — it changed WHICH
coding tasks the 4B can do — but did NOT change HOW MANY. Net coding capability
is flat.

## Interpretation (the pivotal finding)

The menagerie aggregate (which the composites lifted ~2–3× over base, and where
state_track edged count_walk by a confirmed-but-soft +0.02) is DISCONNECTED from
real agentic coding capability. Optimizing the proxy did not move the target.
This reframes the program: continuing to push the menagerie aggregate is not, on
this evidence, a path to a better coding agent.

## Honest scope / caveats

- n=35, one difficulty band (3–5), one harness, `bare` config only.
- `bare` measures RAW model capability. Duet's actual product value is the
  crystallized-workflow LIBRARY lift (`--config library`/`both`) — the scaffolding
  designed to make weak local models useful. That was NOT measured here and is
  the natural, higher-relevance next measurement: can scaffolding make the 4B
  useful even though bare capability is flat?
- Temperature 0, single rep (deterministic capability read, no sampling spread).
- This does not say the menagerie work was worthless — it produced durable
  laws (install≠convert, execute≠induce, replay-compounding bounds, the
  divergent-skill add) and a clean reproducible lineage. It says the menagerie
  aggregate is not a valid PROXY for the coding target.

## Firewall (measurement-only, no contamination)

`duet-eval` is private and holds answer keys. This measurement copied NOTHING
from it into the repo and trained on NOTHING from it — only the aggregate
pass counts and per-scenario pass/fail booleans (this repo's own measurements)
are recorded here. Serving recipe and raw per-scenario booleans live in the
session scratchpad, not the repo.

## Serving notes (for reproduction)

Qwen3.5-4B is a thinking model that emits Qwen-XML tool calls. The working
vLLM serve config is: `--enable-auto-tool-choice --tool-call-parser qwen3_coder
--reasoning-parser deepseek_r1` (hermes parser silently drops the XML tool
calls; without a reasoning parser the `<think>` block corrupts tool parsing).
The merged composites also need the base model's `preprocessor_config.json` +
`video_preprocessor_config.json` present to serve (the merger omits them).

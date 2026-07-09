# Base Qwen3.5-4B baselines

Base model (no adapter), **no-think, greedy**, seed `20260709`, on the
parse-corrected instrument. These are the honest starting lines the install
experiments must beat. Scores are per-family mean; aggregate is the mean of
family means.

| family | quick | medium | slow | deep | what it measures |
|---|---|---|---|---|---|
| chronicle | 0.125 | 0.083 | 0.080 | 0.048 | event-stream state tracking |
| lockpick | 0.000 | 0.000 | 0.000 | 0.010 | active rule induction → exploit |
| menders | 0.000 | 0.000 | 0.007 | 0.013 | program repair from failing traces |
| mirage | 0.000 | 0.000 | 0.000 | 0.000 | calibrated abstention (provable unsolvability) |
| rites | 0.000 | 0.000 | 0.000 | 0.000 | state-machine / spec compliance |
| siftstack | 0.000 | 0.000 | 0.000 | 0.000 | information triage under noise/contradiction |
| sirens | 0.500 | 0.292 | 0.220 | 0.221 | goal fidelity under prompt injection |
| stockade | 0.019 | 0.096 | 0.056 | 0.054 | bounded optimization vs brute-forced optimum |
| toolsmith | 0.344 | 0.236 | 0.191 | 0.188 | dependent tool-call chaining |
| warren | 0.000 | 0.090 | 0.197 | 0.147 | partially-observable exploration + memory |
| **aggregate** | **0.099** | **0.080** | **0.075** | **0.068** | |

| tier | wall | budget | headroom |
|---|---|---|---|
| quick | 8.0 s | 60 s | 7.5× |
| medium | 49.1 s | 300 s | 6.1× |
| slow | 187.7 s | 1200 s | 6.4× |
| deep | 459.4 s (7.7 min) | 3600 s | 7.8× |

All four tiers finish comfortably within budget on a single RTX 4090, confirming
the lockstep batched design on real hardware. (Aggregate drifts down across tiers
because higher tiers add harder L3/L4 levels the quick tier omits.)

## The instrument is hard and honest (by design)

Aggregate ~0.07–0.10 with six families at/near the floor is the **desired**
property for a measurement instrument: no ceiling effect, maximal headroom for an
install method to demonstrate real gain. A blackbox suite the base model already
aced would tell the install experiments nothing.

A parse bug initially masqueraded as capability failure — the model omitted the
exact `ANSWER:` prefix, so `score()` discarded well-formed answers. That was
fixed (last-line format instruction + a tightly-constrained bare-answer
fallback). A post-fix debug audit (via `codex`, honoring the read-firewall) then
bucketed every remaining zero to confirm capability vs residual format:

- **Genuine capability floor (6/7 of the previously-suspect families):**
  lockpick, menders, mirage, rites, siftstack, warren. Their zeros are
  wrong-value or wrong-shape/no-answer outputs, not recoverable-but-misformatted
  answers. Consistent with the corpus's own findings that the 4B's hardest limits
  are induction, multi-step planning, and spec execution — and that these need
  chain-of-thought (try `--think`, expected to lift several of these).
- **chronicle** is the exception: ~27% of its zeros are the correct answer
  wrapped in a reasoning *sentence*, which the constrained fallback correctly
  rejects. We **deliberately do not** add a "pull the right name out of a
  sentence" extractor: it would violate the contract's machine-checkable /
  no-fuzzy-semantics rule and is gameable by prose policies. chronicle's baseline
  is therefore a mild **under**-estimate — the honest, un-gameable scoring is
  worth the few points.

## Tier predictiveness: instrument vs base model

Two different questions, two different answers:

- **Instrument predictiveness (the design property): validated.** On synthetic
  policies of graded strength (noisy-oracle ladder ε∈{0,.25,.5,.75,1}), tier
  aggregate scores rank-correlate **Spearman 1.000 for every tier pair**
  (`validate_suite.py`). Cheap tiers rank *capability levels* exactly as the
  expensive tiers do. This is what makes quick usable as a training-time proxy.
- **Base-model tier correlation: 0.77 quick→deep**, lower than 1.0 — and
  honestly so. With six families pinned at the floor, the family ranking is
  tie-dominated and therefore unstable, and one family (`warren`) is floored at
  the easy levels quick tests but capable at the harder levels slow/deep add
  (0.000 → 0.197). So **for a floored model the cheap tier under-predicts the
  families whose signal only appears at depth.** As install methods lift families
  off the floor, ties break and the real-model tier correlation should climb
  toward the synthetic 1.0. Practical rule: trust quick for *relative* progress
  tracking during training; confirm on slow/deep before drawing a conclusion, and
  watch depth-sensitive families (warren-like) specifically.

## Reproduce

```bash
PY=/home/ericflo/Development/small-model-experimentation/.venv/bin/python
cd benchmarks/menagerie
$PY run.py --tier quick  --backend qwen --seed <fresh> --out results/quick_<tag>.json
$PY run.py --tier deep   --backend qwen --seed <fresh> --max-batch 32 --out results/deep_<tag>.json
# add --think to give the model chain-of-thought (slower; expected to lift
# the induction/planning/spec families)
```

Use a **fresh seed** each time (items stay unexposed; determinism is per-seed).
Compare runs only at the same seed+tier. `--max-batch 32` lowers peak VRAM if the
card is shared.

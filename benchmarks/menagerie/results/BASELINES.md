# Base Qwen3.5-4B baselines

Base model (no adapter), **no-think, greedy**, seed `20260709`, on the
parse-corrected instrument. These are the honest starting lines the install
experiments must beat. Scores are per-family mean; aggregate is the mean of
family means.

| family | quick | medium | what it measures |
|---|---|---|---|
| chronicle | 0.125 | 0.083 | event-stream state tracking |
| lockpick | 0.000 | 0.000 | active rule induction → exploit |
| menders | 0.000 | 0.000 | program repair from failing traces |
| mirage | 0.000 | 0.000 | calibrated abstention (provable unsolvability) |
| rites | 0.000 | 0.000 | state-machine / spec compliance |
| siftstack | 0.000 | 0.000 | information triage under noise/contradiction |
| sirens | 0.500 | 0.292 | goal fidelity under prompt injection |
| stockade | 0.019 | 0.096 | bounded optimization vs brute-forced optimum |
| toolsmith | 0.344 | 0.236 | dependent tool-call chaining |
| warren | 0.000 | 0.090 | partially-observable exploration + memory |
| **aggregate** | **0.099** | **0.080** | |

Timing: quick **8.0 s** / 60 s budget · medium **49.1 s** / 300 s budget — both
comfortably within tier budgets, confirming the lockstep design on real hardware.
Slow/deep pending an uninterrupted GPU window (the training pipeline shares the
card; larger-batch runs keep losing the race to CUDA contention).

## The instrument is hard and honest (by design)

Aggregate ~0.08–0.10 with six families at the floor is the **desired** property
for a measurement instrument: no ceiling effect, maximal headroom for an install
method to demonstrate real gain. A blackbox suite the base model already aced
would tell the install experiments nothing.

A parse bug initially masqueraded as capability failure — the model omitted the
exact `ANSWER:` prefix, so `score()` discarded well-formed answers. That was
fixed (last-line format instruction + a tightly-constrained bare-answer
fallback). A post-fix debug audit (via `codex`, honoring the read-firewall)
then bucketed every remaining zero to confirm what is capability vs residual
format:

- **Genuine capability floor (6/7 of the previously-suspect families):**
  lockpick, menders, mirage, rites, siftstack, warren. Their zeros are
  wrong-value or wrong-shape/no-answer outputs, not recoverable-but-misformatted
  answers. These floors are real. Consistent with the corpus's own findings that
  the 4B's hardest limits are induction, multi-step planning, and spec execution
  — and that these need chain-of-thought (try `--think`, expected to lift
  several of these).
- **chronicle** is the one exception: ~27% of its zeros are the correct answer
  wrapped in a reasoning *sentence* (multiple tokens on the final line), which
  the constrained fallback correctly rejects. We **deliberately do not** add a
  "pull the right name out of a sentence" extractor: it would violate the
  contract's machine-checkable / no-fuzzy-semantics rule and is gameable by
  prose policies. chronicle's baseline is therefore a mild **under**-estimate;
  the honest, un-gameable scoring is worth the few points.

## Reproduce

```bash
PY=/home/ericflo/Development/small-model-experimentation/.venv/bin/python
cd benchmarks/menagerie
$PY run.py --tier quick  --backend qwen --seed <fresh> --out results/quick_<tag>.json
$PY run.py --tier medium --backend qwen --seed <fresh> --out results/medium_<tag>.json
# add --think to give the model chain-of-thought (slower; expected to lift
# the induction/planning/spec families)
```

Use a **fresh seed** each time (items stay unexposed; determinism is per-seed).
Compare runs only at the same seed+tier.

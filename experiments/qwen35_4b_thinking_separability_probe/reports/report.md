# Qwen3.5-4B Thinking Separability Probe Report

## Summary

We probe whether Qwen3.5-4B's **answer-token hidden state** linearly encodes whether its **own**
generated MBPP solution is correct, and whether **thinking** makes that correctness more decodable.
Three results. (1) Correctness IS moderately linearly decodable from a single answer-token
activation (best-layer AUC ~0.64–0.76; shuffled-label control ~0.50) — a "models know more than
they say" effect, here on a *self-generated real-code* answer token, which prior corpus probes
never touched. (2) Thinking **robustly raises** decodability at essentially every layer (no_think
~0.5–0.64 → thinking ~0.67–0.75): the model "knows it's right better after thinking." (3) But that
increase is **not coherent reasoning** — *shuffled* thinking matches or exceeds *real* thinking on
separability at both budgets and across all layers, refuting our pre-registered hypothesis. This
**converges with the behavioral shuffle finding (C9) at the representational level**: thinking's
benefit, behaviorally and now internally, is largely compute/scaffold/token-presence, not logical
content. A modest deployable spinoff: among visible-test passers (the C2 false-pass regime), the
probe has weak signal under thinking (AUC ~0.60–0.68) but ~chance under no-think.

## Research Program Fit

Third experiment of `test_time_reasoning_budget`. It turns C9's "reasoning vs compute" debate into
a measurable interpretability quantity (probe AUC by layer, think vs shuffle vs no-think) and gives
an internal-signal angle on the C2 selection bottleneck. The pre-registered hypothesis — that real
thinking would raise separability *above* shuffled thinking and thereby isolate the genuine-reasoning
contribution — is **falsified**, which is itself the durable lesson.

## Method

- Model Qwen3.5-4B frozen (bf16, sdpa, fast path). MBPP sanitized `test`, 100 tasks, k=8 sampled
  solutions per task per condition: `no_think`, `think_512`, `shuffle_512`, `think_1024`, `shuffle_1024`.
- Signal: the **last-token** hidden state of the model's own generated sequence (prompt + thinking +
  answer), extracted per layer (33 states: embeddings + 32 layers) via a clean **right-padded**
  forward pass (so the linear-attention recurrence is uncorrupted by padding).
- Probe: per-layer standardized logistic regression, **GroupKFold by task** (no task-identity
  leakage), out-of-fold AUC predicting full-test execution pass; bootstrap CI by resampling tasks.
- Controls: shuffled-label probe (must ≈0.5); real vs shuffled thinking at matched budget.

## Results

Best-layer probe AUC for predicting full-test pass (n=100 tasks, 800 samples/condition):

| condition | base pass | best layer | probe AUC | 95% CI | shuffled-label | visible-passer AUC (n) |
| --- | ---: | ---: | ---: | :---: | ---: | ---: |
| no_think | 0.77 | 24 | 0.642 | [0.55,0.74] | 0.514 | 0.518 (647) |
| think_512 | 0.85 | 11 | 0.708 | [0.65,0.77] | 0.482 | 0.684 (722) |
| shuffle_512 | 0.79 | 6 | 0.733 | [0.67,0.81] | 0.511 | 0.626 (661) |
| think_1024 | 0.85 | 10 | 0.720 | [0.61,0.81] | 0.456 | 0.598 (716) |
| shuffle_1024 | 0.83 | 16 | 0.755 | [0.68,0.81] | 0.520 | 0.669 (696) |

Per-layer AUC (every 4th layer) — the pattern is robust across depth, not a best-layer artifact:

| condition | L4 | L8 | L12 | L16 | L20 | L24 | L28 | L32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_think | 0.49 | 0.50 | 0.57 | 0.55 | 0.59 | 0.64 | 0.63 | 0.59 |
| think_512 | 0.68 | 0.68 | 0.65 | 0.61 | 0.68 | 0.68 | 0.68 | 0.66 |
| shuffle_512 | 0.67 | 0.69 | 0.67 | 0.67 | 0.70 | 0.69 | 0.71 | 0.70 |
| think_1024 | 0.64 | 0.67 | 0.68 | 0.67 | 0.69 | 0.65 | 0.64 | 0.69 |
| shuffle_1024 | 0.69 | 0.68 | 0.70 | 0.76 | 0.70 | 0.71 | 0.73 | 0.74 |

Figure: `analysis/auc_vs_layer.png`.

### Finding 1 — correctness is moderately decodable from one answer-token activation
All conditions beat the shuffled-label control (~0.50) with best-layer AUC 0.64–0.76. So the
model's residual stream linearly encodes whether its own just-written solution is correct — a
"models know more than they say" effect, novel on a self-generated real-code answer token (prior
corpus probes read external candidates on synthetic tasks). Decodability is highest in **early–mid
layers** (best layers 6–16 for thinking conditions), not the final layer.

### Finding 2 — thinking robustly raises decodability
Every thinking condition sits above no_think at essentially every layer (thinking ~0.67–0.75 vs
no_think ~0.49–0.64). The model's correctness becomes more internally decodable once it has produced
thinking tokens — it "knows it's right better after thinking," consistent across depth (so not a
best-layer fluke, even though individual best-layer CIs overlap no_think's).

### Finding 3 — but it is NOT coherent reasoning (hypothesis falsified)
Shuffled thinking **matches or exceeds** real thinking on separability at both budgets and across
all layers (shuffle_512 ≥ think_512; shuffle_1024 is the single highest curve). The pre-registered
hypothesis — real > shuffled would isolate genuine reasoning — is refuted. Whatever makes correctness
more decodable after thinking is the **presence/length/compute of the thinking region, not its
coherent content**. This converges with the behavioral shuffle finding (C9: behavioral
shuffled-thinking reproduced much of the accuracy gain) — now mirrored at the representational level.

## Controls

Shuffled-label probes give AUC 0.46–0.52 (≈ chance) in every condition → the probes are not
exploiting task-identity or pipeline leakage (GroupKFold splits by task). The real-vs-shuffled
comparison is the mechanism control; the consistent shuffle ≥ real ordering across depth is the
load-bearing result (individual best-layer CIs overlap, so we claim "real does **not** exceed
shuffled", not "shuffled is significantly higher").

## Oracle Versus Deployable Evidence

The probe is trained on hidden-test pass labels and is a non-deployable **decodability diagnostic**.
The deployable angle is the **visible-passer** column: among candidates that pass the *visible* test
(what the budget controller commits), can the probe rank true full-test passes above the C2
false-passes? Under thinking the probe has weak-moderate signal (AUC ~0.60–0.68); under no-think it
is ~chance (0.518). So an internal probe can *partially* flag the C2 false-passes the visible test
misses — but only once the model has thought, and not strongly enough to be a standalone selector.

## Interpretation

This is the cleanest evidence yet that, for this 4B on this task, native thinking's benefit is
largely **not coherent reasoning** — the conclusion now holds at two independent levels: behavioral
accuracy (C9 shuffle control) and internal representation (this probe). Thinking does make the model
more internally "aware" of its own correctness, but scrambled thinking does so equally, so the active
ingredient is compute/scaffold/token-presence. The result also yields a weak verifier-free signal
(internal probe vs C2 false-passes) that only appears under thinking.

### Limitations
- AUCs are moderate (0.64–0.76), n=100 single seed, and best-layer real-vs-shuffle differences are
  within overlapping CIs; the robust claim is the across-depth *ordering* (shuffle ≥ real, thinking >
  no-think), not precise magnitudes.
- Probes read only the last token; other positions/poolings (e.g. mean over the answer, or a Kadavath
  P(IK) probe token) could differ.
- One model, one benchmark; MBPP is coverage-limited and likely partly in pretraining (see the
  contamination direction in the program backlog).

## Next Experiments

- Foreign-task-thinking arm (remove token-presence, not just order): does it drop separability below
  shuffled? If even foreign thinking holds separability up, the active ingredient is pure compute.
- Probe-position sweep (post-`</think>`, mean-pooled answer, P(IK) probe token) and an MLP probe.
- Combine the internal probe with the visible test as a controller signal (does it lift the
  controller past the 0.91 deployable / 0.93 oracle wall?).
- Replicate on a contamination-controlled / harder substrate where the no-think baseline is weaker
  (more headroom for a reasoning-content effect to appear, if it exists).

## Artifact Manifest

See `artifact_manifest.yaml`. Activations (~0.7 GB) are external/regenerable; small records, labels,
probe results, and the figure are in-repo.

# Qwen3.5-4B Simulation Keystone Repair Report

## Summary

The intervention test of C13's causal claim, fully pre-registered (`reports/prereg.md`). C13 diagnosed one
broken primitive — multi-step mental simulation — under every inverse capability. We **repaired the
primitive and watched whether the untrained capabilities moved.** They did not. (Phase 0) The gate passed:
simulation is broken in isolation (0.96→0.30 by depth 4), and — refining C13's P12 — thinking *helps*
single-pipeline simulation (it is length-fragile, not globally wrong). (Phase 1–2) QLoRA-SFT on
interpreter-generated state-chain traces **fully repaired the simulator** — 0.80–0.84 through depth 5,
length-generalizing +54pp beyond trained depths, transferring to held-out primitives (0.42→0.85) — yet
the **inverse-capability ladder did not move**: bare identification 0.08→0.09, segmented 0.14→0.17. The
matched-token control (PROD: direct I/O→code training) moved segmented identification 3× (0.14→0.41 —
format-adjacent transfer) while degrading transcription (0.93→0.72), and both adapters crashed thinking-
2AFC via **format capture** (answering in trained format instead of A/B; verified on raw generations).
**Verdict per the locked decision rules: KEYSTONE REFUTED — separable-representation branch.**
Capability in a fixed small model is organized by **input→output format mappings, not shared internal
primitives**: repairing the "underlying" skill does not propagate, transfer follows format adjacency, and
narrow-format SFT taxes unrelated instruction-following. Mechanistic diagnoses do not license
training-transfer predictions.

## Research Program Fit

`structured_execution_and_compilers` + `posttraining_and_adaptation`. Executes C13's next_tests #1–2.
Every branch of the pre-registered outcome matrix was a durable law; the realized branch (separable
representation + format locality) directly bounds the whole banking program (C11/C12) and the mission's
"train broken primitives" hope.

## Method

- **Phase 0** (gate): frozen simulator microbenchmark — stated pipeline + one input → write the full state
  chain, no code. d 1–5 × k {0,2}, n=25/cell, no-think + thinking. Kill condition: d4 ≥ 0.8.
- **Phase 1**: matched-token QLoRA arms from base (~230k tokens each, identical hyperparams): **SIM** =
  pipeline+input→chain (depths 1–3 only; 3 primitives held out); **PROD** = I/O examples→reference code
  (direct end-task training; same unlimited generator ground truth — the only difference is supervised
  *content*).
- **Phase 2**: all three models on (a) simulation (in-distribution, length-gen d4–5, held-out primitives)
  and (b) the five-rung C13 ladder on fresh verified tasks (bare, plan-given, segmented, 2AFC no-think,
  2AFC thinking). All predictions/decision rules locked in advance.

## Results

### Phase 0 — gate passed; P-K0b refuted (C13 refinement)

Output exact-match by depth — no-think: 0.84/0.52/0.46/0.30/0.16; thinking: 0.96/0.88/0.58/0.30/0.36.
Simulation is broken in isolation (gate passes, P-K0a ✓), but thinking HELPS it at every depth (P-K0b ✗):
deliberate single-pipeline simulation is **length-fragile, not globally wrong** — P12's chance-level 2AFC
reflects the double-simulation + comparison load, not simulation per se.

### Phase 2a — the simulator is repaired (P-K1 ~, P-K2 ✓✓, P-K6 ✓✓)

| simulation (thinking) | d1 | d2 | d3 | d4* | d5* | held-out prims |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.96 | 0.88 | 0.58 | 0.30 | 0.36 | 0.42 |
| **SIM** | 0.92 | 0.82 | **0.80** | **0.84** | **0.76** | **0.85** |
| PROD | 0.88 | 0.98 | 0.88 | 0.60 | 0.36 | 0.59 |

(*beyond SIM's trained depth.) P-K1's letter (+30pp at d3) just missed (+22pp) but the repair is massive
and **strongest beyond trained depths** (+54pp at d4 — P-K2 confirmed); held-out-primitive transfer is
nearly full (P-K6) — the model learned chain-simulation as a *skill*. (PROD also lifts simulation
in-distribution — substrate exposure — but not at length and much less on held-out primitives.)

### Phase 2b — the ladder does not move (P-K3 ✗, P-K4 inverted)

| model | bare | segmented | 2AFC no-think | 2AFC thinking | plan-given |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.08 | 0.14 | 0.75 | 0.47 | 0.93 |
| **SIM** | 0.09 | 0.17 | 0.78 | 0.10 | 0.90 |
| PROD | 0.13 | **0.41** | 0.82 | 0.15 | **0.72** |

- **SIM (repaired simulator): every inverse rung flat.** The primitive works; nothing downstream noticed.
- **PROD moved segmented 3×** (d3k0 0.20→0.65, d4k0 0.00→0.45): transfer exists but follows **format
  adjacency** (segmented shares PROD's I/O→code output format), not primitive dependency. Bare
  identification at depth stays dead for both.
- **Both adapters crash thinking-2AFC (0.10–0.15, below chance) via format capture** — raw generations
  show the SIM model answering the A/B question with ```python code blocks. Narrow-format SFT installs
  output-mode priors that override unrelated instructions (and PROD's transcription drop 0.93→0.72 is the
  same tax).

## Controls

Matched training tokens (±2%); identical hyperparams; PROD as content-control (same ground-truth regime,
different supervised content); held-out primitives excluded from both arms; fresh verified ladder tasks;
format-capture verified on raw generations before interpreting the 2AFC crash; all predictions and
decision rules pre-registered.

## Oracle Versus Deployable Evidence

All behavioral; simulation graded by exact match against interpreter ground truth; ladder hidden-graded on
fresh verified tasks.

## Interpretation

**Capability in this fixed small model is organized by input→output format mappings, not by shared
internal primitives.** Three mutually reinforcing observations: (1) repairing the hypothesized keystone —
genuinely, with length generalization and skill-level transfer *within* the format — moves nothing outside
its format; (2) the only cross-task transfer observed follows format adjacency (PROD→segmented); (3) SFT's
strongest side effect is format capture — output-mode priors that damage unrelated instruction-following.
Consequences: (a) C13's diagnosis stands, but its causal reading ("fix simulation → fix inverse tasks")
is refuted — **mechanism diagnoses do not license training-transfer predictions**; (b) the banking program
(C11/C12) is format-local: it teaches mappings, not components — which retroactively explains why banking
never moved the planner (C12) and why production-SFT never improved verification; (c) the mission lesson —
for a fixed small model, eliciting a "skill" via SFT buys exactly the trained mapping plus its format
neighborhood, nothing more. The efficient strategy remains C13's: **externalize the missing primitive with
tools** rather than trying to install it and hoping it propagates.

### Limitations
One substrate family; QLoRA (r32) not full fine-tuning — conceivably full FT propagates differently;
single training run per arm; 2AFC format capture means that rung measures instruction-robustness, not
discrimination, for the adapters; P-K1's d3 letter missed by 8pp (d4/d5 vastly exceeded it).

## Next Experiments

- Mixed-format SIM training (chains + A/B + code in one adapter) — does format diversity prevent capture
  and unlock cross-format use of the repaired simulator?
- Explicit composition: prompt the SIM model to *use* its repaired simulation inside identification
  ("simulate candidate pipelines, compare") — can prompting bridge what SFT does not?
- Full-FT vs QLoRA on the same design (is separability an adapter artifact?).

## Artifact Manifest

See `artifact_manifest.yaml`. Adapters (~170MB ×2) regenerable via `scripts/phase12_chain.sh`; per-task
results + prereg + figure in-repo.

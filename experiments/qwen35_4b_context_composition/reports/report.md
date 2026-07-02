# Qwen3.5-4B Context Composition Report

## Summary

The third capability-installation mechanism, pre-registered (`reports/prereg.md`). C14 showed WEIGHTS
install capability format-locally (a fully repaired simulator moved nothing downstream). Here we test
CONTEXT: explicit orchestration and few-shot demonstration on the same verified tasks, same decoys, base
vs the SIM adapter. Four findings. (1) **Context composes discrimination**: an explicit
simulate-both-compare procedure lifts base 2AFC to 0.83 (parse 1.00), flat through depth 4 — where the
plain condition sits at 0.74. (2) **The weight-installed module IS accessible in-context**: under the
identical procedure the SIM adapter reaches **0.95 parse-conditional** (+12pp over base) — the trained
simulator genuinely composes — **but format capture gates the interface** (parse rate 0.53: half its
generations answer in trained format instead of `Answer: X`), crushing deployable accuracy to 0.51. (3)
**Hypothesis generation is the un-composable wall**: no context strategy moves bare identification
(base 0.08, SIM 0.13) — procedure, demonstrations, and a working simulator all fail to help the model
*propose* hypotheses. (4) **Retro-correction**: the keystone's "thinking-2AFC at chance" (P12) was
inflated by budget-512 + a weak first-char parser; with budget 1024 and a strict answer format, base
thinking-2AFC ≈ the no-think logit read. Net insight (C15): **deployable capability = module × interface
× procedure** — weights install modules but capture interfaces; context supplies procedures but cannot
create generators; only tools cross the generation wall.

## Research Program Fit

`structured_execution_and_compilers` + `posttraining_and_adaptation`. Completes the
installation-mechanism triptych (tools C12/C13, weights C14, context C15) and executes C14's next_tests
#1–2 (prompt-bridging; format-capture characterization).

## Method

Same 120 verified ladder tasks (d {2,3,4} × k {0,2}) and identical 2AFC items/decoys (fixed seed 4242) as
`qwen35_4b_simulation_keystone_repair`. SIM adapter regenerated from the committed recipe. Conditions —
2AFC (greedy, thinking budget 1024, strict `Answer: A/B` format, last-match parsing): plain (budget
control), ORCHESTRATED (stepwise simulate-both-compare procedure, `Step i: [...]` lines), ICL (two
programmatically-constructed worked examples, disjoint tasks); identification: orchestrated
generate-and-test (propose → simulate stepwise → check → revise → emit code), pass@2.

## Results

| 2AFC | raw | parse | parse-conditional |
| --- | ---: | ---: | ---: |
| base plain@1024 | 0.74 | 0.94 | 0.79 |
| base orchestrated | **0.83** | 1.00 | 0.83 |
| base ICL | 0.78 | 0.97 | 0.81 |
| SIM plain@1024 | 0.46 | 0.53 | 0.87 |
| SIM orchestrated | 0.51 | 0.53 | **0.95** |

Identification: base gen-and-test 0.08 (= bare 0.08); SIM gen-and-test 0.13 (bare 0.09).

- **P-C1** (SIM+orch ≥ 0.70): REFUTED on raw (0.51) — but 0.95 parse-conditional. The star cell splits
  along the module/interface distinction the prereg didn't anticipate.
- **P-C2** (orchestration helps base, gain shrinking with depth): direction confirmed (0.74→0.83) but the
  gain does NOT shrink — 0.85 at d4. Discrimination needs only partial simulation (the pipelines differ in
  one op), so the procedure stays viable at depth.
- **P-C3** (ICL < +0.10): CONFIRMED (+0.04).
- **P-C4** (gen-and-test lifts < 2×): CONFIRMED — no material lift for either model.
- **P-C5** (interaction): raw A5−A2 = −0.32 (interface capture dominates); parse-conditional +0.12 (the
  module adds real capability when invocable). Both facts are the finding.

## Controls

Same tasks/decoys/seed as the keystone (paired); budget control (plain@1024) separates budget from
procedure; parse rates reported everywhere; ICL demos constructed programmatically on disjoint tasks;
identification graded hidden, code-extracted as before.

## Oracle Versus Deployable Evidence

Raw accuracy is deployable; parse-conditional is diagnostic (module capability given a working interface).
The gap between them (0.51 vs 0.95) is itself the finding: format capture is an interface failure, not a
capability failure.

## Interpretation

The installation-mechanism triptych, completed:

| mechanism | installs | fails at |
| --- | --- | --- |
| tools (C12/C13) | search + simulation externally | nothing measured — but costs interpreter calls |
| weights/SFT (C14) | modules, format-locally | interface capture; no implicit propagation |
| context (C15) | procedures (composition recipes) | cannot create *generators* (hypothesis proposal) |

**Deployable capability = module × interface × procedure.** The keystone's "sealed modules" softens: the
module was never sealed — its *output channel* was hijacked. And the deepest wall of the arc sharpens:
everything except **hypothesis generation** can now be installed or composed by some mechanism; proposing
candidate programs from behavior remains untouched by weights, context, and procedure alike — only
external enumeration (tools) crosses it. C13's deployment rule gets its final form: *let tools generate,
let context orchestrate, let the model simulate-and-transcribe.*

### Limitations
Parse-conditional accuracy conditions on a non-random half of SIM generations (tasks where the format
survives may be easier — treat 0.95 as an upper bound); one substrate; QLoRA adapter; P12 correction rests
on a prompt+budget+parser change measured jointly, not factorially.

## Next Experiments

- Interface repair: brief mixed-format SFT (chains + `Answer: X` + code) — does 10% format diversity
  restore parse rates and make the 0.95 deployable?
- Generation-wall autopsy: why can't proposal be composed? (Constrained candidate menus vs free proposal.)
- Factorial P12 re-measurement (budget × parser × format) to close the correction cleanly.

## Artifact Manifest

See `artifact_manifest.yaml`. Adapter regenerable; per-task records + prereg + figure in-repo.

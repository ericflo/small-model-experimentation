# Qwen3.5-4B Context Composition

## Research Program

- Program: `structured_execution_and_compilers` (+ `posttraining_and_adaptation`). Pre-registered
  (`reports/prereg.md`). The **third capability-installation mechanism**: C12/C13 measured TOOLS
  (externalization works), C14 measured WEIGHTS (SFT is format-local, doesn't compose). This measures
  CONTEXT: can explicit orchestration or few-shot demonstration compose what weight-training cannot?

## Design

Same 120 verified ladder tasks + same 2AFC decoys/seed as the keystone experiment (exact comparability).
Conditions: {base, SIM adapter (regenerated; simulator verified 0.8+ to depth 5)} ×
{plain-think@1024 (budget control), ORCHESTRATED (explicit simulate-both-compare procedure),
ICL (2 worked examples)} for 2AFC; orchestrated generate-and-test for bare identification.

## Results (see reports/report.md; figure analysis/context_composition.png)

| 2AFC | raw | parse-rate | parse-conditional |
| --- | ---: | ---: | ---: |
| base plain@1024 | 0.74 | 0.94 | 0.79 |
| base ORCHESTRATED | **0.83** | 1.00 | 0.83 |
| base ICL | 0.78 | 0.97 | 0.81 |
| SIM plain@1024 | 0.46 | 0.53 | 0.87 |
| SIM ORCHESTRATED | 0.51 | **0.53** | **0.95** |

Identification (all context strategies): base 0.08, SIM 0.13 — unmoved.

1. **Context composes discrimination**: the explicit procedure lifts base to 0.83, flat through depth 4.
2. **The weight-installed module IS accessible in-context and adds capability** — SIM+orchestrated hits
   **0.95 parse-conditional** (vs base 0.83, same procedure) — **but format capture gates the interface**
   (parse 0.53), crushing deployable accuracy to 0.51. *The module composes; the interface is captured.*
3. **Hypothesis GENERATION is the un-composable wall**: no context strategy (procedure, demos, working
   simulator) moves bare identification (0.08–0.13). Inverse inference cannot be assembled in-context
   from forward primitives.
4. **Retro-correction of P12** (keystone/C13): "thinking-2AFC at chance" was inflated by budget-512 + a
   weak first-char parser; at budget 1024 with strict answer format, base thinking-2AFC ≈ the no-think
   logit read (0.74–0.79 vs 0.73–0.78). Thinking doesn't hurt discrimination; it just doesn't beat the
   surface heuristic without an explicit procedure.

**Insight (claim C15): the three installation mechanisms have distinct failure modes** — weights install
capability but capture the interface; context composes procedures but cannot create generators; tools
alone cross the generation wall. Deployable capability = module × interface × procedure.

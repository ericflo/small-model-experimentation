# Structured Execution And Compilers

## Purpose

Discover when small models should stop emitting direct answers and instead emit, condition on, or be supervised through executable structure: typed slots, latent registers, bytecode, candidate programs, state traces, compiler heads, or differentiable runtimes.

## Why This Is A Program

The imported experiments contain many successful variants, but the line is much broader than those prototypes. Future work should compare representations, supervision density, curricula, model sizes, and task substrates under a shared question: what structure turns small-model brittleness into reliable execution?

## Progress Signals

- Longer-horizon execution improves without beam search or oracle selection.
- Paraphrase and distribution-shift splits preserve the same executable trace.
- Program/state supervision explains gains beyond raw data scale.
- Failures can be localized by operator, step, state prefix, or representation.

## Boundaries

This program studies representation and execution. Selection between many candidates belongs primarily to Evidence-Conditioned Selection. Reusable banks of primitives belong primarily to Operator And Skill Inventories.

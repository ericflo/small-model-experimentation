# Process Control And Tool Use

## Purpose

Use small models as controllers over tools, verifiers, budgets, repair loops, and intermediate actions rather than single-shot answer generators.

## Why This Is A Program

Many future lines will need models that decide what to do next: call a tool, gather evidence, stop, repair, search, or commit. That deserves a program-level treatment across substrates.

## Progress Signals

- Controllers improve utility at fixed budgets.
- Tool-state representations transfer across pools or tasks.
- STOP/MORE and commit/repair decisions are calibrated.
- Policies remain useful when oracle labels are removed.

## Boundaries

This line owns process decisions. It often collaborates with Active Evidence Acquisition and Evidence-Conditioned Selection.

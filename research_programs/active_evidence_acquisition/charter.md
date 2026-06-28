# Active Evidence Acquisition

## Purpose

Choose or synthesize the next example, probe, trace, query, test, or tool action that most reduces uncertainty for a small model or selector.

## Why This Is A Program

The initial active-learning experiments are prototypes for a much broader loop: systems should spend scarce evidence budget where it changes decisions, not where it merely adds context.

## Progress Signals

- A budgeted policy beats random, order, diversity-only, and oracle-naive baselines.
- The selected evidence improves downstream decisions, not just intermediate confidence.
- Acquisition policies transfer across families or expose why they do not.
- Generated probes help without hidden expected answers.

## Boundaries

This program owns evidence gathering. The final commit/abstain decision belongs to Evidence-Conditioned Selection.

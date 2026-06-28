# Experiment Lifecycle

## 0. Choose The Program

Every new experiment must advance an existing research program or justify a new one. Start from `research_programs/README.md` and `knowledge/research_program_index.md`.

If no current program fits, create one before adding the experiment:

```bash
make new-program PROGRAM=<program-id> TITLE="<Title>" FOCUS="<one-sentence focus>"
```

## 1. Claim The Question

Create the experiment scaffold before the expensive run:

```bash
make new-experiment EXPERIMENT=<id> PROGRAM=<program-id> TITLE="<Title>"
```

Keep the question narrow enough that a single result can move belief.

The README should include:

- Research program.
- Question.
- Hypothesis.
- Baseline and controls.
- Dataset or task source.
- Smoke command.
- Full command.
- Expected primary metrics.
- Artifact plan.

## 2. Make The Smoke Path Real

Run a small version first. Save the smoke config and enough output to prove the path works. A smoke result is not evidence for the hypothesis, but it is evidence that the run path is alive.

## 3. Run With Controls

Prefer controls that test the mechanism:

- shuffled labels or shuffled retrieval queries,
- random retrieval or random acquisition,
- frozen-model versus trained-model comparisons,
- public-test versus hidden-test separation,
- oracle ceilings clearly labeled as non-deployable.

## 4. Analyze

Put derived outputs under `analysis/` and final narrative under `reports/`. Preserve raw run outputs when they are small enough or write a manifest for external artifacts.

## 5. Update Shared Knowledge

Run:

```bash
make check
```

Then update the human knowledge pages when the result changes strategy.

Also update the owning program's `evidence.md` and `backlog.md` when the result changes what should be tried next.

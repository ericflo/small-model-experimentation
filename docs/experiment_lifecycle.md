# Experiment Lifecycle

## 1. Claim The Question

Create `experiments/<id>/README.md` before the expensive run. Keep the question narrow enough that a single result can move belief.

The README should include:

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
make catalog
make validate
```

Then update the human knowledge pages when the result changes strategy.


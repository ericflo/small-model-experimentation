# Experiment Title

## Research Program

- Program:
- Program question:
- Prior anchors:

## Question

What specific uncertainty does this experiment resolve?

## Hypothesis

State the mechanism you expect to work and why it should beat the baseline.

## Setup

- Model:
- Dataset/task source:
- Train/eval split:
- Baseline:
- Controls:
- Primary metric:
- Oracle-only metrics:
- Hidden-label boundary:

## Run

Smoke:

```bash
python scripts/run.py --smoke
```

Full:

```bash
python scripts/run.py
```

## Results

Fill this after the run. Separate deployable evidence from oracle/hidden evaluation.

## Interpretation

What changed after this result? What is now more likely, less likely, or still unknown?

## Knowledgebase Update

- Program evidence updated:
- Program backlog updated:
- Claim ledger updated:

## Artifacts

- `src/`
- `scripts/`
- `configs/`
- `data/`
- `runs/`
- `analysis/`
- `reports/`

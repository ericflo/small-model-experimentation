# Deeper analysis (main)

## Deployable greedy flips vs no_think

| budget | greedy@1 | fail→pass | pass→fail | net |
| --- | ---: | ---: | ---: | ---: |
| no_think | 0.760 | – | – | – |
| think_256 | 0.870 | 15 | 4 | +11 |
| think_512 | 0.870 | 16 | 5 | +11 |
| think_1024 | 0.910 | 17 | 2 | +15 |
| think_2048 | 0.860 | 13 | 3 | +10 |
| think_unbudgeted | 0.840 | 13 | 5 | +8 |

## Difficulty slices (by no_think oracle pass@8)

Greedy@1 within each slice, no_think vs best thinking budget.

| slice | n | no_think greedy | best-think greedy | Δ |
| --- | ---: | ---: | ---: | ---: |
| always (8/8) | 51 | 0.941 | 1.000 | +0.059 |
| sometimes (1-7/8) | 40 | 0.700 | 0.925 | +0.225 |
| never (0/8) | 9 | 0.000 | 0.333 | +0.333 |

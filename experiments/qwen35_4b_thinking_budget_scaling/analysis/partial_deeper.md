# Deeper analysis (partial)

## Deployable greedy flips vs no_think

| budget | greedy@1 | fail→pass | pass→fail | net |
| --- | ---: | ---: | ---: | ---: |
| no_think | 0.700 | – | – | – |
| think_256 | 0.800 | 19 | 9 | +10 |
| think_512 | 0.790 | 19 | 10 | +9 |
| think_1024 | 0.830 | 22 | 9 | +13 |
| think_2048 | 0.780 | 16 | 8 | +8 |

## Difficulty slices (by no_think oracle pass@8)

Greedy@1 within each slice, no_think vs best thinking budget.

| slice | n | no_think greedy | best-think greedy | Δ |
| --- | ---: | ---: | ---: | ---: |
| always (8/8) | 20 | 0.950 | 1.000 | +0.050 |
| sometimes (1-7/8) | 69 | 0.739 | 0.855 | +0.116 |
| never (0/8) | 11 | 0.000 | 0.455 | +0.455 |

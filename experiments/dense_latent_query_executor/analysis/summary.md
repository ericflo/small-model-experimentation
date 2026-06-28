# Dense Latent Query Executor Analysis Summary

This summary is generated from `runs/*/metrics_final.csv`.

## Main Modulus-31 Threshold

| L | best query mass K<L | best probe belief mass K<L | first K>=L | query mass | probe belief mass | query top1 |
|---|---|---|---|---|---|---|
| 4 | 44.9% | 11.3% | 4 | 70.2% | 41.6% | 79.1% |
| 8 | 21.8% | 2.5% | 8 | 49.3% | 18.8% | 59.3% |
| 12 | 14.9% | 1.4% | 12 | 36.1% | 12.0% | 44.6% |
| 16 | 10.8% | 0.8% | 16 | 29.6% | 8.8% | 37.1% |
| 24 | 6.0% | 0.3% | 24 | 21.5% | 5.7% | 28.8% |

## Main Modulus-31 Per-Query Threshold

| L | query | first K>=L | query mass | probe belief mass | query top1 |
|---|---|---|---|---|---|
| 4 | A | 4 | 73.9% | 40.8% | 82.0% |
| 4 | A_MINUS_B | 4 | 66.3% | 41.7% | 77.0% |
| 4 | A_PLUS_B | 4 | 63.3% | 41.7% | 71.1% |
| 4 | B | 4 | 77.3% | 42.2% | 86.3% |
| 8 | A | 8 | 52.9% | 18.7% | 64.6% |
| 8 | A_MINUS_B | 8 | 46.2% | 18.3% | 56.6% |
| 8 | A_PLUS_B | 8 | 44.4% | 20.0% | 52.7% |
| 8 | B | 8 | 53.7% | 18.3% | 63.1% |
| 12 | A | 12 | 38.8% | 12.1% | 46.9% |
| 12 | A_MINUS_B | 12 | 34.4% | 12.2% | 41.4% |
| 12 | A_PLUS_B | 12 | 31.1% | 11.9% | 41.6% |
| 12 | B | 12 | 40.0% | 11.8% | 48.4% |
| 16 | A | 16 | 32.8% | 8.5% | 41.2% |
| 16 | A_MINUS_B | 16 | 27.4% | 9.4% | 34.0% |
| 16 | A_PLUS_B | 16 | 24.6% | 9.0% | 31.2% |
| 16 | B | 16 | 33.6% | 8.3% | 42.0% |
| 24 | A | 24 | 24.0% | 5.5% | 31.4% |
| 24 | A_MINUS_B | 24 | 19.2% | 5.7% | 23.2% |
| 24 | A_PLUS_B | 24 | 17.5% | 6.0% | 26.2% |
| 24 | B | 24 | 25.4% | 5.7% | 34.2% |

## Modulus-31 Controls

| model | L | k | query mass | query top1 | probe belief mass | probe belief top1 |
|---|---|---|---|---|---|---|
| Static compiler p=31 | 4 | -1 | 50.4% | 54.8% | 13.0% | 17.3% |
| Static compiler p=31 | 8 | -1 | 26.9% | 29.1% | 3.7% | 4.5% |
| Static compiler p=31 | 12 | -1 | 13.6% | 11.8% | 0.8% | 0.5% |
| Static compiler p=31 | 16 | -1 | 9.0% | 9.5% | 0.4% | 0.1% |
| Static compiler p=31 | 24 | -1 | 5.5% | 5.6% | 0.2% | 0.1% |
| Dense recurrent p=31 | 4 | 4 | 70.2% | 79.1% | 41.6% | 55.5% |
| Dense recurrent p=31 | 8 | 8 | 49.3% | 59.3% | 18.8% | 26.2% |
| Dense recurrent p=31 | 12 | 12 | 36.1% | 44.6% | 12.0% | 18.9% |
| Dense recurrent p=31 | 16 | 16 | 29.6% | 37.1% | 8.8% | 13.1% |
| Dense recurrent p=31 | 24 | 24 | 21.5% | 28.8% | 5.7% | 7.9% |

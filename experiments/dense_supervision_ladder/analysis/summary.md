# Dense Supervision Ladder Analysis Summary

This summary is generated from `runs/*/metrics_final.csv`.

## Modulus 31 Ladder at K=L

| supervision | L | K | query mass | query top1 | probe belief | decoder belief |
|---|---|---|---|---|---|---|
| Sampled final | 4 | 4 | 51.0% | 55.9% | 12.9% | 1.4% |
| Sampled final | 8 | 8 | 28.2% | 31.7% | 4.0% | 0.7% |
| Sampled final | 12 | 12 | 20.5% | 23.5% | 2.7% | 0.4% |
| Sampled final | 16 | 16 | 13.6% | 14.9% | 1.8% | 0.3% |
| Sampled final | 24 | 24 | 9.4% | 12.5% | 1.2% | 0.2% |
| Soft final query | 4 | 4 | 62.0% | 69.7% | 21.3% | 1.4% |
| Soft final query | 8 | 8 | 39.4% | 47.0% | 9.1% | 0.7% |
| Soft final query | 12 | 12 | 30.2% | 38.2% | 6.2% | 0.4% |
| Soft final query | 16 | 16 | 22.2% | 28.4% | 4.7% | 0.3% |
| Soft final query | 24 | 24 | 16.1% | 20.7% | 3.2% | 0.2% |
| Prefix query | 4 | 4 | 72.8% | 80.5% | 39.5% | 1.4% |
| Prefix query | 8 | 8 | 46.3% | 56.5% | 15.3% | 0.7% |
| Prefix query | 12 | 12 | 34.1% | 39.7% | 8.8% | 0.4% |
| Prefix query | 16 | 16 | 25.8% | 32.7% | 6.2% | 0.3% |
| Prefix query | 24 | 24 | 18.5% | 23.1% | 4.3% | 0.2% |
| Sparse belief | 4 | 4 | 66.7% | 74.4% | 29.4% | 26.8% |
| Sparse belief | 8 | 8 | 42.4% | 51.6% | 11.8% | 10.7% |
| Sparse belief | 12 | 12 | 31.5% | 35.7% | 7.0% | 6.5% |
| Sparse belief | 16 | 16 | 22.8% | 29.2% | 4.8% | 4.4% |
| Sparse belief | 24 | 24 | 16.2% | 20.7% | 3.2% | 3.0% |
| Full belief | 4 | 4 | 74.1% | 83.1% | 49.4% | 51.4% |
| Full belief | 8 | 8 | 49.7% | 58.7% | 21.5% | 21.7% |
| Full belief | 12 | 12 | 35.8% | 42.6% | 11.8% | 11.3% |
| Full belief | 16 | 16 | 27.3% | 32.8% | 7.9% | 7.4% |
| Full belief | 24 | 24 | 19.4% | 24.8% | 4.8% | 4.6% |

## Modulus 11 Pilot at K=L

| supervision | L | K | query mass | query top1 | probe belief | decoder belief |
|---|---|---|---|---|---|---|
| Sampled final | 3 | 3 | 69.1% | 76.4% | 42.7% | 4.7% |
| Sampled final | 6 | 6 | 48.1% | 54.6% | 19.8% | 3.1% |
| Sampled final | 9 | 9 | 35.8% | 41.9% | 13.7% | 2.1% |
| Sampled final | 12 | 12 | 28.9% | 33.1% | 9.9% | 1.7% |
| Soft final query | 3 | 3 | 81.7% | 88.7% | 66.3% | 4.7% |
| Soft final query | 6 | 6 | 60.5% | 69.6% | 37.8% | 3.1% |
| Soft final query | 9 | 9 | 46.8% | 54.9% | 24.5% | 2.1% |
| Soft final query | 12 | 12 | 39.2% | 45.5% | 20.1% | 1.7% |
| Prefix query | 3 | 3 | 90.2% | 94.6% | 77.2% | 4.7% |
| Prefix query | 6 | 6 | 70.9% | 77.8% | 53.4% | 3.1% |
| Prefix query | 9 | 9 | 56.9% | 64.6% | 36.9% | 2.1% |
| Prefix query | 12 | 12 | 45.3% | 52.5% | 26.4% | 1.7% |
| Sparse belief | 3 | 3 | 86.8% | 94.6% | 76.2% | 70.7% |
| Sparse belief | 6 | 6 | 62.8% | 72.2% | 46.9% | 41.2% |
| Sparse belief | 9 | 9 | 47.8% | 57.0% | 29.5% | 25.0% |
| Sparse belief | 12 | 12 | 38.8% | 47.2% | 20.6% | 17.5% |
| Full belief | 3 | 3 | 90.9% | 95.8% | 84.6% | 85.6% |
| Full belief | 6 | 6 | 71.3% | 80.0% | 60.8% | 59.7% |
| Full belief | 9 | 9 | 57.7% | 66.2% | 43.2% | 41.5% |
| Full belief | 12 | 12 | 45.7% | 51.8% | 30.2% | 28.9% |

## Generated Figures

- `figures/mod31_ladder_query_mass_at_k_ge_l.png`
- `figures/mod31_ladder_probe_belief_mass_at_k_ge_l.png`
- `figures/mod31_ladder_decoder_belief_mass_at_k_ge_l.png`
- `figures/mod31_full_belief_query_mass_heatmap.png`
- `figures/mod31_full_belief_decoder_mass_heatmap.png`
- `figures/mod31_query_mass_by_k_<supervision>.png`

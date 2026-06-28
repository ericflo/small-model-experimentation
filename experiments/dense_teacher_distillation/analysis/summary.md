# Dense Teacher Distillation Analysis Summary

## Runs

| phase | modulus | variant | transition | decoder_type | state_dim | train_steps | train_max_len |
|---|---|---|---|---|---|---|---|
| main | 31 | main_gru_mlp_d256 | gru | mlp | 256 | 1500 | 8 |
| main | 31 | main_gru_mlp_d512 | gru | mlp | 512 | 1500 | 8 |
| main | 31 | main_residual_mlp_d512 | residual | mlp | 512 | 1500 | 8 |
| pilot | 11 | pilot_gru_lowrank_d256_r16 | gru | low_rank | 256 | 800 | 6 |
| pilot | 11 | pilot_gru_mlp_d128 | gru | mlp | 128 | 800 | 6 |
| pilot | 11 | pilot_gru_mlp_d256 | gru | mlp | 256 | 800 | 6 |
| pilot | 11 | pilot_residual_mlp_d256 | residual | mlp | 256 | 800 | 6 |
| smoke | 7 | smoke_gru_lowrank_d32_r4 | gru | low_rank | 32 | 2 | 3 |
| smoke | 7 | smoke_gru_mlp_d32 | gru | mlp | 32 | 2 | 3 |
| smoke | 7 | smoke_residual_mlp_d32 | residual | mlp | 32 | 2 | 3 |

## First K >= L Summary

| phase | modulus | variant | length | decoder_query_target_mass | decoder_belief_target_mass | probe_belief_target_mass | query_target_mass |
|---|---|---|---|---|---|---|---|
| main | 31 | main_gru_mlp_d256 | 4 | 69.9% | 32.6% | 30.6% | 37.4% |
| main | 31 | main_gru_mlp_d256 | 8 | 36.1% | 9.0% | 9.1% | 19.3% |
| main | 31 | main_gru_mlp_d256 | 12 | 23.7% | 4.0% | 4.3% | 11.9% |
| main | 31 | main_gru_mlp_d256 | 16 | 16.8% | 2.5% | 2.6% | 8.0% |
| main | 31 | main_gru_mlp_d256 | 24 | 11.7% | 1.6% | 1.6% | 5.2% |
| main | 31 | main_gru_mlp_d512 | 4 | 78.1% | 45.7% | 41.4% | 37.4% |
| main | 31 | main_gru_mlp_d512 | 8 | 49.2% | 19.3% | 18.3% | 19.4% |
| main | 31 | main_gru_mlp_d512 | 12 | 35.4% | 9.9% | 10.0% | 11.8% |
| main | 31 | main_gru_mlp_d512 | 16 | 26.3% | 6.8% | 6.8% | 8.0% |
| main | 31 | main_gru_mlp_d512 | 24 | 18.7% | 4.3% | 4.3% | 5.2% |
| main | 31 | main_residual_mlp_d512 | 4 | 80.5% | 50.1% | 44.5% | 37.4% |
| main | 31 | main_residual_mlp_d512 | 8 | 52.1% | 21.9% | 20.0% | 19.3% |
| main | 31 | main_residual_mlp_d512 | 12 | 36.5% | 11.5% | 10.6% | 11.8% |
| main | 31 | main_residual_mlp_d512 | 16 | 27.9% | 7.8% | 7.7% | 8.0% |
| main | 31 | main_residual_mlp_d512 | 24 | 20.5% | 5.1% | 5.1% | 5.2% |
| pilot | 11 | pilot_gru_lowrank_d256_r16 | 3 | 78.8% | 49.4% | 53.6% | 46.3% |
| pilot | 11 | pilot_gru_lowrank_d256_r16 | 6 | 50.3% | 21.2% | 23.0% | 29.7% |
| pilot | 11 | pilot_gru_lowrank_d256_r16 | 9 | 36.4% | 12.0% | 12.9% | 21.4% |
| pilot | 11 | pilot_gru_lowrank_d256_r16 | 12 | 28.5% | 8.9% | 9.5% | 17.1% |
| pilot | 11 | pilot_gru_mlp_d128 | 3 | 90.5% | 70.6% | 68.7% | 46.1% |
| pilot | 11 | pilot_gru_mlp_d128 | 6 | 61.9% | 37.8% | 36.5% | 29.6% |
| pilot | 11 | pilot_gru_mlp_d128 | 9 | 44.8% | 21.6% | 20.9% | 21.3% |
| pilot | 11 | pilot_gru_mlp_d128 | 12 | 35.2% | 15.0% | 15.2% | 16.9% |
| pilot | 11 | pilot_gru_mlp_d256 | 3 | 94.7% | 81.9% | 80.4% | 46.5% |
| pilot | 11 | pilot_gru_mlp_d256 | 6 | 71.3% | 52.9% | 52.7% | 29.7% |
| pilot | 11 | pilot_gru_mlp_d256 | 9 | 53.1% | 32.7% | 33.2% | 21.5% |
| pilot | 11 | pilot_gru_mlp_d256 | 12 | 41.4% | 22.0% | 21.9% | 17.2% |
| pilot | 11 | pilot_residual_mlp_d256 | 3 | 94.8% | 81.0% | 79.2% | 46.3% |
| pilot | 11 | pilot_residual_mlp_d256 | 6 | 71.6% | 52.8% | 51.9% | 29.7% |
| pilot | 11 | pilot_residual_mlp_d256 | 9 | 54.7% | 33.8% | 33.0% | 21.4% |
| pilot | 11 | pilot_residual_mlp_d256 | 12 | 42.6% | 22.5% | 22.4% | 17.1% |
| smoke | 7 | smoke_gru_lowrank_d32_r4 | 2 | 58.0% | 9.0% | 9.0% | 57.7% |
| smoke | 7 | smoke_gru_lowrank_d32_r4 | 3 | 45.2% | 7.6% | 7.6% | 45.3% |
| smoke | 7 | smoke_gru_mlp_d32 | 2 | 58.3% | 9.2% | 9.2% | 57.7% |
| smoke | 7 | smoke_gru_mlp_d32 | 3 | 45.0% | 7.7% | 7.7% | 45.4% |
| smoke | 7 | smoke_residual_mlp_d32 | 2 | 58.3% | 9.1% | 9.3% | 58.2% |
| smoke | 7 | smoke_residual_mlp_d32 | 3 | 45.2% | 7.6% | 7.7% | 44.6% |

## Files

- `all_metrics_long.csv`: per-query metrics loaded from each run.
- `final_metrics_query_mean.csv`: metrics averaged across query types.
- `threshold_summary.csv`: first available K satisfying K >= L.
- `figures/`: generated metric plots.

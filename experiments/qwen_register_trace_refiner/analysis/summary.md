# Qwen Register Trace Refiner Analysis Summary

Primary run: `main_register_trace_refiner_s512`

| split | base | prior | soft-trace | learned | guarded | pair-rerank | oracle | learned gap | guarded gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_len24 | 14.8% | 14.8% | 14.8% | 16.6% | 16.6% | n/a | 21.9% | 25.0% | 25.0% |
| val_len24 | 15.6% | 15.6% | 15.6% | 17.2% | 17.2% | n/a | 20.3% | 33.3% | 33.3% |
| fresh_standard_len24 | 23.4% | 23.4% | 23.4% | 26.6% | 26.6% | n/a | 37.1% | 22.9% | 22.9% |
| fresh_paraphrase_len24 | 4.7% | 4.7% | 4.7% | 4.7% | 4.7% | n/a | 7.0% | 0.0% | 0.0% |
| fresh_paired_len24 | 12.3% | 12.3% | 12.3% | 12.9% | 12.9% | 12.9% | 18.6% | 9.4% | 9.4% |

## Fresh Paired Details

| metric | base | learned | guarded | pair-rerank | oracle |
|---|---:|---:|---:|---:|---:|
| executor_accuracy | 12.3% | 12.9% | 12.9% | 12.9% | 18.6% |
| program_exact | 12.1% | 12.7% | 12.7% | 12.7% | 18.4% |
| state_prefix_fraction | 79.5% | 79.7% | 79.7% | 79.7% | 80.7% |
| pair_both_correct | 1.6% | 1.6% | 1.6% | 1.6% | 1.6% |
| pair_state_consistency | 1.6% | 1.6% | 1.6% | 1.6% | 1.6% |
| changed_fraction | 0.0% | 17.6% | 16.8% | 17.6% | 6.2% |
| avg_edits | 0.00 | 0.18 | 0.17 | 0.18 | 0.09 |

## Candidate Set

| split | candidates/example | positive candidates/example | oracle found |
|---|---:|---:|---:|
| train_len24 | 1299.0 | 0.45 | 21.9% |
| val_len24 | 1299.0 | 0.36 | 20.3% |
| fresh_standard_len24 | 1299.0 | 0.68 | 37.1% |
| fresh_paraphrase_len24 | 1299.0 | 0.15 | 7.0% |
| fresh_paired_len24 | 1299.0 | 0.47 | 18.6% |

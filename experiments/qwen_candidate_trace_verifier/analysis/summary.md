# Candidate-Trace Verifier Analysis Summary

Primary run: `main_trace_verifier_s512`

| split | base | trace verifier | pair rerank | oracle | gap recovered |
|---|---:|---:|---:|---:|---:|
| train_len24 | 28.7% | 58.6% | n/a | 86.7% | 51.5% |
| val_len24 | 32.0% | 56.2% | n/a | 85.2% | 45.6% |
| fresh_standard_len24 | 28.5% | 50.4% | n/a | 90.6% | 35.2% |
| fresh_paraphrase_len24 | 28.5% | 55.5% | n/a | 86.7% | 46.3% |
| fresh_paired_len24 | 30.3% | 53.7% | 56.2% | 88.1% | 40.5% |

## Fresh Paired Details

| metric | base | trace verifier | pair rerank | oracle |
|---|---:|---:|---:|---:|
| executor_accuracy | 30.3% | 53.7% | 56.2% | 88.1% |
| program_exact | 30.3% | 53.7% | 56.2% | 87.7% |
| state_prefix_fraction | 58.6% | 76.4% | 78.0% | 90.7% |
| pair_both_correct | 28.1% | 39.8% | 52.3% | 85.9% |
| pair_state_consistency | 71.1% | 58.2% | 87.1% | 92.6% |

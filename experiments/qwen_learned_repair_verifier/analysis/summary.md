# Learned Repair Verifier Analysis Summary

Primary run: `main_rich_learned_verifier_s512`

| split | base | learned | pair-rerank | oracle | gap recovered | learned changed |
|---|---:|---:|---:|---:|---:|---:|
| train_len24 | 28.7% | 58.8% | n/a | 86.7% | 51.9% | 46.1% |
| val_len24 | 32.0% | 50.8% | n/a | 85.2% | 35.3% | 48.4% |
| fresh_standard_len24 | 28.5% | 44.1% | n/a | 90.6% | 25.2% | 42.6% |
| fresh_paraphrase_len24 | 28.5% | 48.0% | n/a | 86.7% | 33.6% | 46.9% |
| fresh_paired_len24 | 30.3% | 47.3% | 51.0% | 88.1% | 29.4% | 49.6% |

## Paired Split Details

| metric | base | learned | pair-rerank | oracle |
|---|---:|---:|---:|---:|
| executor_accuracy | 30.3% | 47.3% | 51.0% | 88.1% |
| program_exact | 30.3% | 47.3% | 51.0% | 87.7% |
| state_prefix_fraction | 58.6% | 71.2% | 73.4% | 90.7% |
| pair_both_correct | 28.1% | 34.4% | 46.5% | 85.9% |
| pair_state_consistency | 71.1% | 55.1% | 82.8% | 92.6% |

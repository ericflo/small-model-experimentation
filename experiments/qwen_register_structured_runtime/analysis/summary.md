# Qwen Register-Token Structured Runtime Analysis Summary

Primary run: `main_structured_trace_state_consistency_s600`

## Final Metrics

| variant | split | direct | executor | mass | init | op | arg | program | prefix | pair both | pair state consistency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| structured_trace_state_consistency | standard_len4 | n/a | 100.0% | 96.2% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| structured_trace_state_consistency | standard_len8 | n/a | 100.0% | 95.8% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| structured_trace_state_consistency | standard_len12 | n/a | 100.0% | 94.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| structured_trace_state_consistency | standard_len24 | n/a | 25.0% | 21.8% | 100.0% | 93.8% | 89.7% | 25.0% | 80.5% | n/a | n/a |
| structured_trace_state_consistency | paraphrase_len4 | n/a | 100.0% | 96.9% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| structured_trace_state_consistency | paraphrase_len8 | n/a | 100.0% | 96.2% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| structured_trace_state_consistency | paraphrase_len12 | n/a | 99.2% | 93.7% | 100.0% | 99.9% | 99.9% | 99.2% | 99.7% | n/a | n/a |
| structured_trace_state_consistency | paraphrase_len24 | n/a | 5.5% | 5.4% | 100.0% | 88.9% | 83.8% | 4.7% | 81.0% | n/a | n/a |
| structured_trace_state_consistency | paired_len4 | n/a | 100.0% | 94.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| structured_trace_state_consistency | paired_len8 | n/a | 100.0% | 97.3% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| structured_trace_state_consistency | paired_len12 | n/a | 100.0% | 93.6% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| structured_trace_state_consistency | paired_len24 | n/a | 11.7% | 9.9% | 100.0% | 89.7% | 85.7% | 10.2% | 79.1% | 1.6% | 1.6% |

## All Runs

| run | variant | split | direct | executor | program | pair both |
|---|---|---|---:|---:|---:|---:|
| control_trace_state_no_pair_s600 | register_trace_state | standard_len24 | n/a | 3.9% | 2.3% | n/a |
| control_trace_state_no_pair_s600 | register_trace_state | paraphrase_len24 | n/a | 1.6% | 0.0% | n/a |
| control_trace_state_no_pair_s600 | register_trace_state | paired_len24 | n/a | 1.6% | 0.8% | 0.0% |
| main_structured_trace_state_consistency_s600 | structured_trace_state_consistency | standard_len24 | n/a | 25.0% | 25.0% | n/a |
| main_structured_trace_state_consistency_s600 | structured_trace_state_consistency | paraphrase_len24 | n/a | 5.5% | 4.7% | n/a |
| main_structured_trace_state_consistency_s600 | structured_trace_state_consistency | paired_len24 | n/a | 11.7% | 10.2% | 1.6% |
| pilot_structured_state_consistency_s240 | structured_trace_state_consistency | standard_len24 | n/a | 7.8% | 6.2% | n/a |
| pilot_structured_state_consistency_s240 | structured_trace_state_consistency | paraphrase_len24 | n/a | 9.4% | 7.8% | n/a |
| pilot_structured_state_consistency_s240 | structured_trace_state_consistency | paired_len24 | n/a | 9.4% | 7.8% | 3.1% |

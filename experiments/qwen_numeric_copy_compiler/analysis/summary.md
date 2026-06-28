# Qwen Numeric-Copy Compiler Analysis Summary

## Final Metrics

| run | variant | split | direct_accuracy | executor_accuracy | executor_target_mass | init_accuracy | init_pos_accuracy | op_accuracy | arg_accuracy | op_pos_accuracy | arg_pos_accuracy | program_exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control_qwen3_4b_direct_numeric_copy_distribution_l12 | direct | standard_len4 | 3.1% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_numeric_copy_distribution_l12 | direct | standard_len8 | 0.8% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_numeric_copy_distribution_l12 | direct | standard_len12 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_numeric_copy_distribution_l12 | direct | standard_len24 | 0.8% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_numeric_copy_distribution_l12 | direct | paraphrase_len4 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_numeric_copy_distribution_l12 | direct | paraphrase_len8 | 3.1% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_numeric_copy_distribution_l12 | direct | paraphrase_len12 | 3.9% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_numeric_copy_distribution_l12 | direct | paraphrase_len24 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_qlora_numeric_copy_answer_only_l12 | copy_answer_only | standard_len4 | n/a | 100.0% | 17.6% | 100.0% | 17.2% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| control_qwen3_4b_qlora_numeric_copy_answer_only_l12 | copy_answer_only | standard_len8 | n/a | 0.0% | 1.1% | 100.0% | 25.0% | 76.7% | 66.1% | 0.0% | 1.1% | 0.0% |
| control_qwen3_4b_qlora_numeric_copy_answer_only_l12 | copy_answer_only | standard_len12 | n/a | 1.6% | 1.0% | 100.0% | 16.4% | 74.9% | 64.6% | 0.0% | 7.8% | 0.0% |
| control_qwen3_4b_qlora_numeric_copy_answer_only_l12 | copy_answer_only | standard_len24 | n/a | 2.3% | 1.0% | 100.0% | 22.7% | 59.6% | 41.4% | 0.0% | 6.0% | 0.0% |
| control_qwen3_4b_qlora_numeric_copy_answer_only_l12 | copy_answer_only | paraphrase_len4 | n/a | 100.0% | 16.1% | 100.0% | 21.1% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| control_qwen3_4b_qlora_numeric_copy_answer_only_l12 | copy_answer_only | paraphrase_len8 | n/a | 7.0% | 1.2% | 100.0% | 27.3% | 84.7% | 78.4% | 0.0% | 4.2% | 6.2% |
| control_qwen3_4b_qlora_numeric_copy_answer_only_l12 | copy_answer_only | paraphrase_len12 | n/a | 1.6% | 1.0% | 100.0% | 23.4% | 81.8% | 77.9% | 0.0% | 13.8% | 0.0% |
| control_qwen3_4b_qlora_numeric_copy_answer_only_l12 | copy_answer_only | paraphrase_len24 | n/a | 0.0% | 1.0% | 100.0% | 22.7% | 62.2% | 45.6% | 0.0% | 7.7% | 0.0% |
| main_qwen3_4b_qlora_numeric_copy_trace_mixed_l12 | copy_trace | standard_len4 | n/a | 89.8% | 8.2% | 100.0% | 100.0% | 100.0% | 97.5% | 100.0% | 96.9% | 89.8% |
| main_qwen3_4b_qlora_numeric_copy_trace_mixed_l12 | copy_trace | standard_len8 | n/a | 72.7% | 1.2% | 100.0% | 100.0% | 100.0% | 96.1% | 100.0% | 95.7% | 72.7% |
| main_qwen3_4b_qlora_numeric_copy_trace_mixed_l12 | copy_trace | standard_len12 | n/a | 46.9% | 1.0% | 100.0% | 100.0% | 100.0% | 94.1% | 100.0% | 94.0% | 46.9% |
| main_qwen3_4b_qlora_numeric_copy_trace_mixed_l12 | copy_trace | standard_len24 | n/a | 20.3% | 1.0% | 100.0% | 100.0% | 99.9% | 92.9% | 99.8% | 95.3% | 18.0% |
| main_qwen3_4b_qlora_numeric_copy_trace_mixed_l12 | copy_trace | paraphrase_len4 | n/a | 85.9% | 7.2% | 100.0% | 100.0% | 100.0% | 96.5% | 100.0% | 95.1% | 85.9% |
| main_qwen3_4b_qlora_numeric_copy_trace_mixed_l12 | copy_trace | paraphrase_len8 | n/a | 63.3% | 1.3% | 100.0% | 100.0% | 100.0% | 94.5% | 100.0% | 93.6% | 63.3% |
| main_qwen3_4b_qlora_numeric_copy_trace_mixed_l12 | copy_trace | paraphrase_len12 | n/a | 46.1% | 1.0% | 100.0% | 100.0% | 100.0% | 93.6% | 100.0% | 92.6% | 46.1% |
| main_qwen3_4b_qlora_numeric_copy_trace_mixed_l12 | copy_trace | paraphrase_len24 | n/a | 5.5% | 1.0% | 100.0% | 100.0% | 97.4% | 85.8% | 96.5% | 84.9% | 5.5% |
| pilot_qwen3_4b_frozen_numeric_copy_trace_mixed_l12 | copy_trace | standard_len4 | n/a | 83.6% | 8.1% | 100.0% | 100.0% | 100.0% | 95.5% | 100.0% | 94.5% | 82.8% |
| pilot_qwen3_4b_frozen_numeric_copy_trace_mixed_l12 | copy_trace | standard_len8 | n/a | 46.1% | 1.2% | 100.0% | 100.0% | 100.0% | 90.2% | 100.0% | 88.5% | 45.3% |
| pilot_qwen3_4b_frozen_numeric_copy_trace_mixed_l12 | copy_trace | standard_len12 | n/a | 16.4% | 1.0% | 100.0% | 100.0% | 100.0% | 86.5% | 100.0% | 84.2% | 16.4% |
| pilot_qwen3_4b_frozen_numeric_copy_trace_mixed_l12 | copy_trace | standard_len24 | n/a | 1.6% | 1.0% | 100.0% | 100.0% | 93.9% | 77.6% | 91.5% | 75.3% | 0.0% |
| pilot_qwen3_4b_frozen_numeric_copy_trace_mixed_l12 | copy_trace | paraphrase_len4 | n/a | 87.5% | 7.2% | 100.0% | 100.0% | 100.0% | 96.9% | 100.0% | 94.1% | 87.5% |
| pilot_qwen3_4b_frozen_numeric_copy_trace_mixed_l12 | copy_trace | paraphrase_len8 | n/a | 68.8% | 1.3% | 100.0% | 100.0% | 100.0% | 95.4% | 100.0% | 91.5% | 68.8% |
| pilot_qwen3_4b_frozen_numeric_copy_trace_mixed_l12 | copy_trace | paraphrase_len12 | n/a | 46.1% | 1.0% | 100.0% | 100.0% | 100.0% | 93.7% | 100.0% | 90.4% | 45.3% |
| pilot_qwen3_4b_frozen_numeric_copy_trace_mixed_l12 | copy_trace | paraphrase_len24 | n/a | 14.8% | 1.0% | 100.0% | 100.0% | 99.5% | 90.2% | 99.8% | 88.2% | 13.3% |
| smoke_qwen3_4b_frozen_copy_trace | copy_trace | standard_len2 | n/a | 50.0% | 9.6% | 50.0% | 50.0% | 100.0% | 75.0% | 50.0% | 50.0% | 50.0% |
| smoke_tiny_frozen_copy_trace | copy_trace | standard_len2 | n/a | 12.5% | 19.3% | 37.5% | 0.0% | 62.5% | 12.5% | 0.0% | 0.0% | 0.0% |
| smoke_tiny_frozen_copy_trace | copy_trace | standard_len4 | n/a | 0.0% | 5.7% | 37.5% | 0.0% | 87.5% | 46.9% | 9.4% | 3.1% | 0.0% |

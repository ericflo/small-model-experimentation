# Qwen Shared Parser Compiler Analysis Summary

## Final Metrics

| run | variant | split | direct_accuracy | executor_accuracy | executor_target_mass | init_accuracy | init_pos_accuracy | op_accuracy | arg_accuracy | op_pos_accuracy | arg_pos_accuracy | program_exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main_qwen35_after_op_retention | compiler_trace_then_answer | standard_len4 | n/a | 60.2% | 15.2% | 100.0% | 99.2% | 95.2% | 90.8% | 36.1% | 47.6% | 60.2% |
| main_qwen35_after_op_retention | compiler_trace_then_answer | standard_len8 | n/a | 0.4% | 1.0% | 100.0% | 99.2% | 78.8% | 69.5% | 18.5% | 24.3% | 0.0% |
| main_qwen35_after_op_retention | compiler_trace_then_answer | standard_len12 | n/a | 1.2% | 1.0% | 100.0% | 99.2% | 68.1% | 55.0% | 12.1% | 16.4% | 0.0% |
| main_qwen35_after_op_retention | compiler_trace_then_answer | standard_len24 | n/a | 0.8% | 1.0% | 100.0% | 96.5% | 55.4% | 36.3% | 6.4% | 8.1% | 0.0% |
| main_qwen35_after_op_retention | compiler_trace_then_answer | paraphrase_len4 | n/a | 3.1% | 1.5% | 49.2% | 62.9% | 61.7% | 44.3% | 9.0% | 33.6% | 0.8% |
| main_qwen35_after_op_retention | compiler_trace_then_answer | paraphrase_len8 | n/a | 2.0% | 1.1% | 44.9% | 57.8% | 50.9% | 30.7% | 6.9% | 20.3% | 0.0% |
| main_qwen35_after_op_retention | compiler_trace_then_answer | paraphrase_len12 | n/a | 2.0% | 1.0% | 44.5% | 64.1% | 47.1% | 24.7% | 5.0% | 14.4% | 0.0% |
| main_qwen35_after_op_retention | compiler_trace_then_answer | paraphrase_len24 | n/a | 1.2% | 1.0% | 39.1% | 54.3% | 43.7% | 21.8% | 8.6% | 13.5% | 0.0% |
| main_qwen35_after_op_retention | compiler_trace_then_answer_low_lr | standard_len4 | n/a | 78.1% | 16.9% | 100.0% | 100.0% | 97.3% | 96.4% | 47.9% | 50.4% | 78.1% |
| main_qwen35_after_op_retention | compiler_trace_then_answer_low_lr | standard_len8 | n/a | 6.6% | 1.3% | 100.0% | 100.0% | 86.9% | 81.6% | 24.2% | 29.2% | 5.5% |
| main_qwen35_after_op_retention | compiler_trace_then_answer_low_lr | standard_len12 | n/a | 0.8% | 1.0% | 100.0% | 100.0% | 77.0% | 67.0% | 16.0% | 20.0% | 0.0% |
| main_qwen35_after_op_retention | compiler_trace_then_answer_low_lr | standard_len24 | n/a | 0.0% | 1.0% | 100.0% | 100.0% | 62.4% | 43.7% | 8.1% | 10.1% | 0.0% |
| main_qwen35_after_op_retention | compiler_trace_then_answer_low_lr | paraphrase_len4 | n/a | 2.0% | 1.5% | 59.8% | 82.0% | 62.1% | 53.4% | 18.1% | 40.4% | 0.8% |
| main_qwen35_after_op_retention | compiler_trace_then_answer_low_lr | paraphrase_len8 | n/a | 0.8% | 1.0% | 57.0% | 81.2% | 56.6% | 40.2% | 19.1% | 29.3% | 0.0% |
| main_qwen35_after_op_retention | compiler_trace_then_answer_low_lr | paraphrase_len12 | n/a | 0.4% | 1.0% | 58.6% | 79.7% | 54.5% | 36.0% | 19.5% | 27.6% | 0.0% |
| main_qwen35_after_op_retention | compiler_trace_then_answer_low_lr | paraphrase_len24 | n/a | 1.2% | 1.0% | 55.1% | 82.0% | 51.4% | 31.5% | 22.2% | 25.0% | 0.0% |
| main_qwen35_after_op_trace_controls | direct | standard_len4 | 2.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_after_op_trace_controls | direct | standard_len8 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_after_op_trace_controls | direct | standard_len12 | 1.2% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_after_op_trace_controls | direct | standard_len24 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_after_op_trace_controls | direct | paraphrase_len4 | 1.2% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_after_op_trace_controls | direct | paraphrase_len8 | 0.8% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_after_op_trace_controls | direct | paraphrase_len12 | 2.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_after_op_trace_controls | direct | paraphrase_len24 | 1.2% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_after_op_trace_controls | compiler_trace | standard_len4 | n/a | 78.5% | 8.4% | 100.0% | 100.0% | 99.9% | 94.0% | 100.0% | 95.1% | 78.1% |
| main_qwen35_after_op_trace_controls | compiler_trace | standard_len8 | n/a | 62.5% | 1.2% | 100.0% | 100.0% | 99.2% | 94.8% | 100.0% | 94.8% | 62.1% |
| main_qwen35_after_op_trace_controls | compiler_trace | standard_len12 | n/a | 39.1% | 1.0% | 100.0% | 100.0% | 99.0% | 93.2% | 100.0% | 93.7% | 38.7% |
| main_qwen35_after_op_trace_controls | compiler_trace | standard_len24 | n/a | 0.4% | 1.0% | 100.0% | 100.0% | 82.6% | 70.7% | 72.9% | 68.8% | 0.0% |
| main_qwen35_after_op_trace_controls | compiler_trace | paraphrase_len4 | n/a | 4.3% | 1.5% | 60.2% | 92.2% | 68.8% | 54.9% | 60.0% | 58.9% | 2.7% |
| main_qwen35_after_op_trace_controls | compiler_trace | paraphrase_len8 | n/a | 0.8% | 1.0% | 50.4% | 94.1% | 59.5% | 40.4% | 41.9% | 40.8% | 0.4% |
| main_qwen35_after_op_trace_controls | compiler_trace | paraphrase_len12 | n/a | 2.3% | 1.0% | 49.6% | 95.3% | 53.6% | 31.4% | 30.0% | 30.2% | 0.0% |
| main_qwen35_after_op_trace_controls | compiler_trace | paraphrase_len24 | n/a | 0.4% | 1.0% | 42.6% | 94.5% | 45.3% | 18.5% | 15.9% | 16.0% | 0.0% |
| main_qwen35_after_op_trace_controls | compiler_answer_only | standard_len4 | n/a | 2.3% | 1.0% | 1.6% | 0.0% | 33.4% | 3.8% | 0.0% | 0.0% | 0.0% |
| main_qwen35_after_op_trace_controls | compiler_answer_only | standard_len8 | n/a | 2.3% | 1.0% | 0.4% | 0.0% | 33.3% | 4.4% | 0.0% | 0.0% | 0.0% |
| main_qwen35_after_op_trace_controls | compiler_answer_only | standard_len12 | n/a | 2.0% | 1.0% | 2.0% | 0.0% | 33.1% | 4.3% | 0.0% | 0.0% | 0.0% |
| main_qwen35_after_op_trace_controls | compiler_answer_only | standard_len24 | n/a | 3.1% | 1.0% | 0.0% | 0.0% | 33.7% | 4.7% | 0.0% | 0.0% | 0.0% |
| main_qwen35_after_op_trace_controls | compiler_answer_only | paraphrase_len4 | n/a | 1.2% | 1.0% | 0.8% | 0.0% | 36.4% | 4.3% | 0.0% | 0.0% | 0.0% |
| main_qwen35_after_op_trace_controls | compiler_answer_only | paraphrase_len8 | n/a | 1.6% | 1.0% | 0.8% | 0.0% | 32.5% | 4.6% | 0.0% | 0.0% | 0.0% |
| main_qwen35_after_op_trace_controls | compiler_answer_only | paraphrase_len12 | n/a | 1.2% | 1.0% | 0.8% | 0.0% | 32.2% | 4.0% | 0.0% | 0.0% | 0.0% |
| main_qwen35_after_op_trace_controls | compiler_answer_only | paraphrase_len24 | n/a | 1.6% | 1.0% | 1.6% | 0.0% | 32.8% | 3.5% | 0.0% | 0.0% | 0.0% |
| pilot_qwen35_shared_parser_l12_after_op_arg4_trace | compiler_trace | standard_len4 | n/a | 75.0% | 8.1% | 100.0% | 100.0% | 99.4% | 94.1% | 100.0% | 95.9% | 75.0% |
| pilot_qwen35_shared_parser_l12_after_op_arg4_trace | compiler_trace | standard_len8 | n/a | 43.0% | 1.2% | 100.0% | 100.0% | 96.6% | 92.8% | 100.0% | 94.3% | 41.4% |
| pilot_qwen35_shared_parser_l12_after_op_arg4_trace | compiler_trace | standard_len12 | n/a | 18.0% | 1.0% | 100.0% | 100.0% | 95.0% | 90.3% | 100.0% | 92.5% | 17.2% |
| pilot_qwen35_shared_parser_l12_after_op_arg4_trace | compiler_trace | standard_len24 | n/a | 3.9% | 1.0% | 100.0% | 100.0% | 91.6% | 88.6% | 100.0% | 92.5% | 2.3% |
| pilot_qwen35_shared_parser_l12_after_op_arg4_trace | compiler_trace | paraphrase_len4 | n/a | 1.6% | 1.5% | 66.4% | 93.8% | 63.9% | 47.9% | 51.2% | 50.6% | 0.0% |
| pilot_qwen35_shared_parser_l12_after_op_arg4_trace | compiler_trace | paraphrase_len8 | n/a | 0.8% | 1.0% | 68.0% | 92.2% | 57.8% | 37.2% | 36.5% | 35.4% | 0.0% |
| pilot_qwen35_shared_parser_l12_after_op_arg4_trace | compiler_trace | paraphrase_len12 | n/a | 1.6% | 1.0% | 60.9% | 95.3% | 53.6% | 32.1% | 30.1% | 30.3% | 0.0% |
| pilot_qwen35_shared_parser_l12_after_op_arg4_trace | compiler_trace | paraphrase_len24 | n/a | 1.6% | 1.0% | 59.4% | 91.4% | 46.5% | 20.4% | 16.3% | 16.1% | 0.0% |
| pilot_qwen35_shared_parser_l12_after_op_strong_trace | compiler_trace | standard_len4 | n/a | 80.1% | 9.1% | 100.0% | 100.0% | 99.0% | 95.2% | 100.0% | 94.5% | 80.1% |
| pilot_qwen35_shared_parser_l12_after_op_strong_trace | compiler_trace | standard_len8 | n/a | 50.4% | 1.2% | 100.0% | 100.0% | 96.7% | 94.9% | 100.0% | 94.1% | 50.0% |
| pilot_qwen35_shared_parser_l12_after_op_strong_trace | compiler_trace | standard_len12 | n/a | 28.5% | 1.0% | 100.0% | 100.0% | 96.4% | 93.6% | 100.0% | 92.4% | 27.7% |
| pilot_qwen35_shared_parser_l12_after_op_strong_trace | compiler_trace | standard_len24 | n/a | 1.2% | 1.0% | 100.0% | 100.0% | 76.0% | 63.0% | 64.5% | 60.2% | 0.0% |
| pilot_qwen35_shared_parser_l12_arg4_trace | compiler_trace | standard_len4 | n/a | 75.8% | 7.4% | 100.0% | 100.0% | 98.6% | 94.1% | 100.0% | 94.1% | 75.8% |
| pilot_qwen35_shared_parser_l12_arg4_trace | compiler_trace | standard_len8 | n/a | 39.1% | 1.2% | 100.0% | 100.0% | 97.5% | 86.9% | 100.0% | 88.4% | 38.3% |
| pilot_qwen35_shared_parser_l12_arg4_trace | compiler_trace | standard_len12 | n/a | 9.4% | 1.0% | 100.0% | 100.0% | 97.1% | 79.1% | 100.0% | 80.9% | 8.6% |
| pilot_qwen35_shared_parser_l12_arg4_trace | compiler_trace | standard_len24 | n/a | 0.8% | 1.0% | 100.0% | 100.0% | 95.6% | 68.5% | 100.0% | 71.5% | 0.0% |
| pilot_qwen35_shared_parser_l12_arg4_trace | compiler_trace | paraphrase_len4 | n/a | 2.3% | 1.3% | 32.0% | 93.8% | 60.4% | 33.8% | 50.4% | 24.8% | 0.0% |
| pilot_qwen35_shared_parser_l12_arg4_trace | compiler_trace | paraphrase_len8 | n/a | 0.8% | 1.0% | 35.9% | 94.5% | 56.4% | 19.6% | 34.8% | 13.5% | 0.0% |
| pilot_qwen35_shared_parser_l12_arg4_trace | compiler_trace | paraphrase_len12 | n/a | 0.0% | 1.0% | 28.1% | 95.3% | 48.7% | 12.9% | 27.1% | 7.2% | 0.0% |
| pilot_qwen35_shared_parser_l12_arg4_trace | compiler_trace | paraphrase_len24 | n/a | 0.0% | 1.0% | 28.1% | 88.3% | 41.7% | 9.4% | 11.5% | 4.3% | 0.0% |
| pilot_qwen35_shared_parser_l12_trace | compiler_trace | standard_len4 | n/a | 50.0% | 6.8% | 100.0% | 100.0% | 98.2% | 85.9% | 100.0% | 93.6% | 50.0% |
| pilot_qwen35_shared_parser_l12_trace | compiler_trace | standard_len8 | n/a | 15.6% | 1.2% | 100.0% | 100.0% | 96.8% | 78.9% | 100.0% | 87.5% | 15.6% |
| pilot_qwen35_shared_parser_l12_trace | compiler_trace | standard_len12 | n/a | 3.1% | 1.0% | 100.0% | 100.0% | 96.0% | 70.8% | 100.0% | 78.4% | 2.3% |
| pilot_qwen35_shared_parser_l12_trace | compiler_trace | standard_len24 | n/a | 1.6% | 1.0% | 100.0% | 100.0% | 95.2% | 62.3% | 100.0% | 70.1% | 0.0% |
| pilot_qwen35_shared_parser_l12_trace | compiler_trace | paraphrase_len4 | n/a | 3.1% | 1.3% | 44.5% | 93.0% | 60.9% | 34.8% | 50.0% | 28.9% | 0.0% |
| pilot_qwen35_shared_parser_l12_trace | compiler_trace | paraphrase_len8 | n/a | 0.8% | 1.0% | 38.3% | 87.5% | 56.8% | 23.1% | 34.6% | 16.6% | 0.0% |
| pilot_qwen35_shared_parser_l12_trace | compiler_trace | paraphrase_len12 | n/a | 1.6% | 1.0% | 35.9% | 93.8% | 48.0% | 15.2% | 26.8% | 9.8% | 0.0% |
| pilot_qwen35_shared_parser_l12_trace | compiler_trace | paraphrase_len24 | n/a | 0.8% | 1.0% | 27.3% | 89.1% | 41.2% | 9.8% | 11.5% | 4.9% | 0.0% |
| pilot_qwen35_shared_parser_l4_count_trace | compiler_trace | standard_len4 | n/a | 55.5% | 6.8% | 99.2% | 100.0% | 98.4% | 85.2% | 100.0% | 88.1% | 53.9% |
| pilot_qwen35_shared_parser_l4_count_trace | compiler_trace | standard_len8 | n/a | 0.8% | 1.1% | 99.2% | 100.0% | 89.3% | 64.8% | 100.0% | 60.7% | 0.0% |
| pilot_qwen35_shared_parser_l4_count_trace | compiler_trace | standard_len12 | n/a | 0.8% | 1.0% | 100.0% | 100.0% | 89.5% | 48.6% | 100.0% | 46.7% | 0.0% |
| pilot_qwen35_shared_parser_l4_count_trace | compiler_trace | standard_len24 | n/a | 1.6% | 1.0% | 100.0% | 100.0% | 85.5% | 24.3% | 99.6% | 21.5% | 0.0% |
| pilot_qwen35_shared_parser_l4_count_trace | compiler_trace | paraphrase_len4 | n/a | 0.0% | 1.0% | 54.7% | 96.1% | 63.3% | 32.2% | 53.7% | 29.5% | 0.0% |
| pilot_qwen35_shared_parser_l4_count_trace | compiler_trace | paraphrase_len8 | n/a | 0.8% | 1.0% | 50.8% | 94.5% | 59.3% | 22.7% | 38.6% | 18.9% | 0.0% |
| pilot_qwen35_shared_parser_l4_count_trace | compiler_trace | paraphrase_len12 | n/a | 0.8% | 1.0% | 46.1% | 95.3% | 55.5% | 17.8% | 33.8% | 12.4% | 0.0% |
| pilot_qwen35_shared_parser_l4_count_trace | compiler_trace | paraphrase_len24 | n/a | 0.8% | 1.0% | 38.3% | 89.1% | 47.4% | 10.6% | 17.7% | 5.9% | 0.0% |
| pilot_qwen35_shared_parser_l4_trace | compiler_trace | standard_len4 | n/a | 50.8% | 7.8% | 100.0% |  | 93.8% | 89.8% |  |  | 50.8% |
| pilot_qwen35_shared_parser_l4_trace | compiler_trace | standard_len8 | n/a | 3.1% | 1.0% | 100.0% |  | 77.1% | 58.1% |  |  | 0.0% |
| pilot_qwen35_shared_parser_l4_trace | compiler_trace | standard_len12 | n/a | 0.0% | 1.0% | 100.0% |  | 74.1% | 42.4% |  |  | 0.0% |
| pilot_qwen35_shared_parser_l4_trace | compiler_trace | standard_len24 | n/a | 0.0% | 1.0% | 100.0% |  | 69.8% | 21.5% |  |  | 0.0% |
| pilot_qwen35_shared_parser_l4_trace | compiler_trace | paraphrase_len4 | n/a | 0.8% | 1.0% | 68.8% |  | 60.5% | 37.5% |  |  | 0.0% |
| pilot_qwen35_shared_parser_l4_trace | compiler_trace | paraphrase_len8 | n/a | 1.6% | 1.0% | 63.3% |  | 60.7% | 25.4% |  |  | 0.0% |
| pilot_qwen35_shared_parser_l4_trace | compiler_trace | paraphrase_len12 | n/a | 1.6% | 1.0% | 57.8% |  | 57.1% | 20.1% |  |  | 0.0% |
| pilot_qwen35_shared_parser_l4_trace | compiler_trace | paraphrase_len24 | n/a | 1.6% | 1.0% | 47.7% |  | 50.0% | 12.0% |  |  | 0.0% |
| pilot_qwen35_shared_parser_l4_trace_diag | compiler_trace | standard_len4 | n/a | 60.9% | 7.4% | 100.0% | 100.0% | 97.9% | 89.3% | 100.0% | 90.8% | 60.2% |
| pilot_qwen35_shared_parser_l4_trace_diag | compiler_trace | standard_len8 | n/a | 3.1% | 1.1% | 100.0% | 100.0% | 87.7% | 65.7% | 100.0% | 62.5% | 2.3% |
| pilot_qwen35_shared_parser_l4_trace_diag | compiler_trace | standard_len12 | n/a | 0.8% | 1.0% | 100.0% | 100.0% | 88.2% | 48.6% | 100.0% | 45.2% | 0.0% |
| pilot_qwen35_shared_parser_l4_trace_diag | compiler_trace | standard_len24 | n/a | 1.6% | 1.0% | 100.0% | 100.0% | 83.1% | 25.6% | 99.4% | 21.2% | 0.0% |
| pilot_qwen35_shared_parser_l4_trace_diag | compiler_trace | paraphrase_len4 | n/a | 1.6% | 1.0% | 66.4% | 93.0% | 63.1% | 35.9% | 52.5% | 35.9% | 0.0% |
| pilot_qwen35_shared_parser_l4_trace_diag | compiler_trace | paraphrase_len8 | n/a | 0.8% | 1.0% | 61.7% | 87.5% | 59.7% | 25.9% | 37.1% | 24.0% | 0.0% |
| pilot_qwen35_shared_parser_l4_trace_diag | compiler_trace | paraphrase_len12 | n/a | 2.3% | 1.0% | 56.2% | 92.2% | 54.1% | 20.1% | 31.1% | 16.4% | 0.0% |
| pilot_qwen35_shared_parser_l4_trace_diag | compiler_trace | paraphrase_len24 | n/a | 0.8% | 1.0% | 47.7% | 87.5% | 45.2% | 11.3% | 15.8% | 7.5% | 0.0% |
| smoke_tiny | direct | standard_len2 | 16.7% | n/a | n/a | n/a |  | n/a | n/a |  |  | n/a |
| smoke_tiny | direct | standard_len4 | 0.0% | n/a | n/a | n/a |  | n/a | n/a |  |  | n/a |
| smoke_tiny | direct | paraphrase_len2 | 0.0% | n/a | n/a | n/a |  | n/a | n/a |  |  | n/a |
| smoke_tiny | direct | paraphrase_len4 | 0.0% | n/a | n/a | n/a |  | n/a | n/a |  |  | n/a |
| smoke_tiny | compiler_trace | standard_len2 | n/a | 0.0% | 6.2% | 0.0% |  | 25.0% | 25.0% |  |  | 0.0% |
| smoke_tiny | compiler_trace | standard_len4 | n/a | 0.0% | 5.7% | 0.0% |  | 31.2% | 4.2% |  |  | 0.0% |
| smoke_tiny | compiler_trace | paraphrase_len2 | n/a | 0.0% | 5.9% | 0.0% |  | 25.0% | 16.7% |  |  | 0.0% |
| smoke_tiny | compiler_trace | paraphrase_len4 | n/a | 8.3% | 5.9% | 8.3% |  | 27.1% | 8.3% |  |  | 0.0% |
| smoke_tiny | compiler_answer_only | standard_len2 | n/a | 16.7% | 6.1% | 16.7% |  | 16.7% | 8.3% |  |  | 0.0% |
| smoke_tiny | compiler_answer_only | standard_len4 | n/a | 8.3% | 5.7% | 0.0% |  | 43.8% | 2.1% |  |  | 0.0% |
| smoke_tiny | compiler_answer_only | paraphrase_len2 | n/a | 8.3% | 6.0% | 8.3% |  | 33.3% | 0.0% |  |  | 0.0% |
| smoke_tiny | compiler_answer_only | paraphrase_len4 | n/a | 8.3% | 6.0% | 8.3% |  | 31.2% | 8.3% |  |  | 0.0% |
| smoke_tiny | compiler_trace_then_answer | standard_len2 | n/a | 0.0% | 6.1% | 8.3% |  | 29.2% | 12.5% |  |  | 0.0% |
| smoke_tiny | compiler_trace_then_answer | standard_len4 | n/a | 0.0% | 5.7% | 0.0% |  | 43.8% | 2.1% |  |  | 0.0% |
| smoke_tiny | compiler_trace_then_answer | paraphrase_len2 | n/a | 25.0% | 5.9% | 0.0% |  | 41.7% | 16.7% |  |  | 0.0% |
| smoke_tiny | compiler_trace_then_answer | paraphrase_len4 | n/a | 0.0% | 5.9% | 0.0% |  | 33.3% | 6.2% |  |  | 0.0% |
| smoke_tiny_after_op | compiler_trace | standard_len2 | n/a | 0.0% | 6.1% | 8.3% | 0.0% | 25.0% | 4.2% | 0.0% | 0.0% | 0.0% |
| smoke_tiny_after_op | compiler_trace | standard_len4 | n/a | 8.3% | 5.7% | 8.3% | 0.0% | 29.2% | 2.1% | 4.2% | 6.2% | 0.0% |
| smoke_tiny_sparse_init | compiler_trace | standard_len2 | n/a | 0.0% | 6.1% | 8.3% |  | 25.0% | 4.2% |  |  | 0.0% |
| smoke_tiny_sparse_init | compiler_trace | standard_len4 | n/a | 8.3% | 5.7% | 8.3% |  | 29.2% | 2.1% |  |  | 0.0% |

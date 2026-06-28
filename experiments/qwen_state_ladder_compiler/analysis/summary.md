# Qwen State-Ladder Compiler Analysis Summary

## Final Metrics

| run | variant | split | direct_accuracy | executor_accuracy | executor_target_mass | init_accuracy | init_pos_accuracy | op_accuracy | arg_accuracy | op_pos_accuracy | arg_pos_accuracy | program_exact | state_accuracy | state_all_exact | state_prefix_fraction | executor_pair_answer_consistency | executor_pair_both_correct | compiler_pair_program_consistency | compiler_pair_state_consistency | direct_pair_answer_consistency | direct_pair_both_correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control_qwen3_4b_qlora_answer_only_curriculum_s900 | direct | standard_len4 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_curriculum_s900 | direct | standard_len8 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_curriculum_s900 | direct | standard_len12 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_curriculum_s900 | direct | standard_len24 | 3.1% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_curriculum_s900 | direct | paraphrase_len4 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_curriculum_s900 | direct | paraphrase_len8 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_curriculum_s900 | direct | paraphrase_len12 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_curriculum_s900 | direct | paraphrase_len24 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_curriculum_s900 | direct | paired_len4 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  | 100.0% | 1.6% |
| control_qwen3_4b_qlora_answer_only_curriculum_s900 | direct | paired_len8 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  | 100.0% | 1.6% |
| control_qwen3_4b_qlora_answer_only_curriculum_s900 | direct | paired_len12 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  | 100.0% | 0.0% |
| control_qwen3_4b_qlora_answer_only_curriculum_s900 | direct | paired_len24 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  | 100.0% | 1.6% |
| control_qwen3_4b_qlora_curriculum_no_state_ladder_s900 | copy_trace | standard_len4 | n/a | 92.2% | 8.3% | 100.0% | 100.0% | 100.0% | 97.7% | 100.0% | 97.7% | 92.2% | 94.9% | 92.2% | 94.9% |  |  |  |  |  |  |
| control_qwen3_4b_qlora_curriculum_no_state_ladder_s900 | copy_trace | standard_len8 | n/a | 65.6% | 1.3% | 100.0% | 100.0% | 100.0% | 95.3% | 100.0% | 94.7% | 65.6% | 79.7% | 65.6% | 79.7% |  |  |  |  |  |  |
| control_qwen3_4b_qlora_curriculum_no_state_ladder_s900 | copy_trace | standard_len12 | n/a | 43.8% | 1.0% | 100.0% | 100.0% | 100.0% | 94.1% | 100.0% | 94.3% | 43.8% | 70.6% | 43.8% | 70.6% |  |  |  |  |  |  |
| control_qwen3_4b_qlora_curriculum_no_state_ladder_s900 | copy_trace | standard_len24 | n/a | 39.1% | 1.0% | 100.0% | 100.0% | 100.0% | 95.3% | 100.0% | 95.1% | 39.1% | 63.7% | 39.1% | 63.7% |  |  |  |  |  |  |
| control_qwen3_4b_qlora_curriculum_no_state_ladder_s900 | copy_trace | paraphrase_len4 | n/a | 78.1% | 7.8% | 100.0% | 100.0% | 100.0% | 94.5% | 100.0% | 95.3% | 78.1% | 85.2% | 78.1% | 85.2% |  |  |  |  |  |  |
| control_qwen3_4b_qlora_curriculum_no_state_ladder_s900 | copy_trace | paraphrase_len8 | n/a | 64.1% | 1.2% | 100.0% | 100.0% | 100.0% | 94.9% | 100.0% | 94.5% | 64.1% | 84.0% | 64.1% | 84.0% |  |  |  |  |  |  |
| control_qwen3_4b_qlora_curriculum_no_state_ladder_s900 | copy_trace | paraphrase_len12 | n/a | 59.4% | 1.0% | 100.0% | 100.0% | 100.0% | 95.7% | 100.0% | 95.8% | 59.4% | 74.2% | 59.4% | 74.2% |  |  |  |  |  |  |
| control_qwen3_4b_qlora_curriculum_no_state_ladder_s900 | copy_trace | paraphrase_len24 | n/a | 15.6% | 1.0% | 100.0% | 100.0% | 99.9% | 92.4% | 100.0% | 92.1% | 15.6% | 45.2% | 15.6% | 45.2% |  |  |  |  |  |  |
| control_qwen3_4b_qlora_curriculum_no_state_ladder_s900 | copy_trace | paired_len4 | n/a | 86.7% | 7.5% | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 96.5% | 86.7% | 93.2% | 86.7% | 93.2% | 98.4% | 85.9% | 98.4% | 98.4% |  |  |
| control_qwen3_4b_qlora_curriculum_no_state_ladder_s900 | copy_trace | paired_len8 | n/a | 58.6% | 1.3% | 100.0% | 100.0% | 100.0% | 93.7% | 100.0% | 94.2% | 58.6% | 75.6% | 58.6% | 75.6% | 84.4% | 56.2% | 84.4% | 84.4% |  |  |
| control_qwen3_4b_qlora_curriculum_no_state_ladder_s900 | copy_trace | paired_len12 | n/a | 59.4% | 1.0% | 100.0% | 100.0% | 100.0% | 95.1% | 100.0% | 95.4% | 57.8% | 79.2% | 57.8% | 78.0% | 73.4% | 54.7% | 73.4% | 73.4% |  |  |
| control_qwen3_4b_qlora_curriculum_no_state_ladder_s900 | copy_trace | paired_len24 | n/a | 21.1% | 1.0% | 100.0% | 100.0% | 99.9% | 93.5% | 99.9% | 93.7% | 20.3% | 48.6% | 20.3% | 48.6% | 42.2% | 12.5% | 42.2% | 42.2% |  |  |
| main_qwen3_4b_qlora_state_ladder_curriculum_s900 | copy_trace_state_ladder | standard_len4 | n/a | 93.8% | 8.4% | 100.0% | 100.0% | 100.0% | 98.0% | 100.0% | 98.0% | 93.8% | 96.1% | 93.8% | 96.1% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_curriculum_s900 | copy_trace_state_ladder | standard_len8 | n/a | 67.2% | 1.3% | 100.0% | 100.0% | 100.0% | 95.5% | 100.0% | 96.1% | 67.2% | 81.2% | 67.2% | 81.2% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_curriculum_s900 | copy_trace_state_ladder | standard_len12 | n/a | 51.6% | 1.0% | 100.0% | 100.0% | 100.0% | 94.8% | 100.0% | 94.7% | 51.6% | 76.2% | 51.6% | 76.2% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_curriculum_s900 | copy_trace_state_ladder | standard_len24 | n/a | 29.7% | 1.0% | 100.0% | 100.0% | 99.9% | 94.9% | 100.0% | 95.1% | 28.1% | 62.9% | 28.1% | 62.8% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_curriculum_s900 | copy_trace_state_ladder | paraphrase_len4 | n/a | 84.4% | 8.0% | 100.0% | 100.0% | 100.0% | 96.1% | 100.0% | 97.3% | 84.4% | 91.0% | 84.4% | 91.0% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_curriculum_s900 | copy_trace_state_ladder | paraphrase_len8 | n/a | 64.1% | 1.2% | 100.0% | 100.0% | 100.0% | 94.7% | 100.0% | 94.5% | 64.1% | 84.6% | 64.1% | 84.6% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_curriculum_s900 | copy_trace_state_ladder | paraphrase_len12 | n/a | 56.2% | 1.0% | 100.0% | 100.0% | 100.0% | 95.2% | 100.0% | 95.6% | 56.2% | 74.7% | 56.2% | 74.7% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_curriculum_s900 | copy_trace_state_ladder | paraphrase_len24 | n/a | 0.0% | 1.0% | 100.0% | 100.0% | 92.6% | 81.1% | 90.2% | 82.2% | 0.0% | 40.0% | 0.0% | 40.0% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_curriculum_s900 | copy_trace_state_ladder | paired_len4 | n/a | 86.7% | 7.6% | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 97.5% | 86.7% | 93.2% | 86.7% | 93.2% | 98.4% | 85.9% | 98.4% | 98.4% |  |  |
| main_qwen3_4b_qlora_state_ladder_curriculum_s900 | copy_trace_state_ladder | paired_len8 | n/a | 62.5% | 1.4% | 100.0% | 100.0% | 100.0% | 94.5% | 100.0% | 95.0% | 62.5% | 79.5% | 62.5% | 79.5% | 84.4% | 59.4% | 84.4% | 84.4% |  |  |
| main_qwen3_4b_qlora_state_ladder_curriculum_s900 | copy_trace_state_ladder | paired_len12 | n/a | 53.1% | 1.0% | 100.0% | 100.0% | 100.0% | 95.2% | 100.0% | 95.8% | 53.1% | 78.1% | 53.1% | 78.1% | 71.9% | 45.3% | 71.9% | 71.9% |  |  |
| main_qwen3_4b_qlora_state_ladder_curriculum_s900 | copy_trace_state_ladder | paired_len24 | n/a | 14.8% | 1.0% | 100.0% | 100.0% | 95.5% | 87.3% | 94.7% | 88.0% | 14.8% | 49.9% | 14.8% | 49.8% | 3.1% | 1.6% | 1.6% | 1.6% |  |  |
| main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900 | copy_trace_state_ladder | standard_len4 | n/a | 90.6% | 8.3% | 100.0% | 100.0% | 100.0% | 97.7% | 100.0% | 98.0% | 90.6% | 93.4% | 90.6% | 93.4% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900 | copy_trace_state_ladder | standard_len8 | n/a | 64.1% | 1.3% | 100.0% | 100.0% | 100.0% | 95.1% | 100.0% | 94.5% | 64.1% | 78.1% | 64.1% | 78.1% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900 | copy_trace_state_ladder | standard_len12 | n/a | 45.3% | 1.0% | 100.0% | 100.0% | 100.0% | 94.3% | 100.0% | 94.3% | 45.3% | 71.0% | 45.3% | 71.0% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900 | copy_trace_state_ladder | standard_len24 | n/a | 37.5% | 1.0% | 100.0% | 100.0% | 100.0% | 95.4% | 100.0% | 95.2% | 37.5% | 63.7% | 37.5% | 63.7% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900 | copy_trace_state_ladder | paraphrase_len4 | n/a | 75.0% | 7.5% | 100.0% | 100.0% | 100.0% | 93.8% | 100.0% | 94.5% | 75.0% | 83.6% | 75.0% | 83.6% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900 | copy_trace_state_ladder | paraphrase_len8 | n/a | 59.4% | 1.2% | 100.0% | 100.0% | 100.0% | 93.8% | 100.0% | 94.1% | 59.4% | 79.1% | 59.4% | 79.1% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900 | copy_trace_state_ladder | paraphrase_len12 | n/a | 57.8% | 1.0% | 100.0% | 100.0% | 100.0% | 95.3% | 100.0% | 95.7% | 57.8% | 72.4% | 57.8% | 72.4% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900 | copy_trace_state_ladder | paraphrase_len24 | n/a | 14.1% | 1.0% | 100.0% | 100.0% | 99.6% | 91.1% | 99.8% | 91.0% | 14.1% | 46.1% | 14.1% | 46.1% |  |  |  |  |  |  |
| main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900 | copy_trace_state_ladder | paired_len4 | n/a | 86.7% | 7.4% | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 96.7% | 86.7% | 93.2% | 86.7% | 93.2% | 95.3% | 85.9% | 95.3% | 95.3% |  |  |
| main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900 | copy_trace_state_ladder | paired_len8 | n/a | 60.9% | 1.3% | 100.0% | 100.0% | 100.0% | 94.3% | 100.0% | 94.5% | 60.9% | 78.4% | 60.9% | 78.4% | 87.5% | 60.9% | 87.5% | 87.5% |  |  |
| main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900 | copy_trace_state_ladder | paired_len12 | n/a | 59.4% | 1.0% | 100.0% | 100.0% | 100.0% | 95.2% | 100.0% | 95.4% | 57.8% | 79.6% | 57.8% | 78.5% | 76.6% | 54.7% | 76.6% | 76.6% |  |  |
| main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900 | copy_trace_state_ladder | paired_len24 | n/a | 19.5% | 1.0% | 100.0% | 100.0% | 99.8% | 93.1% | 100.0% | 93.3% | 18.8% | 49.0% | 18.8% | 49.0% | 28.1% | 12.5% | 26.6% | 26.6% |  |  |

## Best Logged Paired L24 Checkpoints

| run | variant | step | stage | paired_len24_executor_accuracy | paired_len24_state_prefix_fraction | paired_len24_compiler_pair_state_consistency | standard_len24_executor_accuracy | paraphrase_len24_executor_accuracy |
|---|---|---|---|---|---|---|---|---|
| control_qwen3_4b_qlora_curriculum_no_state_ladder_s900 | copy_trace | 800 | trace_long | 25.0% | 51.1% | 53.1% | 32.8% | 23.4% |
| main_qwen3_4b_qlora_state_ladder_curriculum_s900 | copy_trace_state_ladder | 800 | trace_state_ladder_long | 23.4% | 52.3% | 43.8% | 31.2% | 10.9% |
| main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900 | copy_trace_state_ladder | 800 | trace_state_ladder_long | 28.1% | 52.7% | 56.2% | 35.9% | 21.9% |

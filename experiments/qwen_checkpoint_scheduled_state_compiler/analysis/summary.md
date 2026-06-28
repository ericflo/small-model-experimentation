# Checkpoint-Selected Scheduled-State Compiler Analysis Summary

## Final Metrics

| run | variant | split | direct_accuracy | executor_accuracy | executor_target_mass | init_accuracy | init_pos_accuracy | op_accuracy | arg_accuracy | op_pos_accuracy | arg_pos_accuracy | program_exact | state_accuracy | state_all_exact | state_prefix_fraction | executor_pair_answer_consistency | executor_pair_both_correct | compiler_pair_program_consistency | compiler_pair_state_consistency | direct_pair_answer_consistency | direct_pair_both_correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main_no_state_selected_s900 | copy_trace | standard_len4 | n/a | 92.2% | 8.3% | 100.0% | 100.0% | 100.0% | 98.0% | 100.0% | 98.4% | 92.2% | 94.9% | 92.2% | 94.9% |  |  |  |  |  |  |
| main_no_state_selected_s900 | copy_trace | standard_len8 | n/a | 65.6% | 1.3% | 100.0% | 100.0% | 100.0% | 95.3% | 100.0% | 94.9% | 65.6% | 79.7% | 65.6% | 79.7% |  |  |  |  |  |  |
| main_no_state_selected_s900 | copy_trace | standard_len12 | n/a | 43.8% | 1.0% | 100.0% | 100.0% | 100.0% | 94.0% | 100.0% | 94.3% | 42.2% | 70.6% | 43.8% | 70.6% |  |  |  |  |  |  |
| main_no_state_selected_s900 | copy_trace | standard_len24 | n/a | 32.8% | 1.0% | 100.0% | 100.0% | 100.0% | 95.0% | 100.0% | 95.3% | 32.8% | 61.4% | 32.8% | 61.4% |  |  |  |  |  |  |
| main_no_state_selected_s900 | copy_trace | paraphrase_len4 | n/a | 76.6% | 7.8% | 100.0% | 100.0% | 100.0% | 94.1% | 100.0% | 94.5% | 76.6% | 84.0% | 76.6% | 84.0% |  |  |  |  |  |  |
| main_no_state_selected_s900 | copy_trace | paraphrase_len8 | n/a | 62.5% | 1.2% | 100.0% | 100.0% | 100.0% | 94.5% | 100.0% | 94.1% | 62.5% | 81.6% | 62.5% | 81.6% |  |  |  |  |  |  |
| main_no_state_selected_s900 | copy_trace | paraphrase_len12 | n/a | 57.8% | 1.0% | 100.0% | 100.0% | 100.0% | 95.4% | 100.0% | 95.6% | 57.8% | 73.8% | 57.8% | 73.8% |  |  |  |  |  |  |
| main_no_state_selected_s900 | copy_trace | paraphrase_len24 | n/a | 6.2% | 1.0% | 100.0% | 100.0% | 98.9% | 89.2% | 99.7% | 90.4% | 6.2% | 45.5% | 6.2% | 45.5% |  |  |  |  |  |  |
| main_no_state_selected_s900 | copy_trace | paired_len4 | n/a | 86.7% | 7.5% | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 96.9% | 86.7% | 93.2% | 86.7% | 93.2% | 96.9% | 85.9% | 96.9% | 96.9% |  |  |
| main_no_state_selected_s900 | copy_trace | paired_len8 | n/a | 60.9% | 1.4% | 100.0% | 100.0% | 100.0% | 94.4% | 100.0% | 94.6% | 60.9% | 78.4% | 60.9% | 78.4% | 85.9% | 60.9% | 85.9% | 85.9% |  |  |
| main_no_state_selected_s900 | copy_trace | paired_len12 | n/a | 60.2% | 1.0% | 100.0% | 100.0% | 100.0% | 95.6% | 100.0% | 95.7% | 58.6% | 80.9% | 58.6% | 79.7% | 84.4% | 56.2% | 84.4% | 84.4% |  |  |
| main_no_state_selected_s900 | copy_trace | paired_len24 | n/a | 15.6% | 1.0% | 100.0% | 100.0% | 99.2% | 91.4% | 99.5% | 92.0% | 14.8% | 48.2% | 14.8% | 48.2% | 18.8% | 3.1% | 17.2% | 17.2% |  |  |
| main_state_l12_off_long_selected_s900 | copy_trace_state_scheduled | standard_len4 | n/a | 92.2% | 8.2% | 100.0% | 100.0% | 100.0% | 98.0% | 100.0% | 98.4% | 92.2% | 95.7% | 92.2% | 95.7% |  |  |  |  |  |  |
| main_state_l12_off_long_selected_s900 | copy_trace_state_scheduled | standard_len8 | n/a | 67.2% | 1.3% | 100.0% | 100.0% | 100.0% | 95.5% | 100.0% | 94.9% | 67.2% | 81.2% | 67.2% | 81.2% |  |  |  |  |  |  |
| main_state_l12_off_long_selected_s900 | copy_trace_state_scheduled | standard_len12 | n/a | 43.8% | 1.0% | 100.0% | 100.0% | 100.0% | 94.1% | 100.0% | 94.4% | 43.8% | 70.6% | 43.8% | 70.6% |  |  |  |  |  |  |
| main_state_l12_off_long_selected_s900 | copy_trace_state_scheduled | standard_len24 | n/a | 37.5% | 1.0% | 100.0% | 100.0% | 100.0% | 95.3% | 100.0% | 95.3% | 37.5% | 63.9% | 37.5% | 63.9% |  |  |  |  |  |  |
| main_state_l12_off_long_selected_s900 | copy_trace_state_scheduled | paraphrase_len4 | n/a | 82.8% | 7.9% | 100.0% | 100.0% | 100.0% | 95.7% | 100.0% | 95.7% | 82.8% | 89.8% | 82.8% | 89.8% |  |  |  |  |  |  |
| main_state_l12_off_long_selected_s900 | copy_trace_state_scheduled | paraphrase_len8 | n/a | 64.1% | 1.2% | 100.0% | 100.0% | 100.0% | 94.7% | 100.0% | 94.5% | 64.1% | 83.6% | 64.1% | 83.6% |  |  |  |  |  |  |
| main_state_l12_off_long_selected_s900 | copy_trace_state_scheduled | paraphrase_len12 | n/a | 57.8% | 1.0% | 100.0% | 100.0% | 100.0% | 95.4% | 100.0% | 95.7% | 57.8% | 74.9% | 57.8% | 74.9% |  |  |  |  |  |  |
| main_state_l12_off_long_selected_s900 | copy_trace_state_scheduled | paraphrase_len24 | n/a | 6.2% | 1.0% | 100.0% | 100.0% | 97.3% | 87.6% | 96.9% | 87.8% | 1.6% | 45.4% | 1.6% | 45.1% |  |  |  |  |  |  |
| main_state_l12_off_long_selected_s900 | copy_trace_state_scheduled | paired_len4 | n/a | 86.7% | 7.4% | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 96.5% | 86.7% | 93.2% | 86.7% | 93.2% | 98.4% | 85.9% | 98.4% | 98.4% |  |  |
| main_state_l12_off_long_selected_s900 | copy_trace_state_scheduled | paired_len8 | n/a | 60.9% | 1.3% | 100.0% | 100.0% | 100.0% | 94.2% | 100.0% | 94.5% | 60.9% | 78.7% | 60.9% | 78.7% | 92.2% | 60.9% | 92.2% | 92.2% |  |  |
| main_state_l12_off_long_selected_s900 | copy_trace_state_scheduled | paired_len12 | n/a | 61.7% | 1.0% | 100.0% | 100.0% | 100.0% | 95.7% | 100.0% | 95.8% | 60.2% | 80.9% | 60.2% | 79.8% | 89.1% | 60.9% | 89.1% | 89.1% |  |  |
| main_state_l12_off_long_selected_s900 | copy_trace_state_scheduled | paired_len24 | n/a | 18.0% | 1.0% | 100.0% | 100.0% | 98.0% | 91.3% | 98.3% | 91.3% | 16.4% | 49.4% | 16.4% | 49.3% | 21.9% | 6.2% | 20.3% | 20.3% |  |  |
| main_state_l12_w025_long_selected_s900 | copy_trace_state_scheduled | standard_len4 | n/a | 92.2% | 8.3% | 100.0% | 100.0% | 100.0% | 97.7% | 100.0% | 98.0% | 92.2% | 94.9% | 92.2% | 94.9% |  |  |  |  |  |  |
| main_state_l12_w025_long_selected_s900 | copy_trace_state_scheduled | standard_len8 | n/a | 67.2% | 1.3% | 100.0% | 100.0% | 100.0% | 95.5% | 100.0% | 94.9% | 67.2% | 81.2% | 67.2% | 81.2% |  |  |  |  |  |  |
| main_state_l12_w025_long_selected_s900 | copy_trace_state_scheduled | standard_len12 | n/a | 43.8% | 1.0% | 100.0% | 100.0% | 100.0% | 94.1% | 100.0% | 94.1% | 43.8% | 72.0% | 43.8% | 72.0% |  |  |  |  |  |  |
| main_state_l12_w025_long_selected_s900 | copy_trace_state_scheduled | standard_len24 | n/a | 31.2% | 1.0% | 100.0% | 100.0% | 99.5% | 95.0% | 99.7% | 94.7% | 29.7% | 63.2% | 29.7% | 63.1% |  |  |  |  |  |  |
| main_state_l12_w025_long_selected_s900 | copy_trace_state_scheduled | paraphrase_len4 | n/a | 81.2% | 8.0% | 100.0% | 100.0% | 100.0% | 95.3% | 100.0% | 95.3% | 81.2% | 88.3% | 81.2% | 88.3% |  |  |  |  |  |  |
| main_state_l12_w025_long_selected_s900 | copy_trace_state_scheduled | paraphrase_len8 | n/a | 62.5% | 1.2% | 100.0% | 100.0% | 100.0% | 94.5% | 100.0% | 94.1% | 62.5% | 82.0% | 62.5% | 82.0% |  |  |  |  |  |  |
| main_state_l12_w025_long_selected_s900 | copy_trace_state_scheduled | paraphrase_len12 | n/a | 60.9% | 1.0% | 100.0% | 100.0% | 100.0% | 96.0% | 100.0% | 96.1% | 60.9% | 76.6% | 60.9% | 76.6% |  |  |  |  |  |  |
| main_state_l12_w025_long_selected_s900 | copy_trace_state_scheduled | paraphrase_len24 | n/a | 3.1% | 1.0% | 100.0% | 100.0% | 97.2% | 88.3% | 97.2% | 88.3% | 0.0% | 46.2% | 0.0% | 45.9% |  |  |  |  |  |  |
| main_state_l12_w025_long_selected_s900 | copy_trace_state_scheduled | paired_len4 | n/a | 86.7% | 7.6% | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 96.3% | 86.7% | 93.2% | 86.7% | 93.2% | 98.4% | 85.9% | 98.4% | 98.4% |  |  |
| main_state_l12_w025_long_selected_s900 | copy_trace_state_scheduled | paired_len8 | n/a | 61.7% | 1.4% | 100.0% | 100.0% | 100.0% | 94.4% | 100.0% | 94.6% | 61.7% | 79.1% | 61.7% | 79.1% | 92.2% | 60.9% | 92.2% | 92.2% |  |  |
| main_state_l12_w025_long_selected_s900 | copy_trace_state_scheduled | paired_len12 | n/a | 58.6% | 1.0% | 100.0% | 100.0% | 100.0% | 95.6% | 100.0% | 95.8% | 57.8% | 79.6% | 57.8% | 79.0% | 89.1% | 56.2% | 89.1% | 89.1% |  |  |
| main_state_l12_w025_long_selected_s900 | copy_trace_state_scheduled | paired_len24 | n/a | 15.6% | 1.0% | 100.0% | 100.0% | 98.0% | 91.1% | 98.2% | 91.5% | 15.6% | 49.4% | 15.6% | 49.4% | 20.3% | 3.1% | 18.8% | 20.3% |  |  |
| main_state_w025_selected_s900 | copy_trace_state_scheduled | standard_len4 | n/a | 92.2% | 8.4% | 100.0% | 100.0% | 100.0% | 98.0% | 100.0% | 98.4% | 92.2% | 94.9% | 92.2% | 94.9% |  |  |  |  |  |  |
| main_state_w025_selected_s900 | copy_trace_state_scheduled | standard_len8 | n/a | 64.1% | 1.3% | 100.0% | 100.0% | 100.0% | 94.9% | 100.0% | 94.5% | 64.1% | 78.1% | 64.1% | 78.1% |  |  |  |  |  |  |
| main_state_w025_selected_s900 | copy_trace_state_scheduled | standard_len12 | n/a | 45.3% | 1.0% | 100.0% | 100.0% | 100.0% | 94.3% | 100.0% | 94.3% | 45.3% | 71.0% | 45.3% | 71.0% |  |  |  |  |  |  |
| main_state_w025_selected_s900 | copy_trace_state_scheduled | standard_len24 | n/a | 32.8% | 1.0% | 100.0% | 100.0% | 100.0% | 94.7% | 100.0% | 95.0% | 32.8% | 62.2% | 32.8% | 61.8% |  |  |  |  |  |  |
| main_state_w025_selected_s900 | copy_trace_state_scheduled | paraphrase_len4 | n/a | 75.0% | 7.6% | 100.0% | 100.0% | 100.0% | 93.8% | 100.0% | 94.1% | 75.0% | 84.4% | 75.0% | 84.4% |  |  |  |  |  |  |
| main_state_w025_selected_s900 | copy_trace_state_scheduled | paraphrase_len8 | n/a | 60.9% | 1.2% | 100.0% | 100.0% | 100.0% | 93.9% | 100.0% | 94.3% | 60.9% | 81.2% | 60.9% | 81.2% |  |  |  |  |  |  |
| main_state_w025_selected_s900 | copy_trace_state_scheduled | paraphrase_len12 | n/a | 53.1% | 1.0% | 100.0% | 100.0% | 100.0% | 94.4% | 100.0% | 95.3% | 53.1% | 74.1% | 53.1% | 74.1% |  |  |  |  |  |  |
| main_state_w025_selected_s900 | copy_trace_state_scheduled | paraphrase_len24 | n/a | 0.0% | 1.0% | 100.0% | 100.0% | 98.0% | 86.4% | 98.2% | 87.1% | 0.0% | 41.0% | 0.0% | 41.0% |  |  |  |  |  |  |
| main_state_w025_selected_s900 | copy_trace_state_scheduled | paired_len4 | n/a | 86.7% | 7.5% | 100.0% | 100.0% | 100.0% | 96.5% | 100.0% | 96.5% | 86.7% | 93.2% | 86.7% | 93.2% | 96.9% | 85.9% | 96.9% | 96.9% |  |  |
| main_state_w025_selected_s900 | copy_trace_state_scheduled | paired_len8 | n/a | 60.2% | 1.3% | 100.0% | 100.0% | 100.0% | 93.8% | 100.0% | 94.0% | 60.2% | 78.1% | 60.2% | 78.1% | 82.8% | 59.4% | 82.8% | 82.8% |  |  |
| main_state_w025_selected_s900 | copy_trace_state_scheduled | paired_len12 | n/a | 55.5% | 1.0% | 100.0% | 100.0% | 100.0% | 94.7% | 100.0% | 95.0% | 54.7% | 79.2% | 54.7% | 78.6% | 71.9% | 46.9% | 71.9% | 71.9% |  |  |
| main_state_w025_selected_s900 | copy_trace_state_scheduled | paired_len24 | n/a | 12.5% | 1.0% | 100.0% | 100.0% | 98.5% | 89.5% | 99.0% | 90.3% | 12.5% | 45.0% | 12.5% | 44.9% | 4.7% | 0.0% | 4.7% | 4.7% |  |  |
| smoke_qwen3_4b_scheduled_state | copy_trace_state_scheduled | standard_len2 | n/a | 0.0% | 9.0% | 50.0% | 75.0% | 100.0% | 62.5% | 0.0% | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% |  |  |  |  |  |  |
| smoke_qwen3_4b_scheduled_state | copy_trace_state_scheduled | standard_len4 | n/a | 0.0% | 0.8% | 75.0% | 75.0% | 62.5% | 43.8% | 0.0% | 25.0% | 0.0% | 0.0% | 0.0% | 0.0% |  |  |  |  |  |  |
| smoke_qwen3_4b_scheduled_state | copy_trace_state_scheduled | paraphrase_len2 | n/a | 25.0% | 8.8% | 75.0% | 75.0% | 100.0% | 62.5% | 12.5% | 75.0% | 25.0% | 25.0% | 25.0% | 25.0% |  |  |  |  |  |  |
| smoke_qwen3_4b_scheduled_state | copy_trace_state_scheduled | paraphrase_len4 | n/a | 0.0% | 1.1% | 100.0% | 25.0% | 68.8% | 56.2% | 0.0% | 37.5% | 0.0% | 25.0% | 0.0% | 25.0% |  |  |  |  |  |  |
| smoke_qwen3_4b_scheduled_state | copy_trace_state_scheduled | paired_len2 | n/a | 50.0% | 5.7% | 75.0% | 87.5% | 100.0% | 68.8% | 6.2% | 62.5% | 50.0% | 56.2% | 50.0% | 56.2% | 50.0% | 25.0% | 50.0% | 50.0% |  |  |
| smoke_qwen3_4b_scheduled_state | copy_trace_state_scheduled | paired_len4 | n/a | 0.0% | 2.7% | 75.0% | 87.5% | 75.0% | 40.6% | 3.1% | 34.4% | 0.0% | 6.2% | 0.0% | 6.2% | 50.0% | 0.0% | 50.0% | 50.0% |  |  |
| smoke_tiny_scheduled_state | copy_trace_state_scheduled | standard_len2 | n/a | 0.0% | 0.7% | 25.0% | 0.0% | 62.5% | 0.0% | 0.0% | 0.0% | 0.0% | 12.5% | 0.0% | 12.5% |  |  |  |  |  |  |
| smoke_tiny_scheduled_state | copy_trace_state_scheduled | standard_len3 | n/a | 25.0% | 4.8% | 25.0% | 0.0% | 66.7% | 0.0% | 0.0% | 0.0% | 0.0% | 25.0% | 0.0% | 16.7% |  |  |  |  |  |  |
| smoke_tiny_scheduled_state | copy_trace_state_scheduled | paraphrase_len2 | n/a | 0.0% | 0.0% | 25.0% | 0.0% | 62.5% | 0.0% | 0.0% | 0.0% | 0.0% | 25.0% | 0.0% | 25.0% |  |  |  |  |  |  |
| smoke_tiny_scheduled_state | copy_trace_state_scheduled | paraphrase_len3 | n/a | 0.0% | 0.1% | 25.0% | 0.0% | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% | 8.3% | 0.0% | 8.3% |  |  |  |  |  |  |
| smoke_tiny_scheduled_state | copy_trace_state_scheduled | paired_len2 | n/a | 12.5% | 12.8% | 50.0% | 0.0% | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% | 18.8% | 0.0% | 12.5% | 50.0% | 0.0% | 50.0% | 50.0% |  |  |
| smoke_tiny_scheduled_state | copy_trace_state_scheduled | paired_len3 | n/a | 0.0% | 0.0% | 62.5% | 0.0% | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% | 4.2% | 0.0% | 4.2% | 25.0% | 0.0% | 25.0% | 25.0% |  |  |

## Selected Checkpoints

| run | variant | selection_metric | selection_value | step | tag | state_loss_schedule | selected_paired_len24_executor_accuracy | selected_paired_len24_compiler_pair_state_consistency | selected_standard_len24_executor_accuracy | selected_paraphrase_len24_executor_accuracy | final_paired_len24_executor_accuracy | final_paired_len24_compiler_pair_state_consistency | final_standard_len24_executor_accuracy | final_paraphrase_len24_executor_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main_no_state_selected_s900 | copy_trace | paired_len24_executor_accuracy | 27.3% | 800 | eval_trace_long | none | 27.3% | 43.8% | 35.9% | 12.5% | 15.6% | 17.2% | 32.8% | 6.2% |
| main_state_l12_off_long_selected_s900 | copy_trace_state_scheduled | paired_len24_executor_accuracy | 22.7% | 800 | eval_trace_state_scheduled_long | short=1.0,medium=1.0,train=1.0,long=0.0 | 22.7% | 32.8% | 34.4% | 4.7% | 18.0% | 20.3% | 37.5% | 6.2% |
| main_state_l12_w025_long_selected_s900 | copy_trace_state_scheduled | paired_len24_executor_accuracy | 30.5% | 800 | eval_trace_state_scheduled_long | short=1.0,medium=1.0,train=1.0,long=0.25 | 30.5% | 75.0% | 39.1% | 21.9% | 15.6% | 20.3% | 31.2% | 3.1% |
| main_state_w025_selected_s900 | copy_trace_state_scheduled | paired_len24_executor_accuracy | 30.5% | 800 | eval_trace_state_scheduled_long | constant | 30.5% | 60.9% | 34.4% | 26.6% | 12.5% | 4.7% | 32.8% | 0.0% |
| smoke_qwen3_4b_scheduled_state | copy_trace_state_scheduled | paired_len4_executor_accuracy | 0.0% | 1 | eval_trace_state_scheduled_short | short=1.0,long=0.0 |  |  |  |  |  |  |  |  |
| smoke_tiny_scheduled_state | copy_trace_state_scheduled | paired_len3_executor_accuracy | 0.0% | 1 | eval_trace_state_scheduled_short | short=1.0,long=0.0 |  |  |  |  |  |  |  |  |

## Fresh Selected-Checkpoint Retest

| run | state_loss_schedule | split | executor_accuracy | program_exact | state_prefix_fraction | compiler_pair_state_consistency | executor_pair_both_correct |
|---|---|---|---|---|---|---|---|
| main_no_state_selected_s900 | none | fresh_standard_len24 | 32.0% | 31.2% | 57.4% |  |  |
| main_no_state_selected_s900 | none | fresh_paraphrase_len24 | 19.5% | 18.4% | 54.4% |  |  |
| main_no_state_selected_s900 | none | fresh_paired_len24 | 25.0% | 24.2% | 55.1% | 40.6% | 17.2% |
| main_state_w025_selected_s900 | constant | fresh_standard_len24 | 34.0% | 33.6% | 59.1% |  |  |
| main_state_w025_selected_s900 | constant | fresh_paraphrase_len24 | 31.2% | 30.1% | 57.8% |  |  |
| main_state_w025_selected_s900 | constant | fresh_paired_len24 | 32.8% | 31.8% | 57.8% | 55.9% | 29.7% |
| main_state_l12_off_long_selected_s900 | short=1.0,medium=1.0,train=1.0,long=0.0 | fresh_standard_len24 | 31.2% | 29.7% | 58.1% |  |  |
| main_state_l12_off_long_selected_s900 | short=1.0,medium=1.0,train=1.0,long=0.0 | fresh_paraphrase_len24 | 7.4% | 7.0% | 52.8% |  |  |
| main_state_l12_off_long_selected_s900 | short=1.0,medium=1.0,train=1.0,long=0.0 | fresh_paired_len24 | 19.1% | 18.4% | 54.2% | 29.3% | 6.2% |
| main_state_l12_w025_long_selected_s900 | short=1.0,medium=1.0,train=1.0,long=0.25 | fresh_standard_len24 | 34.0% | 32.8% | 59.4% |  |  |
| main_state_l12_w025_long_selected_s900 | short=1.0,medium=1.0,train=1.0,long=0.25 | fresh_paraphrase_len24 | 28.1% | 27.7% | 57.7% |  |  |
| main_state_l12_w025_long_selected_s900 | short=1.0,medium=1.0,train=1.0,long=0.25 | fresh_paired_len24 | 32.2% | 31.4% | 57.1% | 70.7% | 30.1% |

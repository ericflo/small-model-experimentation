# Teacher-Distilled Slot Compiler Analysis Summary

## Final Metrics

| run | variant | split | direct_accuracy | executor_accuracy | executor_target_mass | init_accuracy | init_pos_accuracy | op_accuracy | arg_accuracy | op_pos_accuracy | arg_pos_accuracy | program_exact | state_accuracy | state_all_exact | state_prefix_fraction | executor_pair_answer_consistency | executor_pair_both_correct | compiler_pair_program_consistency | compiler_pair_state_consistency | direct_pair_answer_consistency | direct_pair_both_correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main_control_light_state_s900 | copy_trace_state_scheduled | standard_len4 | n/a | 92.2% | 8.4% | 100.0% | 100.0% | 100.0% | 98.0% | 100.0% | 98.4% | 92.2% | 95.7% | 92.2% | 95.7% |  |  |  |  |  |  |
| main_control_light_state_s900 | copy_trace_state_scheduled | standard_len8 | n/a | 60.9% | 1.3% | 100.0% | 100.0% | 100.0% | 94.7% | 100.0% | 94.3% | 60.9% | 76.6% | 60.9% | 76.6% |  |  |  |  |  |  |
| main_control_light_state_s900 | copy_trace_state_scheduled | standard_len12 | n/a | 46.9% | 1.0% | 100.0% | 100.0% | 100.0% | 94.3% | 100.0% | 94.3% | 46.9% | 72.5% | 46.9% | 72.5% |  |  |  |  |  |  |
| main_control_light_state_s900 | copy_trace_state_scheduled | standard_len24 | n/a | 37.5% | 1.0% | 100.0% | 100.0% | 100.0% | 95.2% | 100.0% | 95.4% | 37.5% | 64.7% | 37.5% | 64.3% |  |  |  |  |  |  |
| main_control_light_state_s900 | copy_trace_state_scheduled | paraphrase_len4 | n/a | 81.2% | 7.9% | 100.0% | 100.0% | 100.0% | 95.3% | 100.0% | 95.7% | 81.2% | 87.5% | 81.2% | 87.5% |  |  |  |  |  |  |
| main_control_light_state_s900 | copy_trace_state_scheduled | paraphrase_len8 | n/a | 62.5% | 1.2% | 100.0% | 100.0% | 100.0% | 94.7% | 100.0% | 93.9% | 62.5% | 82.6% | 62.5% | 82.6% |  |  |  |  |  |  |
| main_control_light_state_s900 | copy_trace_state_scheduled | paraphrase_len12 | n/a | 59.4% | 1.0% | 100.0% | 100.0% | 100.0% | 95.6% | 100.0% | 95.8% | 59.4% | 75.3% | 59.4% | 75.3% |  |  |  |  |  |  |
| main_control_light_state_s900 | copy_trace_state_scheduled | paraphrase_len24 | n/a | 14.1% | 1.0% | 100.0% | 100.0% | 99.9% | 92.3% | 100.0% | 92.2% | 12.5% | 47.5% | 12.5% | 47.3% |  |  |  |  |  |  |
| main_control_light_state_s900 | copy_trace_state_scheduled | paired_len4 | n/a | 86.7% | 7.5% | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 96.5% | 86.7% | 93.2% | 86.7% | 93.2% | 98.4% | 85.9% | 98.4% | 98.4% |  |  |
| main_control_light_state_s900 | copy_trace_state_scheduled | paired_len8 | n/a | 61.7% | 1.3% | 100.0% | 100.0% | 100.0% | 94.1% | 100.0% | 94.2% | 61.7% | 79.1% | 61.7% | 79.1% | 87.5% | 60.9% | 87.5% | 87.5% |  |  |
| main_control_light_state_s900 | copy_trace_state_scheduled | paired_len12 | n/a | 59.4% | 1.0% | 100.0% | 100.0% | 100.0% | 95.6% | 100.0% | 95.7% | 58.6% | 80.6% | 58.6% | 80.0% | 85.9% | 54.7% | 85.9% | 85.9% |  |  |
| main_control_light_state_s900 | copy_trace_state_scheduled | paired_len24 | n/a | 23.4% | 1.0% | 100.0% | 100.0% | 99.8% | 93.5% | 99.9% | 93.8% | 22.7% | 50.9% | 22.7% | 50.8% | 45.3% | 15.6% | 45.3% | 45.3% |  |  |
| main_teacher_slot_distill_s900 | copy_trace_state_teacher | standard_len4 | n/a | 90.6% | 8.4% | 100.0% | 100.0% | 100.0% | 97.7% | 100.0% | 98.4% | 90.6% | 94.5% | 90.6% | 94.5% |  |  |  |  |  |  |
| main_teacher_slot_distill_s900 | copy_trace_state_teacher | standard_len8 | n/a | 65.6% | 1.3% | 100.0% | 100.0% | 100.0% | 95.3% | 100.0% | 94.9% | 65.6% | 79.7% | 65.6% | 79.7% |  |  |  |  |  |  |
| main_teacher_slot_distill_s900 | copy_trace_state_teacher | standard_len12 | n/a | 45.3% | 1.0% | 100.0% | 100.0% | 100.0% | 94.1% | 100.0% | 94.3% | 45.3% | 72.1% | 45.3% | 72.1% |  |  |  |  |  |  |
| main_teacher_slot_distill_s900 | copy_trace_state_teacher | standard_len24 | n/a | 35.9% | 1.0% | 100.0% | 100.0% | 100.0% | 95.5% | 100.0% | 95.3% | 35.9% | 65.6% | 35.9% | 65.6% |  |  |  |  |  |  |
| main_teacher_slot_distill_s900 | copy_trace_state_teacher | paraphrase_len4 | n/a | 87.5% | 8.1% | 100.0% | 100.0% | 100.0% | 96.9% | 100.0% | 97.3% | 87.5% | 93.4% | 87.5% | 93.4% |  |  |  |  |  |  |
| main_teacher_slot_distill_s900 | copy_trace_state_teacher | paraphrase_len8 | n/a | 64.1% | 1.2% | 100.0% | 100.0% | 100.0% | 94.9% | 100.0% | 94.3% | 64.1% | 84.0% | 64.1% | 84.0% |  |  |  |  |  |  |
| main_teacher_slot_distill_s900 | copy_trace_state_teacher | paraphrase_len12 | n/a | 59.4% | 1.0% | 100.0% | 100.0% | 100.0% | 95.7% | 100.0% | 96.0% | 59.4% | 76.0% | 59.4% | 76.0% |  |  |  |  |  |  |
| main_teacher_slot_distill_s900 | copy_trace_state_teacher | paraphrase_len24 | n/a | 17.2% | 1.0% | 100.0% | 100.0% | 99.8% | 92.8% | 99.9% | 93.0% | 17.2% | 48.2% | 17.2% | 48.2% |  |  |  |  |  |  |
| main_teacher_slot_distill_s900 | copy_trace_state_teacher | paired_len4 | n/a | 85.9% | 7.6% | 100.0% | 100.0% | 100.0% | 96.5% | 100.0% | 96.9% | 85.9% | 92.6% | 85.9% | 92.6% | 100.0% | 85.9% | 100.0% | 100.0% |  |  |
| main_teacher_slot_distill_s900 | copy_trace_state_teacher | paired_len8 | n/a | 62.5% | 1.4% | 100.0% | 100.0% | 100.0% | 94.6% | 100.0% | 94.8% | 62.5% | 79.5% | 62.5% | 79.5% | 90.6% | 60.9% | 90.6% | 90.6% |  |  |
| main_teacher_slot_distill_s900 | copy_trace_state_teacher | paired_len12 | n/a | 60.2% | 1.0% | 100.0% | 100.0% | 100.0% | 95.8% | 100.0% | 96.0% | 59.4% | 80.5% | 59.4% | 79.9% | 82.8% | 56.2% | 82.8% | 82.8% |  |  |
| main_teacher_slot_distill_s900 | copy_trace_state_teacher | paired_len24 | n/a | 22.7% | 1.0% | 100.0% | 100.0% | 99.8% | 93.9% | 99.9% | 94.3% | 22.7% | 52.4% | 22.7% | 52.4% | 64.1% | 17.2% | 64.1% | 64.1% |  |  |
| main_teacher_softpos_low_s900 | copy_trace_state_teacher | standard_len4 | n/a | 92.2% | 8.4% | 100.0% | 100.0% | 100.0% | 97.7% | 100.0% | 97.7% | 92.2% | 94.5% | 92.2% | 94.5% |  |  |  |  |  |  |
| main_teacher_softpos_low_s900 | copy_trace_state_teacher | standard_len8 | n/a | 64.1% | 1.3% | 100.0% | 100.0% | 100.0% | 95.1% | 100.0% | 94.5% | 64.1% | 78.1% | 64.1% | 78.1% |  |  |  |  |  |  |
| main_teacher_softpos_low_s900 | copy_trace_state_teacher | standard_len12 | n/a | 43.8% | 1.0% | 100.0% | 100.0% | 100.0% | 94.0% | 100.0% | 94.3% | 43.8% | 72.0% | 43.8% | 72.0% |  |  |  |  |  |  |
| main_teacher_softpos_low_s900 | copy_trace_state_teacher | standard_len24 | n/a | 37.5% | 1.0% | 100.0% | 100.0% | 100.0% | 95.4% | 100.0% | 95.3% | 37.5% | 64.9% | 37.5% | 64.5% |  |  |  |  |  |  |
| main_teacher_softpos_low_s900 | copy_trace_state_teacher | paraphrase_len4 | n/a | 81.2% | 8.0% | 100.0% | 100.0% | 100.0% | 94.9% | 100.0% | 95.7% | 81.2% | 89.5% | 81.2% | 89.5% |  |  |  |  |  |  |
| main_teacher_softpos_low_s900 | copy_trace_state_teacher | paraphrase_len8 | n/a | 62.5% | 1.2% | 100.0% | 100.0% | 100.0% | 94.7% | 100.0% | 94.3% | 62.5% | 83.2% | 62.5% | 83.2% |  |  |  |  |  |  |
| main_teacher_softpos_low_s900 | copy_trace_state_teacher | paraphrase_len12 | n/a | 57.8% | 1.0% | 100.0% | 100.0% | 100.0% | 95.6% | 100.0% | 95.4% | 57.8% | 74.0% | 57.8% | 74.0% |  |  |  |  |  |  |
| main_teacher_softpos_low_s900 | copy_trace_state_teacher | paraphrase_len24 | n/a | 14.1% | 1.0% | 100.0% | 100.0% | 99.8% | 92.3% | 100.0% | 91.9% | 14.1% | 46.6% | 14.1% | 46.6% |  |  |  |  |  |  |
| main_teacher_softpos_low_s900 | copy_trace_state_teacher | paired_len4 | n/a | 86.7% | 7.5% | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 96.7% | 86.7% | 93.2% | 86.7% | 93.2% | 98.4% | 85.9% | 98.4% | 98.4% |  |  |
| main_teacher_softpos_low_s900 | copy_trace_state_teacher | paired_len8 | n/a | 62.5% | 1.4% | 100.0% | 100.0% | 100.0% | 94.4% | 100.0% | 94.2% | 62.5% | 80.4% | 62.5% | 80.4% | 89.1% | 60.9% | 89.1% | 89.1% |  |  |
| main_teacher_softpos_low_s900 | copy_trace_state_teacher | paired_len12 | n/a | 60.2% | 1.0% | 100.0% | 100.0% | 100.0% | 95.1% | 100.0% | 94.9% | 58.6% | 80.8% | 58.6% | 79.6% | 81.2% | 56.2% | 81.2% | 81.2% |  |  |
| main_teacher_softpos_low_s900 | copy_trace_state_teacher | paired_len24 | n/a | 18.8% | 1.0% | 100.0% | 100.0% | 99.8% | 93.5% | 99.9% | 93.5% | 18.8% | 50.2% | 18.8% | 50.2% | 46.9% | 9.4% | 46.9% | 46.9% |  |  |
| smoke_qwen3_4b_teacher | copy_trace_state_teacher | standard_len2 | n/a | 0.0% | 9.4% | 50.0% | 0.0% | 100.0% | 62.5% | 25.0% | 75.0% | 0.0% | 0.0% | 0.0% | 0.0% |  |  |  |  |  |  |
| smoke_qwen3_4b_teacher | copy_trace_state_teacher | standard_len4 | n/a | 0.0% | 0.9% | 75.0% | 0.0% | 75.0% | 50.0% | 12.5% | 18.8% | 0.0% | 6.2% | 0.0% | 6.2% |  |  |  |  |  |  |
| smoke_qwen3_4b_teacher | copy_trace_state_teacher | paraphrase_len2 | n/a | 25.0% | 9.0% | 75.0% | 25.0% | 100.0% | 75.0% | 37.5% | 37.5% | 25.0% | 25.0% | 25.0% | 25.0% |  |  |  |  |  |  |
| smoke_qwen3_4b_teacher | copy_trace_state_teacher | paraphrase_len4 | n/a | 0.0% | 1.1% | 100.0% | 75.0% | 62.5% | 43.8% | 6.2% | 18.8% | 0.0% | 6.2% | 0.0% | 6.2% |  |  |  |  |  |  |
| smoke_qwen3_4b_teacher | copy_trace_state_teacher | paired_len2 | n/a | 25.0% | 6.1% | 75.0% | 12.5% | 100.0% | 50.0% | 18.8% | 31.2% | 25.0% | 37.5% | 25.0% | 37.5% | 50.0% | 25.0% | 50.0% | 50.0% |  |  |
| smoke_qwen3_4b_teacher | copy_trace_state_teacher | paired_len4 | n/a | 0.0% | 3.1% | 75.0% | 25.0% | 71.9% | 46.9% | 15.6% | 15.6% | 0.0% | 3.1% | 0.0% | 3.1% | 0.0% | 0.0% | 0.0% | 0.0% |  |  |
| smoke_tiny_teacher | copy_trace_state_teacher | standard_len2 | n/a | 0.0% | 0.7% | 50.0% | 0.0% | 62.5% | 0.0% | 0.0% | 0.0% | 0.0% | 12.5% | 0.0% | 12.5% |  |  |  |  |  |  |
| smoke_tiny_teacher | copy_trace_state_teacher | standard_len3 | n/a | 0.0% | 4.4% | 25.0% | 0.0% | 66.7% | 0.0% | 0.0% | 0.0% | 0.0% | 16.7% | 0.0% | 16.7% |  |  |  |  |  |  |
| smoke_tiny_teacher | copy_trace_state_teacher | paraphrase_len2 | n/a | 0.0% | 0.1% | 50.0% | 0.0% | 62.5% | 0.0% | 0.0% | 0.0% | 0.0% | 25.0% | 0.0% | 25.0% |  |  |  |  |  |  |
| smoke_tiny_teacher | copy_trace_state_teacher | paraphrase_len3 | n/a | 0.0% | 0.2% | 25.0% | 0.0% | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |  |  |  |  |  |  |
| smoke_tiny_teacher | copy_trace_state_teacher | paired_len2 | n/a | 0.0% | 12.7% | 37.5% | 0.0% | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% | 6.2% | 0.0% | 6.2% | 75.0% | 0.0% | 75.0% | 75.0% |  |  |
| smoke_tiny_teacher | copy_trace_state_teacher | paired_len3 | n/a | 0.0% | 0.0% | 62.5% | 0.0% | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 25.0% | 0.0% | 25.0% | 25.0% |  |  |

## Selected Checkpoints

| run | variant | selection_metric | selection_value | step | tag | state_loss_schedule | teacher_position_loss_weight | teacher_rep_loss_weight | selected_paired_len24_executor_accuracy | selected_paired_len24_compiler_pair_state_consistency | selected_standard_len24_executor_accuracy | selected_paraphrase_len24_executor_accuracy | final_paired_len24_executor_accuracy | final_paired_len24_compiler_pair_state_consistency | final_standard_len24_executor_accuracy | final_paraphrase_len24_executor_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main_control_light_state_s900 | copy_trace_state_scheduled | paired_len24_executor_accuracy | 30.5% | 800 | eval_trace_state_scheduled_long | constant | 0.0% | 0.0% | 30.5% | 67.2% | 37.5% | 25.0% | 23.4% | 45.3% | 37.5% | 14.1% |
| main_teacher_slot_distill_s900 | copy_trace_state_teacher | paired_len24_executor_accuracy | 28.1% | 800 | eval_trace_state_teacher_long | constant | 10.0% | 5.0% | 28.1% | 59.4% | 37.5% | 21.9% | 22.7% | 64.1% | 35.9% | 17.2% |
| main_teacher_softpos_low_s900 | copy_trace_state_teacher | paired_len24_executor_accuracy | 23.4% | 800 | eval_trace_state_teacher_long | constant | 3.0% | 0.0% | 23.4% | 32.8% | 29.7% | 15.6% | 18.8% | 46.9% | 37.5% | 14.1% |
| smoke_qwen3_4b_teacher | copy_trace_state_teacher | paired_len4_executor_accuracy | 0.0% | 1 | eval_trace_state_teacher_short | constant | 10.0% | 5.0% |  |  |  |  |  |  |  |  |
| smoke_tiny_teacher | copy_trace_state_teacher | paired_len3_executor_accuracy | 0.0% | 1 | eval_trace_state_teacher_short | constant | 50.0% | 10.0% |  |  |  |  |  |  |  |  |

## Fresh Selected-Checkpoint Retest

| run | state_loss_schedule | teacher_position_loss_weight | teacher_rep_loss_weight | split | executor_accuracy | program_exact | state_prefix_fraction | compiler_pair_state_consistency | executor_pair_both_correct |
|---|---|---|---|---|---|---|---|---|---|
| main_control_light_state_s900 | constant | 0.0% | 0.0% | fresh_standard_len24 | 30.5% | 30.1% | 57.1% |  |  |
| main_control_light_state_s900 | constant | 0.0% | 0.0% | fresh_paraphrase_len24 | 27.7% | 26.6% | 54.8% |  |  |
| main_control_light_state_s900 | constant | 0.0% | 0.0% | fresh_paired_len24 | 27.1% | 26.8% | 54.8% | 72.7% | 25.8% |
| main_teacher_slot_distill_s900 | constant | 10.0% | 5.0% | fresh_standard_len24 | 30.5% | 29.7% | 56.9% |  |  |
| main_teacher_slot_distill_s900 | constant | 10.0% | 5.0% | fresh_paraphrase_len24 | 28.1% | 27.0% | 55.9% |  |  |
| main_teacher_slot_distill_s900 | constant | 10.0% | 5.0% | fresh_paired_len24 | 27.9% | 27.3% | 55.5% | 55.5% | 26.2% |
| main_teacher_softpos_low_s900 | constant | 3.0% | 0.0% | fresh_standard_len24 | 27.0% | 26.2% | 56.0% |  |  |
| main_teacher_softpos_low_s900 | constant | 3.0% | 0.0% | fresh_paraphrase_len24 | 11.7% | 10.5% | 52.2% |  |  |
| main_teacher_softpos_low_s900 | constant | 3.0% | 0.0% | fresh_paired_len24 | 18.4% | 17.0% | 52.7% | 41.0% | 11.3% |

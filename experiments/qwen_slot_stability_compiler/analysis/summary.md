# Qwen Slot-Stability Compiler Analysis Summary

## Final Metrics

| run | variant | split | direct_accuracy | executor_accuracy | executor_target_mass | init_accuracy | init_pos_accuracy | op_accuracy | arg_accuracy | op_pos_accuracy | arg_pos_accuracy | program_exact | executor_pair_answer_consistency | executor_pair_both_correct | compiler_pair_program_consistency | direct_pair_answer_consistency | direct_pair_both_correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control_qwen3_4b_qlora_answer_only_mixed_l12_s600 | direct | standard_len4 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_mixed_l12_s600 | direct | standard_len8 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_mixed_l12_s600 | direct | standard_len12 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_mixed_l12_s600 | direct | standard_len24 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_mixed_l12_s600 | direct | paraphrase_len4 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_mixed_l12_s600 | direct | paraphrase_len8 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_mixed_l12_s600 | direct | paraphrase_len12 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_mixed_l12_s600 | direct | paraphrase_len24 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  |  |  |
| control_qwen3_4b_qlora_answer_only_mixed_l12_s600 | direct | paired_len4 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  | 90.6% | 0.0% |
| control_qwen3_4b_qlora_answer_only_mixed_l12_s600 | direct | paired_len8 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  | 96.9% | 1.6% |
| control_qwen3_4b_qlora_answer_only_mixed_l12_s600 | direct | paired_len12 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  | 100.0% | 0.0% |
| control_qwen3_4b_qlora_answer_only_mixed_l12_s600 | direct | paired_len24 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |  | 100.0% | 1.6% |
| control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600 | copy_trace | standard_len4 | n/a | 85.9% | 8.3% | 100.0% | 100.0% | 100.0% | 96.5% | 100.0% | 96.9% | 85.9% |  |  |  |  |  |
| control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600 | copy_trace | standard_len8 | n/a | 60.9% | 1.3% | 100.0% | 100.0% | 100.0% | 94.5% | 100.0% | 94.3% | 60.9% |  |  |  |  |  |
| control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600 | copy_trace | standard_len12 | n/a | 34.4% | 1.0% | 100.0% | 100.0% | 100.0% | 93.2% | 100.0% | 93.2% | 34.4% |  |  |  |  |  |
| control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600 | copy_trace | standard_len24 | n/a | 6.2% | 1.0% | 100.0% | 100.0% | 96.2% | 87.4% | 94.9% | 87.0% | 6.2% |  |  |  |  |  |
| control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600 | copy_trace | paraphrase_len4 | n/a | 79.7% | 7.4% | 100.0% | 100.0% | 100.0% | 94.9% | 100.0% | 94.9% | 79.7% |  |  |  |  |  |
| control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600 | copy_trace | paraphrase_len8 | n/a | 59.4% | 1.2% | 100.0% | 100.0% | 100.0% | 93.9% | 100.0% | 93.4% | 59.4% |  |  |  |  |  |
| control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600 | copy_trace | paraphrase_len12 | n/a | 57.8% | 1.0% | 100.0% | 100.0% | 100.0% | 95.2% | 100.0% | 94.5% | 57.8% |  |  |  |  |  |
| control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600 | copy_trace | paraphrase_len24 | n/a | 15.6% | 1.0% | 100.0% | 100.0% | 100.0% | 92.6% | 100.0% | 92.2% | 15.6% |  |  |  |  |  |
| control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600 | copy_trace | paired_len4 | n/a | 85.9% | 7.0% | 100.0% | 100.0% | 100.0% | 96.5% | 100.0% | 96.5% | 85.9% | 93.8% | 84.4% | 93.8% |  |  |
| control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600 | copy_trace | paired_len8 | n/a | 60.2% | 1.3% | 100.0% | 100.0% | 100.0% | 93.9% | 100.0% | 93.9% | 60.2% | 87.5% | 57.8% | 87.5% |  |  |
| control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600 | copy_trace | paired_len12 | n/a | 57.8% | 1.0% | 100.0% | 100.0% | 99.9% | 94.8% | 100.0% | 94.3% | 56.2% | 68.8% | 53.1% | 68.8% |  |  |
| control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600 | copy_trace | paired_len24 | n/a | 17.2% | 1.0% | 100.0% | 100.0% | 97.7% | 89.8% | 96.8% | 89.4% | 15.6% | 14.1% | 7.8% | 14.1% |  |  |
| main_qwen3_4b_qlora_slot_stability_mixed_l12_s600 | copy_trace_stability | standard_len4 | n/a | 90.6% | 8.4% | 100.0% | 100.0% | 100.0% | 97.7% | 100.0% | 98.0% | 90.6% |  |  |  |  |  |
| main_qwen3_4b_qlora_slot_stability_mixed_l12_s600 | copy_trace_stability | standard_len8 | n/a | 59.4% | 1.3% | 100.0% | 100.0% | 100.0% | 94.7% | 100.0% | 94.7% | 59.4% |  |  |  |  |  |
| main_qwen3_4b_qlora_slot_stability_mixed_l12_s600 | copy_trace_stability | standard_len12 | n/a | 43.8% | 1.0% | 100.0% | 100.0% | 100.0% | 94.3% | 100.0% | 94.1% | 43.8% |  |  |  |  |  |
| main_qwen3_4b_qlora_slot_stability_mixed_l12_s600 | copy_trace_stability | standard_len24 | n/a | 0.0% | 1.0% | 100.0% | 100.0% | 96.6% | 87.0% | 94.7% | 87.0% | 0.0% |  |  |  |  |  |
| main_qwen3_4b_qlora_slot_stability_mixed_l12_s600 | copy_trace_stability | paraphrase_len4 | n/a | 79.7% | 7.4% | 100.0% | 100.0% | 100.0% | 94.9% | 100.0% | 95.3% | 79.7% |  |  |  |  |  |
| main_qwen3_4b_qlora_slot_stability_mixed_l12_s600 | copy_trace_stability | paraphrase_len8 | n/a | 59.4% | 1.2% | 100.0% | 100.0% | 100.0% | 93.9% | 100.0% | 94.1% | 59.4% |  |  |  |  |  |
| main_qwen3_4b_qlora_slot_stability_mixed_l12_s600 | copy_trace_stability | paraphrase_len12 | n/a | 56.2% | 1.0% | 100.0% | 100.0% | 100.0% | 94.8% | 100.0% | 95.3% | 56.2% |  |  |  |  |  |
| main_qwen3_4b_qlora_slot_stability_mixed_l12_s600 | copy_trace_stability | paraphrase_len24 | n/a | 23.4% | 1.0% | 100.0% | 100.0% | 99.9% | 93.1% | 100.0% | 92.6% | 21.9% |  |  |  |  |  |
| main_qwen3_4b_qlora_slot_stability_mixed_l12_s600 | copy_trace_stability | paired_len4 | n/a | 86.7% | 7.1% | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 96.5% | 86.7% | 96.9% | 85.9% | 96.9% |  |  |
| main_qwen3_4b_qlora_slot_stability_mixed_l12_s600 | copy_trace_stability | paired_len8 | n/a | 57.8% | 1.3% | 100.0% | 100.0% | 100.0% | 93.8% | 100.0% | 94.2% | 57.8% | 75.0% | 53.1% | 75.0% |  |  |
| main_qwen3_4b_qlora_slot_stability_mixed_l12_s600 | copy_trace_stability | paired_len12 | n/a | 58.6% | 1.0% | 100.0% | 100.0% | 100.0% | 95.0% | 100.0% | 94.7% | 57.0% | 68.8% | 53.1% | 68.8% |  |  |
| main_qwen3_4b_qlora_slot_stability_mixed_l12_s600 | copy_trace_stability | paired_len24 | n/a | 22.7% | 1.0% | 100.0% | 100.0% | 98.2% | 90.8% | 97.3% | 90.3% | 20.3% | 15.6% | 9.4% | 12.5% |  |  |

# Qwen LoRA Parser Compiler Analysis Summary

## Final Metrics

| run | variant | split | direct_accuracy | executor_accuracy | executor_target_mass | init_accuracy | init_pos_accuracy | op_accuracy | arg_accuracy | op_pos_accuracy | arg_pos_accuracy | program_exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control_qwen3_4b_direct_mixed_l12 | direct | standard_len4 | 3.1% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_mixed_l12 | direct | standard_len8 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_mixed_l12 | direct | standard_len12 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_mixed_l12 | direct | standard_len24 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_mixed_l12 | direct | paraphrase_len4 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_mixed_l12 | direct | paraphrase_len8 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_mixed_l12 | direct | paraphrase_len12 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_direct_mixed_l12 | direct | paraphrase_len24 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| control_qwen3_4b_qlora_answer_only_mixed_l12 | qlora_answer_only | standard_len4 | n/a | 0.0% | 1.0% | 0.0% | 0.0% | 34.0% | 2.7% | 0.4% | 0.0% | 0.0% |
| control_qwen3_4b_qlora_answer_only_mixed_l12 | qlora_answer_only | standard_len8 | n/a | 1.6% | 1.0% | 0.0% | 0.0% | 32.2% | 1.4% | 0.0% | 0.0% | 0.0% |
| control_qwen3_4b_qlora_answer_only_mixed_l12 | qlora_answer_only | standard_len12 | n/a | 3.1% | 1.0% | 3.1% | 3.1% | 35.2% | 1.7% | 0.0% | 0.4% | 0.0% |
| control_qwen3_4b_qlora_answer_only_mixed_l12 | qlora_answer_only | standard_len24 | n/a | 3.1% | 1.0% | 0.0% | 0.0% | 34.2% | 2.1% | 0.1% | 0.0% | 0.0% |
| control_qwen3_4b_qlora_answer_only_mixed_l12 | qlora_answer_only | paraphrase_len4 | n/a | 0.0% | 1.0% | 0.0% | 0.0% | 30.9% | 1.6% | 6.2% | 2.7% | 0.0% |
| control_qwen3_4b_qlora_answer_only_mixed_l12 | qlora_answer_only | paraphrase_len8 | n/a | 1.6% | 1.0% | 0.0% | 0.0% | 32.4% | 1.2% | 2.5% | 2.5% | 0.0% |
| control_qwen3_4b_qlora_answer_only_mixed_l12 | qlora_answer_only | paraphrase_len12 | n/a | 0.0% | 1.0% | 0.0% | 1.6% | 32.6% | 1.6% | 1.4% | 2.2% | 0.0% |
| control_qwen3_4b_qlora_answer_only_mixed_l12 | qlora_answer_only | paraphrase_len24 | n/a | 0.0% | 1.0% | 1.6% | 0.0% | 34.1% | 1.3% | 0.8% | 0.9% | 0.0% |
| main_qwen3_4b_qlora_trace_argstrong_mixed_l12 | qlora_trace | standard_len4 | n/a | 32.8% | 4.8% | 100.0% | 100.0% | 97.7% | 79.7% | 100.0% | 96.9% | 32.8% |
| main_qwen3_4b_qlora_trace_argstrong_mixed_l12 | qlora_trace | standard_len8 | n/a | 10.9% | 1.1% | 100.0% | 100.0% | 97.7% | 77.9% | 99.8% | 93.0% | 9.4% |
| main_qwen3_4b_qlora_trace_argstrong_mixed_l12 | qlora_trace | standard_len12 | n/a | 0.0% | 1.0% | 100.0% | 100.0% | 94.0% | 70.7% | 98.6% | 91.3% | 0.0% |
| main_qwen3_4b_qlora_trace_argstrong_mixed_l12 | qlora_trace | standard_len24 | n/a | 0.0% | 1.0% | 100.0% | 100.0% | 81.4% | 67.4% | 89.8% | 82.0% | 0.0% |
| main_qwen3_4b_qlora_trace_argstrong_mixed_l12 | qlora_trace | paraphrase_len4 | n/a | 40.6% | 4.9% | 100.0% | 100.0% | 96.9% | 82.4% | 100.0% | 96.5% | 40.6% |
| main_qwen3_4b_qlora_trace_argstrong_mixed_l12 | qlora_trace | paraphrase_len8 | n/a | 6.2% | 1.1% | 100.0% | 100.0% | 97.7% | 75.8% | 100.0% | 93.8% | 6.2% |
| main_qwen3_4b_qlora_trace_argstrong_mixed_l12 | qlora_trace | paraphrase_len12 | n/a | 4.7% | 1.0% | 100.0% | 100.0% | 90.5% | 69.3% | 99.2% | 87.8% | 0.0% |
| main_qwen3_4b_qlora_trace_argstrong_mixed_l12 | qlora_trace | paraphrase_len24 | n/a | 0.0% | 1.0% | 100.0% | 100.0% | 75.5% | 53.1% | 62.6% | 53.3% | 0.0% |
| pilot_qwen3_4b_qlora_tagger_mixed_l12 | qlora_tagger | standard_len4 | n/a | 17.2% | 4.5% | 96.9% | 100.0% | 98.0% | 76.2% | 100.0% | 89.5% | 17.2% |
| pilot_qwen3_4b_qlora_tagger_mixed_l12 | qlora_tagger | standard_len8 | n/a | 3.1% | 1.1% | 98.4% | 98.4% | 94.9% | 73.8% | 100.0% | 85.2% | 0.0% |
| pilot_qwen3_4b_qlora_tagger_mixed_l12 | qlora_tagger | standard_len12 | n/a | 0.0% | 1.0% | 96.9% | 100.0% | 95.6% | 68.8% | 100.0% | 82.4% | 0.0% |
| pilot_qwen3_4b_qlora_tagger_mixed_l12 | qlora_tagger | standard_len24 | n/a | 1.6% | 1.0% | 100.0% | 100.0% | 88.2% | 65.4% | 90.2% | 77.4% | 0.0% |
| pilot_qwen3_4b_qlora_tagger_mixed_l12 | qlora_tagger | paraphrase_len4 | n/a | 32.8% | 4.3% | 98.4% | 100.0% | 99.2% | 77.7% | 100.0% | 94.5% | 32.8% |
| pilot_qwen3_4b_qlora_tagger_mixed_l12 | qlora_tagger | paraphrase_len8 | n/a | 17.2% | 1.1% | 95.3% | 100.0% | 99.0% | 77.5% | 100.0% | 93.0% | 15.6% |
| pilot_qwen3_4b_qlora_tagger_mixed_l12 | qlora_tagger | paraphrase_len12 | n/a | 0.0% | 1.0% | 100.0% | 100.0% | 99.2% | 71.6% | 100.0% | 91.7% | 0.0% |
| pilot_qwen3_4b_qlora_tagger_mixed_l12 | qlora_tagger | paraphrase_len24 | n/a | 1.6% | 1.0% | 93.8% | 98.4% | 97.0% | 71.2% | 99.4% | 87.3% | 0.0% |
| smoke_qwen3_4b_qlora_tagger | qlora_tagger | standard_len2 | n/a | 0.0% | 7.3% | 0.0% | 0.0% | 25.0% | 0.0% | 25.0% | 25.0% | 0.0% |
| smoke_tiny_tagger | qlora_tagger | standard_len2 | n/a | 0.0% | 6.4% | 25.0% | 0.0% | 25.0% | 25.0% | 0.0% | 0.0% | 0.0% |
| smoke_tiny_tagger | qlora_tagger | standard_len4 | n/a | 12.5% | 5.7% | 0.0% | 0.0% | 31.2% | 12.5% | 31.2% | 3.1% | 0.0% |

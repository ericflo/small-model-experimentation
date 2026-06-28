# Qwen Structured Bridge Analysis Summary

## Final Metrics

| run | variant | split | direct_accuracy | executor_accuracy | executor_target_mass | init_accuracy | op_accuracy | arg_accuracy | program_exact |
|---|---|---|---|---|---|---|---|---|---|
| main_qwen35_numeric_spans | direct | len4 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_numeric_spans | direct | len8 | 1.2% | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_numeric_spans | direct | len12 | 0.4% | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_numeric_spans | compiler_trace | len4 | n/a | 100.0% | 99.8% | 100.0% | 100.0% | 100.0% | 100.0% |
| main_qwen35_numeric_spans | compiler_trace | len8 | n/a | 99.2% | 98.8% | 100.0% | 99.9% | 100.0% | 99.2% |
| main_qwen35_numeric_spans | compiler_trace | len12 | n/a | 95.7% | 94.1% | 100.0% | 99.6% | 100.0% | 95.7% |
| main_qwen35_numeric_spans | compiler_answer_only | len4 | n/a | 1.6% | 0.9% | 0.0% | 31.1% | 2.6% | 0.0% |
| main_qwen35_numeric_spans | compiler_answer_only | len8 | n/a | 0.0% | 0.9% | 1.2% | 32.9% | 2.4% | 0.0% |
| main_qwen35_numeric_spans | compiler_answer_only | len12 | n/a | 0.8% | 1.0% | 1.2% | 33.5% | 3.1% | 0.0% |
| pilot_qwen35_frozen_bridge | direct | len3 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a |
| pilot_qwen35_frozen_bridge | direct | len6 | 2.1% | n/a | n/a | n/a | n/a | n/a | n/a |
| pilot_qwen35_frozen_bridge | direct | len8 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a |
| pilot_qwen35_frozen_bridge | compiler_trace | len3 | n/a | 2.1% | 1.0% | 0.0% | 99.3% | 14.6% | 0.0% |
| pilot_qwen35_frozen_bridge | compiler_trace | len6 | n/a | 0.0% | 1.0% | 4.2% | 99.7% | 8.7% | 0.0% |
| pilot_qwen35_frozen_bridge | compiler_trace | len8 | n/a | 0.0% | 1.0% | 2.1% | 95.6% | 9.1% | 0.0% |
| pilot_qwen35_frozen_bridge | compiler_answer_only | len3 | n/a | 2.1% | 1.1% | 0.0% | 32.6% | 0.0% | 0.0% |
| pilot_qwen35_frozen_bridge | compiler_answer_only | len6 | n/a | 2.1% | 1.1% | 4.2% | 32.3% | 0.0% | 0.0% |
| pilot_qwen35_frozen_bridge | compiler_answer_only | len8 | n/a | 0.0% | 1.0% | 4.2% | 33.3% | 0.5% | 0.0% |
| pilot_qwen35_numeric_spans | direct | len3 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a |
| pilot_qwen35_numeric_spans | direct | len6 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a |
| pilot_qwen35_numeric_spans | direct | len8 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a |
| pilot_qwen35_numeric_spans | compiler_trace | len3 | n/a | 71.9% | 69.1% | 71.9% | 100.0% | 100.0% | 71.9% |
| pilot_qwen35_numeric_spans | compiler_trace | len6 | n/a | 76.6% | 72.0% | 85.9% | 99.0% | 99.5% | 76.6% |
| pilot_qwen35_numeric_spans | compiler_trace | len8 | n/a | 71.9% | 62.5% | 81.2% | 98.8% | 100.0% | 71.9% |
| pilot_qwen35_numeric_spans | compiler_answer_only | len3 | n/a | 0.0% | 1.1% | 3.1% | 40.1% | 0.0% | 0.0% |
| pilot_qwen35_numeric_spans | compiler_answer_only | len6 | n/a | 4.7% | 1.1% | 0.0% | 35.7% | 0.0% | 0.0% |
| pilot_qwen35_numeric_spans | compiler_answer_only | len8 | n/a | 0.0% | 1.0% | 1.6% | 33.4% | 0.2% | 0.0% |
| scale_qwen35_length24_trace | compiler_trace | len4 | n/a | 100.0% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| scale_qwen35_length24_trace | compiler_trace | len12 | n/a | 93.8% | 93.6% | 100.0% | 99.5% | 100.0% | 93.8% |
| scale_qwen35_length24_trace | compiler_trace | len16 | n/a | 92.2% | 91.5% | 100.0% | 99.6% | 100.0% | 92.2% |
| scale_qwen35_length24_trace | compiler_trace | len24 | n/a | 87.5% | 82.7% | 100.0% | 99.5% | 100.0% | 87.5% |
| smoke_tiny | direct | len2 | 6.2% | n/a | n/a | n/a | n/a | n/a | n/a |
| smoke_tiny | direct | len4 | 6.2% | n/a | n/a | n/a | n/a | n/a | n/a |
| smoke_tiny | compiler_trace | len2 | n/a | 6.2% | 6.2% | 12.5% | 40.6% | 3.1% | 0.0% |
| smoke_tiny | compiler_trace | len4 | n/a | 18.8% | 5.7% | 6.2% | 29.7% | 6.2% | 0.0% |
| smoke_tiny | compiler_answer_only | len2 | n/a | 0.0% | 6.3% | 0.0% | 25.0% | 0.0% | 0.0% |
| smoke_tiny | compiler_answer_only | len4 | n/a | 6.2% | 5.7% | 0.0% | 31.2% | 0.0% | 0.0% |
| smoke_tiny_numeric_spans | direct | len2 | 6.2% | n/a | n/a | n/a | n/a | n/a | n/a |
| smoke_tiny_numeric_spans | direct | len4 | 6.2% | n/a | n/a | n/a | n/a | n/a | n/a |
| smoke_tiny_numeric_spans | compiler_trace | len2 | n/a | 6.2% | 6.3% | 18.8% | 40.6% | 3.1% | 0.0% |
| smoke_tiny_numeric_spans | compiler_trace | len4 | n/a | 0.0% | 5.7% | 6.2% | 29.7% | 10.9% | 0.0% |
| smoke_tiny_numeric_spans | compiler_answer_only | len2 | n/a | 6.2% | 6.1% | 25.0% | 25.0% | 0.0% | 0.0% |
| smoke_tiny_numeric_spans | compiler_answer_only | len4 | n/a | 0.0% | 5.8% | 0.0% | 31.2% | 1.6% | 0.0% |

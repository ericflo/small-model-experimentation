# Qwen Trace Bootstrap Retention Analysis Summary

## Final Metrics

| run | variant | split | direct_accuracy | executor_accuracy | executor_target_mass | init_accuracy | op_accuracy | arg_accuracy | program_exact |
|---|---|---|---|---|---|---|---|---|---|
| main_qwen35_retention | direct | len4 | 0.8% | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_retention | direct | len8 | 0.8% | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_retention | direct | len12 | 1.2% | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_retention | direct | len24 | 0.4% | n/a | n/a | n/a | n/a | n/a | n/a |
| main_qwen35_retention | compiler_trace | len4 | n/a | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| main_qwen35_retention | compiler_trace | len8 | n/a | 100.0% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| main_qwen35_retention | compiler_trace | len12 | n/a | 99.2% | 98.9% | 100.0% | 99.9% | 100.0% | 99.2% |
| main_qwen35_retention | compiler_trace | len24 | n/a | 96.1% | 94.2% | 100.0% | 99.8% | 100.0% | 96.1% |
| main_qwen35_retention | compiler_answer_only | len4 | n/a | 0.8% | 1.0% | 0.8% | 33.4% | 0.0% | 0.0% |
| main_qwen35_retention | compiler_answer_only | len8 | n/a | 1.2% | 0.8% | 1.2% | 33.3% | 0.0% | 0.0% |
| main_qwen35_retention | compiler_answer_only | len12 | n/a | 1.2% | 0.9% | 0.8% | 33.1% | 0.0% | 0.0% |
| main_qwen35_retention | compiler_answer_only | len24 | n/a | 0.8% | 1.0% | 0.8% | 33.7% | 0.0% | 0.0% |
| main_qwen35_retention | compiler_trace_then_answer | len4 | n/a | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| main_qwen35_retention | compiler_trace_then_answer | len8 | n/a | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| main_qwen35_retention | compiler_trace_then_answer | len12 | n/a | 99.6% | 99.5% | 100.0% | 100.0% | 100.0% | 99.6% |
| main_qwen35_retention | compiler_trace_then_answer | len24 | n/a | 96.9% | 95.3% | 100.0% | 99.9% | 100.0% | 96.9% |
| main_qwen35_retention | compiler_trace_then_answer_low_lr | len4 | n/a | 100.0% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| main_qwen35_retention | compiler_trace_then_answer_low_lr | len8 | n/a | 100.0% | 99.8% | 100.0% | 100.0% | 100.0% | 100.0% |
| main_qwen35_retention | compiler_trace_then_answer_low_lr | len12 | n/a | 99.2% | 98.6% | 100.0% | 99.9% | 100.0% | 99.2% |
| main_qwen35_retention | compiler_trace_then_answer_low_lr | len24 | n/a | 95.7% | 92.8% | 100.0% | 99.8% | 100.0% | 95.7% |
| pilot_qwen35_retention | direct | len3 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a |
| pilot_qwen35_retention | direct | len6 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a |
| pilot_qwen35_retention | direct | len12 | 1.6% | n/a | n/a | n/a | n/a | n/a | n/a |
| pilot_qwen35_retention | compiler_trace | len3 | n/a | 100.0% | 98.2% | 100.0% | 100.0% | 100.0% | 100.0% |
| pilot_qwen35_retention | compiler_trace | len6 | n/a | 98.4% | 96.2% | 98.4% | 100.0% | 100.0% | 98.4% |
| pilot_qwen35_retention | compiler_trace | len12 | n/a | 92.2% | 91.3% | 93.8% | 99.7% | 100.0% | 92.2% |
| pilot_qwen35_retention | compiler_answer_only | len3 | n/a | 0.0% | 0.5% | 0.0% | 31.8% | 1.0% | 0.0% |
| pilot_qwen35_retention | compiler_answer_only | len6 | n/a | 3.1% | 0.9% | 0.0% | 29.9% | 0.8% | 0.0% |
| pilot_qwen35_retention | compiler_answer_only | len12 | n/a | 1.6% | 1.7% | 0.0% | 35.5% | 1.0% | 0.0% |
| pilot_qwen35_retention | compiler_trace_then_answer | len3 | n/a | 81.2% | 75.6% | 82.8% | 100.0% | 99.5% | 81.2% |
| pilot_qwen35_retention | compiler_trace_then_answer | len6 | n/a | 71.9% | 69.5% | 93.8% | 98.7% | 97.1% | 71.9% |
| pilot_qwen35_retention | compiler_trace_then_answer | len12 | n/a | 56.2% | 49.2% | 89.1% | 99.1% | 97.9% | 56.2% |
| pilot_qwen35_retention_lr | compiler_trace_then_answer | len3 | n/a | 76.6% | 73.9% | 82.8% | 100.0% | 97.4% | 76.6% |
| pilot_qwen35_retention_lr | compiler_trace_then_answer | len6 | n/a | 78.1% | 72.6% | 93.8% | 99.7% | 97.1% | 78.1% |
| pilot_qwen35_retention_lr | compiler_trace_then_answer | len12 | n/a | 56.2% | 50.8% | 89.1% | 99.1% | 97.7% | 56.2% |
| pilot_qwen35_retention_lr | compiler_trace_then_answer_low_lr | len3 | n/a | 79.7% | 73.9% | 82.8% | 100.0% | 98.4% | 79.7% |
| pilot_qwen35_retention_lr | compiler_trace_then_answer_low_lr | len6 | n/a | 73.4% | 61.1% | 93.8% | 98.4% | 97.9% | 73.4% |
| pilot_qwen35_retention_lr | compiler_trace_then_answer_low_lr | len12 | n/a | 59.4% | 33.5% | 89.1% | 97.4% | 98.8% | 57.8% |
| smoke_tiny | direct | len2 | 6.2% | n/a | n/a | n/a | n/a | n/a | n/a |
| smoke_tiny | direct | len4 | 6.2% | n/a | n/a | n/a | n/a | n/a | n/a |
| smoke_tiny | direct | len6 | 0.0% | n/a | n/a | n/a | n/a | n/a | n/a |
| smoke_tiny | compiler_trace | len2 | n/a | 0.0% | 6.2% | 12.5% | 40.6% | 0.0% | 0.0% |
| smoke_tiny | compiler_trace | len4 | n/a | 18.8% | 5.7% | 12.5% | 29.7% | 6.2% | 0.0% |
| smoke_tiny | compiler_trace | len6 | n/a | 6.2% | 5.9% | 12.5% | 38.5% | 10.4% | 0.0% |
| smoke_tiny | compiler_answer_only | len2 | n/a | 0.0% | 6.0% | 0.0% | 25.0% | 0.0% | 0.0% |
| smoke_tiny | compiler_answer_only | len4 | n/a | 0.0% | 5.8% | 6.2% | 31.2% | 0.0% | 0.0% |
| smoke_tiny | compiler_answer_only | len6 | n/a | 12.5% | 5.9% | 0.0% | 24.0% | 0.0% | 0.0% |
| smoke_tiny | compiler_trace_then_answer | len2 | n/a | 12.5% | 6.3% | 18.8% | 40.6% | 18.8% | 0.0% |
| smoke_tiny | compiler_trace_then_answer | len4 | n/a | 0.0% | 5.7% | 12.5% | 29.7% | 21.9% | 0.0% |
| smoke_tiny | compiler_trace_then_answer | len6 | n/a | 6.2% | 5.9% | 25.0% | 38.5% | 15.6% | 0.0% |

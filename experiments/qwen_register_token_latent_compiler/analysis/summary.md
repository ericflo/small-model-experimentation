# Qwen Register-Token Latent Compiler Analysis Summary

Primary run: `main_register_trace_s600`

## Final Metrics

| variant | split | direct | executor | mass | init | op | arg | program | prefix | pair both | pair state consistency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| register_trace | standard_len4 | n/a | 88.3% | 70.3% | 88.3% | 100.0% | 100.0% | 88.3% | 88.3% | n/a | n/a |
| register_trace | standard_len8 | n/a | 94.5% | 69.2% | 94.5% | 100.0% | 100.0% | 94.5% | 94.5% | n/a | n/a |
| register_trace | standard_len12 | n/a | 94.5% | 67.5% | 94.5% | 100.0% | 100.0% | 94.5% | 94.5% | n/a | n/a |
| register_trace | standard_len24 | n/a | 21.9% | 1.5% | 91.4% | 97.3% | 95.2% | 21.1% | 83.9% | n/a | n/a |
| register_trace | paraphrase_len4 | n/a | 90.6% | 71.5% | 90.6% | 100.0% | 100.0% | 90.6% | 90.6% | n/a | n/a |
| register_trace | paraphrase_len8 | n/a | 91.4% | 72.4% | 91.4% | 100.0% | 100.0% | 91.4% | 91.4% | n/a | n/a |
| register_trace | paraphrase_len12 | n/a | 89.1% | 63.7% | 89.1% | 100.0% | 100.0% | 89.1% | 89.1% | n/a | n/a |
| register_trace | paraphrase_len24 | n/a | 3.1% | 1.1% | 89.1% | 92.5% | 88.5% | 2.3% | 72.9% | n/a | n/a |
| register_trace | paired_len4 | n/a | 87.9% | 70.5% | 87.9% | 100.0% | 100.0% | 87.9% | 87.9% | 87.5% | 99.2% |
| register_trace | paired_len8 | n/a | 96.1% | 73.3% | 96.1% | 100.0% | 100.0% | 96.1% | 96.1% | 96.1% | 99.2% |
| register_trace | paired_len12 | n/a | 91.0% | 69.1% | 91.0% | 100.0% | 100.0% | 91.0% | 91.0% | 90.6% | 98.4% |
| register_trace | paired_len24 | n/a | 12.5% | 1.4% | 92.2% | 94.7% | 92.4% | 12.1% | 79.5% | 1.6% | 3.1% |

## All Runs

| run | variant | split | direct | executor | program | pair both |
|---|---|---|---:|---:|---:|---:|
| control_direct_answer_s300 | direct | standard_len24 | 3.1% | n/a | n/a | n/a |
| control_direct_answer_s300 | direct | paraphrase_len24 | 0.0% | n/a | n/a | n/a |
| control_direct_answer_s300 | direct | paired_len24 | 1.6% | n/a | n/a | n/a |
| control_register_answer_only_s300 | register_answer_only | standard_len24 | n/a | 1.6% | 0.0% | n/a |
| control_register_answer_only_s300 | register_answer_only | paraphrase_len24 | n/a | 0.0% | 0.0% | n/a |
| control_register_answer_only_s300 | register_answer_only | paired_len24 | n/a | 0.0% | 0.0% | 0.0% |
| main_register_trace_s600 | register_trace | standard_len24 | n/a | 21.9% | 21.1% | n/a |
| main_register_trace_s600 | register_trace | paraphrase_len24 | n/a | 3.1% | 2.3% | n/a |
| main_register_trace_s600 | register_trace | paired_len24 | n/a | 12.5% | 12.1% | 1.6% |
| pilot_frozen_inline_register_trace_s120 | register_trace | standard_len24 | n/a | 0.0% | 0.0% | n/a |
| pilot_frozen_inline_register_trace_s120 | register_trace | paraphrase_len24 | n/a | 6.2% | 0.0% | n/a |
| pilot_frozen_inline_register_trace_s120 | register_trace | paired_len24 | n/a | 1.6% | 0.0% | 0.0% |
| pilot_frozen_register_trace_s120 | register_trace | standard_len24 | n/a | 0.0% | 0.0% | n/a |
| pilot_frozen_register_trace_s120 | register_trace | paraphrase_len24 | n/a | 3.1% | 0.0% | n/a |
| pilot_frozen_register_trace_s120 | register_trace | paired_len24 | n/a | 1.6% | 0.0% | 0.0% |
| pilot_lora_bare_initstrong_s300 | register_trace | standard_len24 | n/a | 0.0% | 0.0% | n/a |
| pilot_lora_bare_initstrong_s300 | register_trace | paraphrase_len24 | n/a | 0.0% | 0.0% | n/a |
| pilot_lora_bare_initstrong_s300 | register_trace | paired_len24 | n/a | 3.1% | 0.0% | 0.0% |
| pilot_lora_named_register_trace_s240 | register_trace | standard_len24 | n/a | 0.0% | 0.0% | n/a |
| pilot_lora_named_register_trace_s240 | register_trace | paraphrase_len24 | n/a | 6.2% | 0.0% | n/a |
| pilot_lora_named_register_trace_s240 | register_trace | paired_len24 | n/a | 0.0% | 0.0% | 0.0% |
| pilot_lora_register_trace_s180 | register_trace | standard_len24 | n/a | 3.1% | 0.0% | n/a |
| pilot_lora_register_trace_s180 | register_trace | paraphrase_len24 | n/a | 0.0% | 0.0% | n/a |
| pilot_lora_register_trace_s180 | register_trace | paired_len24 | n/a | 3.1% | 0.0% | 0.0% |

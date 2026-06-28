# Experiment Log

## 2026-06-26

- Created standalone constrained coverage-DPO experiment package.
- Primary question: can the coverage-efficiency signal from hard-negative preference learning survive explicit pass@1, parseability, and reference-distribution constraints?
- Primary gate: constrained DPO must beat or match sample-more on the coverage/pass@1/forward-token Pareto frontier. Beating K=4 alone is not enough.
- Guardrails:
  - parse successes per task must stay near base;
  - pass@1 proxy must stay within 2 percentage points of tuned-hot base;
  - shuffled constrained-DPO must not match the real constrained-DPO arm.
- If the gate fails, the result will be used to choose the next mechanism, not to stop the research program.

### Setup and Training

- Copied the real-code sampling/evaluation utilities into this standalone package and redirected all model outputs to `/workspace/large_artifacts/qwen35_4b_constrained_coverage_dpo`.
- Rebuilt hard-negative preference pairs from the local training sample pool:
  - real pairs: 58 pairs across 20 tasks;
  - shuffled control: same pairs with labels shuffled;
  - visible-wrong pair rate: 13.8%.
- Trained two ten-step QLoRA adapters:
  - `constrained_dpo_lora`;
  - `constrained_shuffled_dpo_lora`.
- Objective: DPO margin term plus positive-sample NLL anchor plus reference-logprob drift penalty.

### Evaluation

- Evaluated on MBPP test, offset 0, 24 tasks, one visible test, temperature 1.0, top-p 0.98.
- Arms:
  - base hot K=4;
  - base hot K=8 sample-more reference;
  - constrained DPO K=4;
  - constrained shuffled-DPO K=4.

### Result

| arm | K | coverage@K | pass@1 proxy | parse / task | functional diversity | forward tokens |
|---|---:|---:|---:|---:|---:|---:|
| base_hot_k4 | 4 | 58.3% | 37.5% | 3.54 | 51.0% | 23434 |
| base_hot_k8_sample_more | 8 | 66.7% | 41.7% | 7.50 | 32.1% | 45406 |
| constrained_dpo_k4 | 4 | 62.5% | 41.7% | 3.58 | 54.9% | 22411 |
| constrained_shuffled_dpo_k4 | 4 | 58.3% | 25.0% | 3.54 | 55.6% | 25575 |

- Constrained DPO beat base K=4 by one task and beat shuffled by one task while preserving pass@1 and parseability.
- It did not reach the K=8 sample-more coverage reference.
- Task overlap showed complementarity:
  - constrained K=4 recovered task 25, missed by base K=8;
  - base K=8 recovered tasks 33 and 34, missed by constrained K=4;
  - base K=8 union constrained K=4 covered 17/24 tasks.

### Readout

- Formal gate: no scale-up of this exact scalar constrained-DPO sampler yet, because it did not beat sample-more coverage.
- Useful signal: this constrained update preserved pass@1/parseability and beat the shuffled control.
- Next direction: test a sampler portfolio or learned scheduler over base-hot, constrained-DPO, and other generation policies, judged by the same coverage/pass@1/forward-token Pareto gate.

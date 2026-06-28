# Experiment Log

## 2026-06-26

- Created standalone sampler-portfolio scheduler experiment package.
- Primary question: can complementary generation policies improve the coverage/pass@1/forward-token Pareto frontier compared with simply sampling more from one policy?
- Policies under test:
  - base hot sampling;
  - mixed-temperature base sampling;
  - constrained preference LoRA sampling.
- Planned readouts:
  - static portfolio coverage and token cost;
  - oracle best-block headroom after a short base-hot prefix;
  - learned scheduler using only deployable prefix features;
  - pass@1 and parseability guardrails.
- Success criterion: a deployable portfolio or learned scheduler must beat the base sample-more reference on the coverage/pass@1/forward-token Pareto frontier.

### Iteration Notes

- Started a fresh `train_base_mixed_k8` generation run for 36 MBPP train tasks.
- Stopped the run after 4/36 tasks because K=8 mixed sampling was taking roughly one minute per task, which would have delayed the scheduler test by hours before producing any new information.
- Pivoted to complete candidate pools copied into this package under `data/`. The package is standalone: all analysis scripts read only local files in this experiment directory.
- Broad source-policy pool:
  - train: `data/train_policy_pool_records.jsonl`;
  - eval: `data/eval_policy_union_records.jsonl`.
- Constrained-arm subset:
  - `data/subset_base_hot_k4_records.jsonl`;
  - `data/subset_base_hot_k8_records.jsonl`;
  - `data/subset_constrained_dpo_k4_records.jsonl`.

### Evaluations Run

- `scripts/evaluate_source_portfolios.py`
  - trained a linear scheduler on 80 train tasks using prompt and two-sample prefix features;
  - evaluated base/static/learned/oracle source-policy schedules on 80 eval tasks.
- `scripts/evaluate_constrained_subset.py`
  - evaluated static base-hot/constrained portfolios and oracle arm choice on 24 matched tasks.
- `scripts/diagnose_constrained_scheduler.py`
  - ran a leave-one-task-out scheduler diagnostic for stop vs hot-next vs constrained choice.
- `scripts/make_report.py`
  - wrote the final report and figures.

### Results

Broad source-policy run:

| arm | coverage | pass@1 | forward tokens |
|---|---:|---:|---:|
| base_prefix_k4 | 70.0% | 61.3% | 62270 |
| prefix2_mid4 | 70.0% | 61.3% | 57013 |
| learned_scheduler_after_prefix2 | 65.0% | 61.3% | 51517 |
| oracle_best_block_after_prefix2 | 71.2% | 61.3% | 40154 |
| full_union_all_candidates | 86.2% | 61.3% | 504344 |

Constrained-arm subset:

| arm | coverage | pass@1 | forward tokens |
|---|---:|---:|---:|
| subset_base_hot_k4 | 58.3% | 37.5% | 23434 |
| subset_base_hot_k8 | 66.7% | 41.7% | 44219 |
| subset_constrained_k4 | 62.5% | 41.7% | 21785 |
| subset_hot4_plus_constrained4 | 66.7% | 37.5% | 45219 |
| subset_visible_gate_hot4_then_constrained | 62.5% | 37.5% | 29184 |
| subset_oracle_choose_arm | 75.0% | 37.5% | 22223 |

Scheduler diagnostic:

- Source scheduler labels: stop 52, low4 12, mid4 11, high4 5.
- Source learned scheduler actions: stop 55, low4 11, mid4 11, high4 3.
- Constrained leave-one-task-out labels: stop 16, hot_next4 7, constrained4 1.
- Constrained leave-one-task-out result: 62.5% coverage at 30495 forward tokens.

### Readout

- No deployable scheduler or static portfolio beat the single-policy sample-more reference in this pilot.
- The constrained-arm oracle chooser shows real arm-selection headroom: 75.0% coverage at 22223 forward tokens versus base hot K=8 at 66.7% and 44219 tokens.
- The simple feature schedulers did not recover that headroom.
- Next direction: collect a larger matched multi-policy training set and train a policy-value estimator that predicts per-arm marginal coverage from prompt plus cheap prefix evidence.

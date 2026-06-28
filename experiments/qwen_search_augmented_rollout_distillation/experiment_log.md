# Experiment Log

## 2026-06-24

- Created a fresh standalone experiment directory.
- Selected intervention: search-augmented rollout distillation for a dense-state
  recurrent VM agent.
- Core idea: after behavior cloning on oracle traces, roll out the learned
  policy, run bounded answer-verified repair search from policy-visited states,
  and train on the first repair edit plus pairwise positive-vs-negative action
  ranking.
- Main failure targeted: local edit competence without reliable global
  trajectory selection.
- Large artifacts will be stored under
  `large_artifacts/qwen_search_augmented_rollout_distillation/checkpoints/`.
- Implemented the standalone harness:
  - behavior cloning from gold VM traces,
  - on-policy state collection,
  - bounded answer-verified repair search from each visited state,
  - first-repair-action supervision,
  - same-state positive-vs-negative action ranking,
  - greedy, value-gated, forced-step, and value-beam evaluation modes.
- Static syntax check passed for the training script and shared VM core.
- Smoke run `smoke_search_rollout_distill_20260624` completed end to end.
  It exposed a weak search-label source: shallow two-edit repair found verified
  completions for only 1 of 12 policy-visited states.
- Added bounded target-guided completion: candidate first actions are evaluated
  by whether they can still reach a verified answer within the shortest
  remaining edit budget.
- Smoke run `smoke_search_rollout_distill_guided_20260624` completed end to
  end. Search found verified completions for 11 of 11 policy-visited states,
  so the pilot will use guided repair with `repair_max_edits=12`.
- Pilot `pilot_search_r2_rank05_20260624` was stopped after the second-round
  state collection because the first search-distilled policy did not improve
  rollout accuracy and the second-round collection regressed.
  - BC K=8 best deployable accuracy: val mixed 25.0%, fresh standard 8.3%,
    fresh paraphrase 8.3%, paired 0.0%, hard 8.3%.
  - Search-r1 K=8 best deployable accuracy: val mixed 16.7%, fresh standard
    8.3%, fresh paraphrase 8.3%, paired 0.0%, hard 8.3%.
  - Search-r1 state collection found verified completions for 492/502 states,
    but one epoch of rank-weight 0.5 training only reached 53.6% pairwise
    rank accuracy and reduced action accuracy.
  - Search-r2 collection had 46 false-stop states and lower rollout success,
    so the run was intentionally interrupted.
- Added `--eval_modes` so subsequent pilots can skip expensive value-beam
  sweeps during iteration.
- Pilot `pilot_search_r1_rank00_e2_20260624` completed with pairwise ranking
  disabled and two search-distillation epochs.
  - Search-r1 K=8 improved best deployable accuracy over BC on fresh paired
    (0.0% to 16.7%), fresh paraphrase (8.3% to 16.7%), and fresh standard
    (8.3% to 16.7%).
  - Hard composition stayed flat at 8.3% and val mixed stayed flat at 16.7%.
  - Search-r1 action accuracy recovered to 39.6%, better than the rank-0.5
    pilot but still below BC action accuracy.
- Main recipe selected: rank loss disabled, one search-augmented on-policy
  round, two search epochs, no value-beam in the broad evaluation sweep.
- Main run `main_search_r1_rank00_e2_20260624` completed.
  - Native Qwen accuracy: val mixed 43.8%, fresh standard 50.0%, fresh
    paraphrase 25.0%, fresh paired 25.0%, hard composition 50.0%.
  - BC VM active K>0 best: val mixed 6.2%, fresh standard 12.5%, fresh
    paraphrase 37.5%, fresh paired 18.8%, hard composition 37.5%.
  - Search VM active K>0 best: val mixed 12.5%, fresh standard 12.5%, fresh
    paraphrase 31.2%, fresh paired 18.8%, hard composition 18.8%.
  - Oracle teacher best: val mixed 100.0%, fresh standard 87.5%, fresh
    paraphrase 93.8%, fresh paired 100.0%, hard composition 68.8%.
  - Search-state repair found verified completions for 787/802 states, but
    search retraining reduced local action accuracy from 60.0% to 50.4% and
    did not improve the active VM controller.
- Generated charts under `analysis/` and standalone reports under `reports/`.

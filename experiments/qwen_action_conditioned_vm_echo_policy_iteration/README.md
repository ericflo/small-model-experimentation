# Qwen Action-Conditioned VM-ECHO Policy Iteration

**Status:** finished

This standalone experiment tests whether a frozen-Qwen bytecode compiler can
improve by learning the consequences of its own generated candidate programs.

## Hypothesis

If the model proposes a candidate bytecode program, executing that candidate in
the VM gives dense action-conditioned feedback: validity, final value, stack
trace, and whether the candidate solves the prompt. A consequence model trained
on those observations should rank candidates better than raw compiler logprob,
and distilling the best learned candidates back into the compiler should improve
direct bytecode emission.

## Layout

- `src/typed_bytecode_core.py`: standalone task generator, bytecode VM, typed
  decoding, and candidate search.
- `src/qwen_action_conditioned_vm_echo_policy_iteration_experiment.py`: Qwen
  feature extraction, compiler training, candidate generation, consequence
  model training, learned reranking, and policy distillation.
- `src/analyze_qwen_action_conditioned_vm_echo_policy_iteration.py`: aggregate
  metrics, charts, Markdown report, and HTML report.
- `runs/`: per-run manifests, logs, and metrics.
- `analysis/`: aggregate CSVs and figures.
- `reports/`: final writeups.
- `large_artifacts/qwen_action_conditioned_vm_echo_policy_iteration/checkpoints/`:
  checkpoints kept outside the experiment directory.

## Primary Metrics

- Greedy direct executable accuracy.
- Learned candidate-selection accuracy.
- Answer-verified oracle/search accuracy.
- Oracle gap recovered by learned consequence ranking.
- Distilled compiler direct accuracy after training on learned-selected
  candidates.


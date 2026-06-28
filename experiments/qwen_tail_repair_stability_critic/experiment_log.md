# Experiment Log: Qwen Tail Repair Stability Critic

## Objective

Test whether a learned tail-repair critic can improve exact length-24 executable
accuracy while reducing seed-to-seed variance. The experiment is standalone:
its report describes only this experiment's setup, results, and conclusions.

## Design Commitments

- Fresh experiment directory: `experiments/qwen_tail_repair_stability_critic/`.
- Large artifacts separated under `large_artifacts/qwen_tail_repair_stability_critic/`.
- Multi-seed result reporting from the start.
- Stability is a co-equal gate: mean accuracy alone is not enough.
- Candidate selection cannot use target answers or target states at inference.
- Include shuffled-label control.
- Produce Markdown and HTML reports with figures.

## Iterations

### 2026-06-25 Initial Scaffold

Created standalone experiment directories and recorded the intended design.

### 2026-06-25 Smoke Iteration 1

Ran a one-source smoke suite. Candidate export, critic training, control
training, chart generation, Markdown report generation, and HTML report
generation completed.

Findings from the smoke:

- The true critic path can train and evaluate.
- The shuffled-label path is active and can damage performance, which is useful
  as a control signal.
- The initial report aggregation duplicated base/prior rows once per label
  mode.
- The first accuracy chart selected the global minimum critic seed, which hid
  learned-critic bars when base rows used critic seed `-1`.

Patched aggregation and chart filtering before any main run.

### 2026-06-25 Main Iteration 1

Ran `main_tail_repair_critic_v1` across three frozen source compiler seeds.

Candidate coverage was high enough to make the repair task meaningful:

- `standard_L24` base executable accuracy: 44.8%.
- `standard_L24` answer-positive candidate fraction: 91.1%.
- `standard_L24` state-exact candidate fraction: 50.0%.
- Average candidate count: 177 per example.

The learned unweighted critics did not move off the base candidate. Answer and
state critics across seeds all matched the no-repair baseline on standard L24
with zero changed candidates. The candidate ceiling was high, but the critic
input was too aggregate: it did not expose enough concrete tail-slot program
content to identify the correct edit.

Patched the runner for iteration 2:

- Added explicit fixed-window tail-slot features for each candidate.
- Added `train_focus=recoverable_balanced`, which emphasizes base-wrong groups
  with at least one positive candidate while keeping preserve/impossible
  examples for damage control.
- Added utility checkpoint selection: validation accuracy plus recovery bonus
  minus damage penalty.
- Added `cache_run_name` so iteration 2 can reuse the expensive v1 candidate
  caches without re-exporting Qwen outputs.

### 2026-06-25 Main Iteration 2

Ran `main_tail_repair_critic_focus_v2` from the cached candidate groups with
answer labels only, explicit tail-slot features, recoverable-balanced training,
and utility-selected checkpoints.

The focused critics learned to make edits in later epochs, but those edits
damaged already-correct programs more often than they recovered wrong programs.
The validation utility therefore selected the initial no-change checkpoint for
all three true-label critic seeds.

Final standard-L24 gate:

- No repair: 44.8% mean exact accuracy, 44.6% source-seed std.
- Iteration-1 critic: 44.8% mean exact accuracy, 44.6% source-seed std.
- Iteration-2 focused critic: 44.8% mean exact accuracy, 44.6% source-seed std.
- Shuffled-label focused critic: 30.4% mean exact accuracy.
- Answer-oracle candidate selector: 91.1% mean exact accuracy.

Conclusion: candidate coverage is not the bottleneck in this task, but this
small feature-based critic did not learn a safe selector. The stability gate
failed because neither mean exact accuracy nor source-seed variance improved.

### Final Artifacts

- Markdown report:
  `reports/qwen_tail_repair_stability_critic_report.md`
- HTML report:
  `reports/qwen_tail_repair_stability_critic_report.html`
- Figures:
  `reports/figures/`
- Large candidate caches and critic checkpoints:
  `/workspace/large_artifacts/qwen_tail_repair_stability_critic/`

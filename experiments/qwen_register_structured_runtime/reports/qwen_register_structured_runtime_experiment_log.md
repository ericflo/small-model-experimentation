# Qwen Register-Token Structured Runtime Experiment Log

## Objective

Train and evaluate a Qwen-attached register compiler whose predicted slots are
executed by a fixed cyclic modulo runtime. The compiler may read only appended
register-token hidden states. The key test is whether full state-trajectory
supervision plus paired consistency improves long-chain exact execution and
prompt-invariant latent trajectories.

## Success Criteria

- Keep this experiment standalone with its own source, reports, analysis, run
  metadata, and checkpoint manifest.
- Store large checkpoints under `large_artifacts/`.
- Run smoke, pilot, and main configurations instead of relying on a single run.
- Evaluate standard, paraphrase, and paired length generalization.
- Compare the structured state-consistency condition against at least one
  control that removes a load-bearing training signal.

## Runs

### Smoke: Frozen Structured State Consistency

`smoke_frozen_structured_state_consistency`

- Frozen `Qwen/Qwen3-4B` backbone.
- Four-step register bank.
- Two optimizer steps with tiny train/eval sets.
- Purpose: validate appended register construction, the structured cyclic
  runtime, full state loss, paired consistency loss, metric writing, and
  checkpoint writing.
- Result: completed end to end. Accuracy was not expected to move under this
  tiny setup.

### Pilot: QLoRA Structured State Consistency

`pilot_structured_state_consistency_s240`

- QLoRA-adapted `Qwen/Qwen3-4B`.
- Full 24-step bare appended register bank.
- Trace loss, full state trajectory loss, final executor loss, and paired
  program/state consistency loss.
- Curriculum: 80 steps at lengths 1-4, 80 at lengths 1-8, and 80 at lengths
  1-12.
- Result: strong through length 12 before any long-chain training. Final exact
  execution was 85.9% on standard L12, 78.1% on paraphrase L12, and 82.0% on
  paired L12. Paired state consistency was 95.3% at L12. Length 24 moved above
  chance but remained weak: 7.8% standard, 9.4% paraphrase, and 9.4% paired
  exact execution.

Interpretation: the trajectory and consistency losses do not prevent register
learning. They produce prompt-stable trajectories through the trained length
range. Long-chain exactness still needs an explicit long-stage curriculum.

### Main: Structured State Consistency

`main_structured_trace_state_consistency_s600`

- QLoRA-adapted `Qwen/Qwen3-4B`.
- Full 24-step bare appended register bank.
- Register transformer width 512.
- Trace loss, final executor loss, full state trajectory loss, paired
  program-consistency loss, and paired state-consistency loss.
- Curriculum: 150 steps at lengths 1-4, 150 at 1-8, 150 at 1-12, and 150 at
  8-24.

Final metrics:

| Split | Executor exact | Program exact | Init | Op | Arg | Prefix | Pair both | Pair state consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard L4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| Standard L8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| Standard L12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| Standard L24 | 25.0% | 25.0% | 100.0% | 93.8% | 89.7% | 80.5% | n/a | n/a |
| Paraphrase L24 | 5.5% | 4.7% | 100.0% | 88.9% | 83.8% | 81.0% | n/a | n/a |
| Paired L24 | 11.7% | 10.2% | 100.0% | 89.7% | 85.7% | 79.1% | 1.6% | 1.6% |

Interpretation: the main run completely solves the trained length range and
partially improves length-24 standard execution. Long-chain exactness remains
fragile because high individual slot accuracy still compounds across 24 steps.
The paired L24 state-consistency metric remains low.

### Control: State Trace Without Paired Consistency

`control_trace_state_no_pair_s600`

- Same backbone, register bank, model width, curriculum, trace loss, executor
  loss, and full state trajectory loss.
- Removed paired program/state consistency losses.

Final length-24 metrics:

| Split | Executor exact | Program exact | Pair both | Pair state consistency |
|---|---:|---:|---:|---:|
| Standard L24 | 3.9% | 2.3% | n/a | n/a |
| Paraphrase L24 | 1.6% | 0.0% | n/a | n/a |
| Paired L24 | 1.6% | 0.8% | 0.0% | 0.0% |

Interpretation: paired consistency was not needed for perfect L12 performance,
but it was load-bearing for length-24 lift in this run. Without it, the model
also solved L4/L8/L12 but stayed near chance at length 24.

## Overall Interpretation

The experiment gives a narrow positive result. A fixed register-token interface
plus structured cyclic state supervision can make Qwen write executable
programs perfectly through length 12. Paired consistency materially improves
length-24 standard execution under the tested curriculum.

The negative result is equally important. The method still does not produce a
prompt-invariant length-24 latent program. L24 paired both-correct accuracy and
paired state consistency remain at 1.6% in the main run. The next change should
target long-chain repair or recurrent refinement of the register program, not
only stronger per-slot supervision.

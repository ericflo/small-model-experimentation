# Qwen Register-Token Latent Compiler Experiment Log

## Objective

Train and evaluate a Qwen-attached latent compiler that uses a fixed appended
register bank as its only program interface. The bridge may read register hidden
states, but it may not read hand-selected numeric or operation spans from the
prompt.

## Success Criteria

- Keep the experiment standalone with its own source, reports, analysis,
  run metadata, and checkpoint manifest.
- Store large checkpoints under `large_artifacts/`.
- Run smoke, pilot, and main configurations rather than relying on one run.
- Evaluate direct answer prediction, answer-only register discovery, and
  trace-supervised register compilation where feasible.
- Report standard, paraphrase, and paired length generalization.

## Runs

### Smoke: Frozen Register Trace

`smoke_frozen_register_trace`

- Frozen `Qwen/Qwen3-4B` backbone.
- Four-step register bank, two optimizer steps, tiny train/eval sets.
- Purpose: validate appended register construction, register hidden-state
  extraction, bridge training, strict execution metrics, checkpoint writing, and
  analysis aggregation.
- Result: completed end to end. Accuracy was not expected to move under this
  tiny two-step setup.

### Pilot: Frozen Full Register Trace

`pilot_frozen_register_trace_s120`

- Frozen `Qwen/Qwen3-4B` backbone.
- Full 24-step register bank.
- Trace-supervised bridge, 120 optimizer steps across lengths 1-12.
- Purpose: test whether the appended register positions already expose enough
  frozen-model information for a bridge to recover executable slots.
- Result: weak control. Fresh paired L24 exact execution was 1.6%, with 0.0%
  program exact and 0.0% paired state consistency. This indicates that the fixed
  register interface needs model adaptation, not only a trained readout.

### Pilot: LoRA Full Register Trace

`pilot_lora_register_trace_s180`

- QLoRA-adapted `Qwen/Qwen3-4B`.
- Bare appended 24-step register bank.
- Trace-supervised bridge, 180 optimizer steps across lengths 1-12.
- Purpose: test whether a small model adaptation can route prompt information
  into appended register slots.
- Result: operation and argument extraction began to work, but the initial-value
  register did not. On standard L24, operation accuracy was 86.1% and argument
  accuracy was 73.0%, while init accuracy was 0.0%, executor exact was 3.1%,
  and program exact was 0.0%.

### Pilot: Named Register Trace

`pilot_lora_named_register_trace_s240`

- QLoRA-adapted backbone.
- Semantically named appended register bank.
- Trace supervision with stronger init weighting.
- Purpose: test whether natural-language labels around each register slot make
  the latent interface easier to learn.
- Result: worse than the bare register bank. On standard L24, executor exact was
  0.0%, operation accuracy was 31.6%, and argument accuracy was 3.5%.

### Pilot: Bare Register Trace With Strong Init Weight

`pilot_lora_bare_initstrong_s300`

- QLoRA-adapted backbone.
- Bare appended register bank.
- Stronger init and argument trace weights.
- Purpose: test whether the initial-value failure could be fixed by
  reweighting the slot losses.
- Result: init improved only weakly and argument extraction regressed. On
  standard L24, executor exact was 0.0%, init accuracy was 3.1%, operation
  accuracy was 78.1%, and argument accuracy was 17.6%.

### Pilot: Frozen Inline Register Trace

`pilot_frozen_inline_register_trace_s120`

- Frozen backbone.
- Register markers placed next to the source lines rather than only in a suffix
  bank.
- Purpose: diagnostic control for whether proximity to the prompt text is enough
  to make frozen register states linearly decodable.
- Result: still weak. On standard L24, executor exact and program exact were
  0.0%; operation accuracy was 41.7% and argument accuracy was 5.3%.

### Main: Bare Register Trace

`main_register_trace_s600`

- QLoRA-adapted backbone.
- Bare appended 24-step register bank.
- Trace-supervised compiler with a one-layer register transformer, width 512.
- Curriculum: 150 steps at lengths 1-4, 150 at 1-8, 150 at 1-12, and 150 at
  8-24.
- Purpose: run the strongest register-interface configuration found by the
  pilots, with enough training for long-chain exposure.

Main final metrics:

| Split | Executor exact | Program exact | Init | Op | Arg | Prefix | Pair both | Pair state consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard L4 | 88.3% | 88.3% | 88.3% | 100.0% | 100.0% | 88.3% | n/a | n/a |
| Standard L8 | 94.5% | 94.5% | 94.5% | 100.0% | 100.0% | 94.5% | n/a | n/a |
| Standard L12 | 94.5% | 94.5% | 94.5% | 100.0% | 100.0% | 94.5% | n/a | n/a |
| Standard L24 | 21.9% | 21.1% | 91.4% | 97.3% | 95.2% | 83.9% | n/a | n/a |
| Paraphrase L24 | 3.1% | 2.3% | 89.1% | 92.5% | 88.5% | 72.9% | n/a | n/a |
| Paired L24 | 12.5% | 12.1% | 92.2% | 94.7% | 92.4% | 79.5% | 1.6% | 3.1% |

Interpretation: the register interface is trainable. The model learned to use
the appended marker positions as a program-writing surface, because the same
architecture without trace supervision stayed at chance. The long-chain result
is not solved: high per-slot accuracy at L24 still compounds into poor exact
program execution, especially under paraphrases and paired consistency.

### Control: Register Answer-Only

`control_register_answer_only_s300`

- QLoRA-adapted backbone.
- Same bare appended register bank and compiler architecture.
- Trained only through the final soft-executor answer loss, with no trace
  targets.
- Purpose: test whether final-answer supervision alone can discover the
  register program interface.
- Result: no. Standard L24 executor exact was 1.6%, paraphrase L24 was 0.0%,
  paired L24 was 0.0%, and program exact was 0.0% on all length-24 splits.

### Control: Direct Answer Head

`control_direct_answer_s300`

- QLoRA-adapted backbone.
- No register compiler.
- A direct MLP head reads the answer-marker hidden state and predicts the final
  value modulo 97.
- Purpose: test whether the base hidden state plus a small supervised head can
  solve the same answer task under a comparable training budget.
- Result: no. Standard L24 direct accuracy was 3.1%, paraphrase L24 was 0.0%,
  and paired L24 was 1.6%.

## Interpretation

The experiment gives a narrow positive result and a clear bottleneck.

The positive result is that a fixed appended register bank can become a latent
program interface when Qwen is adapted with LoRA and supervised with executable
slot traces. The main run reached 88.3-94.5% exact execution through length 12
on standard prompts, 89.1-90.6% on paraphrase prompts through length 12, and
87.9-96.1% paired execution through length 12.

The bottleneck is long-chain reliability. At length 24 the compiler still
predicts most individual slots correctly, but exact program correctness falls to
21.9% on standard prompts, 3.1% on paraphrases, and 12.5% on paired prompts.
The paired state-consistency metric falls to 3.1%, so the long-chain register
states are not yet prompt-invariant.

Final-answer-only learning is not enough here. Both the direct answer head and
the answer-only register compiler remained near modulo-97 chance. Trace
supervision is the difference between learning a register program and failing to
discover the interface.

## Artifacts

- Small files: `experiments/qwen_register_token_latent_compiler/`
- Large checkpoints:
  `large_artifacts/qwen_register_token_latent_compiler/checkpoints/`

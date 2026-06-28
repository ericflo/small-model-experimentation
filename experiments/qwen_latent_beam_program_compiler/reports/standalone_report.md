# Qwen Latent Program Compiler

## Summary

This experiment tested whether a small posttraining attachment can make a Qwen
4B language model configure a fixed invisible runtime instead of solving a task
only through generated text. The task was modular arithmetic with known
operation traces. The model saw a natural-language prompt plus latent register
tokens, predicted a compact program into those registers, and a differentiable
runtime executed the predicted program.

The main result is positive but bounded. A single supervised compiler learned
to emit exact executable programs up to 12 steps with high reliability. It also
solved one 24-step paraphrase split in the mixed-template run. However,
multi-beam candidate search did not create useful oracle candidates, and
balanced paired-template training did not solve 24-step programs. The next
experiment should therefore focus on structural depth expansion or denser
long-depth supervision, not on adding more beams or paired consistency alone.

## Setup

- Base model: `Qwen/Qwen3-4B` loaded as a 4-bit causal LM with LoRA adapters.
- Runtime: exact modular arithmetic over modulus 97 with operations `ADD`,
  `SUB`, and `MUL`.
- Compiler output: initial value, per-step operation, and per-step argument.
- Primary supervision: exact trace loss, soft runtime answer loss, and state
  trajectory loss.
- Metrics:
  - `selected_accuracy`: final answer accuracy.
  - `selected_program_exact`: exact executable program recovery.
  - `selected_state_prefix_fraction`: fraction of the gold execution trace
    matched before the first error.
  - `oracle_accuracy`: whether any beam was correct. With one beam, oracle and
    selected accuracy are identical.

Large checkpoints are stored separately under
`large_artifacts/qwen_latent_beam_program_compiler/checkpoints/`.

## Results

### Beam Search Did Not Create Useful Programs

The prompt-bank four-beam run and compact latent four-beam run both stayed near
chance. The compact latent run ended with selected accuracy of 0-1.6% and oracle
accuracy of 1.6-4.7% across the evaluated splits. Because oracle accuracy was
also near chance, the failure was not mainly selector failure. The beams were
not producing correct candidate programs.

### One Supervised Compiler Worked

Removing beam search exposed the core mechanism. With one latent program bank
and max 8 program steps, the compiler learned exact executable traces:

| Split | Answer Accuracy | Exact Program | State Prefix |
| --- | ---: | ---: | ---: |
| standard length 4 | 91.4% | 91.4% | 94.9% |
| standard length 8 | 78.1% | 78.1% | 86.3% |
| paraphrase length 4 | 94.5% | 94.5% | 94.5% |
| paraphrase length 8 | 94.5% | 94.5% | 94.5% |

This is the most important positive result: the attachment was not merely
predicting final answers. Answer accuracy tracked exact program recovery.

### Max-24 Mixed Training Was Powerful but Brittle

The unpaired max-24 mixed-template run solved length 8 and 12, and it partially
solved length 24:

| Split | Answer Accuracy | Exact Program | State Prefix |
| --- | ---: | ---: | ---: |
| standard length 8 | 100.0% | 100.0% | 100.0% |
| standard length 12 | 87.5% | 87.5% | 97.8% |
| standard length 24 | 4.7% | 0.0% | 48.6% |
| paraphrase length 8 | 100.0% | 100.0% | 100.0% |
| paraphrase length 12 | 100.0% | 100.0% | 100.0% |
| paraphrase length 24 | 68.8% | 68.8% | 92.8% |

The asymmetry matters. It shows that the fixed-runtime compiler can learn
24-step programs, but the learned interface is still template- and
curriculum-sensitive.

### Paired Template Training Did Not Fix Full Depth

The paired run used standard/paraphrase pairs of the same latent program and a
longer 300-step long stage. It solved length 8 and 12 across all prompt families
but failed at length 24:

| Split | Answer Accuracy | Exact Program | State Prefix |
| --- | ---: | ---: | ---: |
| standard length 12 | 100.0% | 100.0% | 100.0% |
| standard length 24 | 0.0% | 0.0% | 84.6% |
| paraphrase length 12 | 100.0% | 100.0% | 100.0% |
| paraphrase length 24 | 3.1% | 1.6% | 84.4% |
| paired length 12 | 100.0% | 100.0% | 100.0% |
| paired length 24 | 0.8% | 0.0% | 84.6% |

The high length-24 state-prefix score with low exact-answer accuracy means the
compiler often executes much of the long trace correctly but still makes a late
program error. Paired consistency alone did not remove those late errors.

## Figures

- [Final accuracy by run and length](figures/final_accuracy_by_run_length.png)
- [Length-24 training curves](figures/length24_training_curves.png)
- [Program exactness vs answer accuracy](figures/program_exact_vs_answer_accuracy.png)
- [State-prefix heatmap](figures/state_prefix_heatmap.png)
- [Training loss curves](figures/training_loss_curves.png)

## Interpretation

The fixed-runtime idea is viable in a narrow but meaningful sense. A Qwen 4B
backbone plus a small LoRA-trained compiler can write an exact program into
latent registers, and a non-language runtime can execute that program. This is a
concrete test-time-compute substrate: the model configures a computation graph
instead of spelling out every intermediate state in text.

The current bottleneck is depth robustness. The model learns short and medium
programs cleanly, but full 24-step behavior is fragile. The beam experiments
showed that candidate search does not help until the compiler can reliably
create correct candidates. The paired experiment showed that template
consistency does not automatically produce full-depth reliability.

## Next Experiment

The highest-value next experiment is structural compiler expansion:

1. Train a max-8 compiler until exact program recovery is high.
2. Expand the same compiler to max 16 by copying learned slot/head structure and
   initializing new step slots near the learned distribution.
3. Continue training on length 8-16.
4. Expand to max 24 and continue on length 12-24.
5. Evaluate standard, paraphrase, and paired prompts at every expansion point.

This tests whether the failure is caused by asking a 24-slot compiler to learn
all depths from scratch. It also matches the engineering shape of the desired
posttraining tweak: grow a reusable latent execution scaffold rather than
sampling independent candidate programs.

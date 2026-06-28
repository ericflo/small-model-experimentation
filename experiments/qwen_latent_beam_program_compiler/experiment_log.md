# Experiment Log

## Objective

Train a small posttraining attachment on top of a Qwen 4B causal LM so the model
can configure an invisible fixed runtime instead of emitting a chain-of-thought
solution token by token. The immediate testbed is modular arithmetic with known
program traces, because it gives exact supervision for program compilation,
runtime execution, beam oracle coverage, and length generalization.

## Artifact Policy

Large files are kept outside the experiment directory in
`large_artifacts/qwen_latent_beam_program_compiler/checkpoints/`. The experiment
directory keeps source code, CSV logs, JSON summaries, figures, and reports.

## Runs

### `smoke_beam2`

- Purpose: verify the initial prompt-bank implementation with two visible
  candidate banks.
- Outcome: plumbing passed. This was not intended as a learning result.

### `pilot_beam4_s120`

- Purpose: test four actual prompt-side candidate banks on length-12 and
  length-24 modular programs.
- Setup: 4 beams, prompt-bank mode, 120 curriculum steps, max 24 program steps.
- Runtime note: sequence length was 2225 tokens for one example, so evaluation
  and training were expensive.
- Result: selected and oracle accuracy stayed near chance. Final oracle accuracy
  was 0-9.4% across splits, with no reliable selected-vs-oracle gap to recover.
- Decision: actual prompt-side beam banks are too token-expensive and do not
  create a useful candidate set in this form.

### `smoke_latent_beam4`

- Purpose: verify compact latent-beam mode, where the prompt contains one bank
  and learned beam embeddings create multiple candidate programs inside the
  compiler.
- Outcome: plumbing passed. Sequence length dropped to 193 tokens in the tiny
  smoke configuration.

### `pilot_latent_beam4_s160`

- Purpose: test whether compact latent beams can create useful candidate
  coverage without prompt-bank token cost.
- Setup: 4 latent beams, 160 curriculum steps, max 24 program steps, length-12
  and length-24 evaluation.
- Runtime note: full-task sequence length was 597 tokens and the run completed
  in 838 seconds on an RTX 6000 Ada.
- Result: final selected accuracy stayed at 0-1.6%, and oracle accuracy stayed
  at 1.6-4.7%. The executor assigned roughly chance probability to the correct
  answer throughout training (`best_soft_answer_mass` stayed near 1/97).
- Decision: latent beams create some output diversity but do not learn correct
  executable candidate programs under the current set-level objective.

## Current Diagnosis

The failure is not primarily selector selection. Oracle-over-beams is also near
chance, so there is no hidden high-quality beam for the selector to find.
The next run should test the necessary subproblem directly: can the Qwen-attached
compiler learn to emit one supervised executable program trace at all?

### `pilot_single_compiler_len8_s300`

- Purpose: isolate the necessary subproblem by removing beam search and training
  one latent program compiler with direct supervised trace/runtime losses.
- Setup: 1 latent beam, no paired training, max 8 program steps, 300 curriculum
  steps over length 1-4 then 1-8.
- Runtime note: sequence length was 262 tokens and the run completed in 499
  seconds on an RTX 6000 Ada.
- Result: strong positive signal. Final exact program accuracy matched final
  answer accuracy: 91.4% on standard length 4, 78.1% on standard length 8,
  94.5% on paraphrase length 4, and 94.5% on paraphrase length 8. Correct-answer
  probability rose from chance near 1/97 to 0.997 on the final train row.
- Decision: the Qwen-attached fixed-runtime interface can learn executable
  programs. The highest-value next run is a deterministic compiler curriculum to
  24 steps, with standard, paraphrase, and paired-template evaluation.

## Updated Diagnosis

Beam set training failed because useful candidates were not being created, not
because the fixed-runtime interface is unusable. A direct supervised compiler can
write exact executable programs. The main question is now whether that competence
survives longer programs and template variation at 24 steps.

### `pilot_single_compiler_len24_s400` (aborted diagnostic)

- Purpose: try the deterministic supervised compiler directly at max 24 steps
  with paired standard/paraphrase training.
- Setup: 1 latent beam, max 24 steps, paired training, eval at lengths 8, 12,
  and 24.
- Runtime note: sequence length was 580 tokens and the nine-split eval grid was
  too expensive for frequent in-training evaluation.
- Observed before stopping: at step 200, selected accuracy was still at chance
  across standard, paraphrase, and paired splits, and loss remained around 44.
- Decision: do not use this exact setup as the main run. Before adding more
  machinery, test the cleaner scaling variable: unpaired max-24 training with a
  larger batch and lighter eval.

### `pilot_single_compiler_len24_unpaired_s600`

- Purpose: test whether the deterministic compiler scales to 24 slots when
  trained without paired-template batches and with a larger batch.
- Setup: 1 latent beam, max 24 program steps, unpaired mixed-template training,
  600 curriculum steps, batch size 8, eval at lengths 8, 12, and 24.
- Runtime note: sequence length was 602 tokens, peak observed GPU memory was
  roughly 30 GB, and the completed run took 2294 seconds.
- Result: strong scaling to length 12 and partial length-24 success. Final exact
  program accuracy was 100% on standard length 8, 87.5% on standard length 12,
  100% on paraphrase length 8, and 100% on paraphrase length 12. Length 24 was
  asymmetric: standard length 24 reached only 4.7% answer accuracy with 0% exact
  program accuracy, while paraphrase length 24 reached 68.8% exact program
  accuracy.
- Decision: max-24 compilation is viable, but template robustness is not solved.
  Run a balanced paired-template version with a longer long stage before treating
  this as the main result.

### `main_single_compiler_len24_paired_s750`

- Purpose: test whether balanced paired standard/paraphrase training fixes the
  long-depth template asymmetry from the unpaired run.
- Setup: 1 latent beam, max 24 program steps, paired standard/paraphrase
  training, paired evaluation, 750 curriculum steps with a 300-step long stage.
- Runtime note: sequence length was 581 tokens, observed GPU memory was roughly
  19-28 GB during training/eval, and the completed run took 3358 seconds.
- Result: length 8 and 12 were solved across standard, paraphrase, and paired
  splits. Final selected/exact program accuracy was 100% for standard length 8,
  standard length 12, paraphrase length 8, paraphrase length 12, paired length 8,
  and paired length 12. Length 24 did not solve: final selected accuracy was 0%
  on standard length 24, 3.1% on paraphrase length 24, and 0.8% on paired length
  24.
- Decision: paired consistency is not enough for full-depth generalization. The
  current strongest positive result is deterministic fixed-runtime compilation
  through 12 steps and one unpaired 24-step paraphrase split. A better next
  experiment should use structural expansion/resume or denser long-depth
  supervision rather than more paired consistency alone.

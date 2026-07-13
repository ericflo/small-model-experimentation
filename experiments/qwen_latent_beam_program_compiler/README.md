# Qwen Latent Beam Program Compiler

**Status:** finished

This experiment tests whether a Qwen 4B backbone can write a compact latent
program into fixed register slots, execute that program with a differentiable
modular-arithmetic runtime, and improve by selecting among multiple latent
candidate programs.

Small metadata, logs, plots, and reports live in this directory. Large adapter
and head checkpoints are stored under:

`large_artifacts/qwen_latent_beam_program_compiler/checkpoints/`

## Layout

- `src/qwen_latent_beam_program_compiler_experiment.py` - training and
  evaluation script.
- `runs/` - per-run metrics and JSON summaries.
- `reports/` - final standalone write-up and generated figures.
- `checkpoint_manifest.csv` - checkpoint locations for the latest completed
  run written by the training script.
- `experiment_log.md` - running lab notebook for decisions and results.

## Main Measurements

- `selected_accuracy`: answer accuracy of the learned selector.
- `oracle_accuracy`: answer accuracy if any beam is allowed to be selected.
- `selected_oracle_gap_recovered`: how much of the oracle-over-beams advantage
  the selector recovers.
- `avg_distinct_answers`: whether beams collapse or explore distinct programs.
- `program_exact` and `state_prefix_fraction`: whether the compiled program
  matches the executable trace, not just the final answer.

## Current Result

The standalone report is in `reports/standalone_report.md`.

The short version: beam-set training did not create useful candidate programs,
but a single supervised compiler did learn exact executable programs. It solved
8- and 12-step modular programs reliably, reached 68.8% exact program accuracy
on one 24-step paraphrase split, and failed to make 24-step behavior robust under
paired-template training. The next experiment should use staged compiler
expansion/resume rather than more beams.

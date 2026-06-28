# Qwen Episodic Soft-Prompt Task Vectors Experiment Log

## Objective

Test whether a small continuous task vector learned from a task's training examples can improve a frozen model's held-out text-transformation consistency.

The core intervention is episodic soft-prompt optimization: for each task, freeze the model, optimize a short sequence of continuous prefix embeddings on the task's training examples, and then prepend that learned task vector during held-out inference.

The primary metric is strict full-task exact on held-out rows: a task counts as correct only if every held-out output is exactly correct.

The experiment is standalone. It stores run-local caches, learned-prefix diagnostics, analyses, charts, Markdown report, and HTML report under this directory. Large benchmark data is referenced through `/workspace/large_artifacts/qwen_episodic_soft_prompt_task_vectors`.

## Initial Plan

1. Create a fresh experiment directory and separate large-artifact root.
2. Load public text-transformation tasks.
3. Compare direct row-by-row and batched baselines against learned soft prompts.
4. Include controls:
   - frozen random soft prompt
   - shuffled-label learned soft prompt
   - zero/initialized soft prompt baseline
5. Track train loss, train exactness, held-out row exactness, held-out full-task exactness, parse success, and overfit signatures.
6. Run no-Qwen smoke, small real-Qwen pilot, then a main run.
7. Generate CSVs, charts, Markdown report, and HTML report.

## Run Notes

### 2026-06-27 09:08 UTC - Scaffold

- Created fresh experiment directory: `/workspace/experiments/qwen_episodic_soft_prompt_task_vectors`.
- Created separate large-artifact root: `/workspace/large_artifacts/qwen_episodic_soft_prompt_task_vectors`.
- Added standalone runner: `src/qwen_episodic_soft_prompt_task_vectors.py`.
- Added persistent log and later added `README.md`.

### 2026-06-27 09:09 UTC - No-Qwen Smoke

Command:

```bash
python -m py_compile /workspace/experiments/qwen_episodic_soft_prompt_task_vectors/src/qwen_episodic_soft_prompt_task_vectors.py
python /workspace/experiments/qwen_episodic_soft_prompt_task_vectors/src/qwen_episodic_soft_prompt_task_vectors.py \
  --run_name smoke_no_qwen \
  --task_limit 3 \
  --heldout_cap 3 \
  --steps 1 \
  --no_qwen
```

Result:

- Smoke completed on 3 tasks.
- CSVs, charts, Markdown report, and HTML report were generated.
- Metrics are not interpreted because `--no_qwen` leaves generations blank.

### 2026-06-27 09:10 UTC - Tiny Real-Qwen Gradient/Generation Pilot

Command:

```bash
python /workspace/experiments/qwen_episodic_soft_prompt_task_vectors/src/qwen_episodic_soft_prompt_task_vectors.py \
  --run_name pilot_qwen_2_step1 \
  --task_limit 2 \
  --heldout_cap 3 \
  --soft_tokens 4 \
  --steps 1 \
  --lr 0.05
```

Result:

- Real-model training and soft-prefix generation completed successfully.
- This validated that gradients through the frozen 4-bit model to input soft prompts work in this environment.
- One-step optimization was too aggressive on a numeric task: learned and shuffled soft prompts both degraded that row output.

### 2026-06-27 09:12 UTC - Lower-LR Pilot

Command:

```bash
python /workspace/experiments/qwen_episodic_soft_prompt_task_vectors/src/qwen_episodic_soft_prompt_task_vectors.py \
  --run_name pilot_qwen_4_lr001_s6 \
  --task_limit 4 \
  --heldout_cap 3 \
  --soft_tokens 8 \
  --steps 6 \
  --lr 0.01
```

Result:

- `direct_row`: 83.3% row exact, 75.0% full-task exact.
- `learned_soft_row`: 75.0% row exact, 75.0% full-task exact.
- `shuffled_soft_row`: 75.0% row exact, 75.0% full-task exact.
- Training loss sometimes overshot by the final step, so final-step prefix selection was not safe.

### 2026-06-27 09:13 UTC - Best-Loss Prefix Selection

Changed the optimizer to return the best-loss prefix checkpoint instead of the final optimization step.

Reran the four-task pilot:

```bash
python /workspace/experiments/qwen_episodic_soft_prompt_task_vectors/src/qwen_episodic_soft_prompt_task_vectors.py \
  --run_name pilot_qwen_4_best_lr001_s6 \
  --task_limit 4 \
  --heldout_cap 3 \
  --soft_tokens 8 \
  --steps 6 \
  --lr 0.01
```

Result:

- All row methods tied on full-task exact at 75.0%.
- `learned_soft_row` no longer damaged the numeric task relative to direct row-by-row.
- Chose `soft_tokens=8`, `steps=6`, `lr=0.01`, best-loss checkpoint selection for the main runs.

### 2026-06-27 09:15 UTC - Main Run, 20 Tasks

Command:

```bash
python /workspace/experiments/qwen_episodic_soft_prompt_task_vectors/src/qwen_episodic_soft_prompt_task_vectors.py \
  --run_name main_qwen_soft_prompt_20_s6_lr001 \
  --task_limit 20 \
  --heldout_cap 4 \
  --soft_tokens 8 \
  --steps 6 \
  --lr 0.01
```

Result:

- `direct_row`: 67.5% row exact, 50.0% full-task exact.
- `learned_soft_row`: 70.4% row exact, 50.0% full-task exact.
- `init_soft_row`: 68.8% row exact, 50.0% full-task exact.
- `shuffled_soft_row`: 62.5% row exact, 45.0% full-task exact.

Read:

- Learned soft prompts improved row exact slightly but did not solve more full tasks.
- The result was close enough to warrant a 40-task run.

### 2026-06-27 09:16 UTC - Main Run, 40 Tasks

Command:

```bash
python /workspace/experiments/qwen_episodic_soft_prompt_task_vectors/src/qwen_episodic_soft_prompt_task_vectors.py \
  --run_name main_qwen_soft_prompt_40_s6_lr001 \
  --task_limit 40 \
  --heldout_cap 4 \
  --soft_tokens 8 \
  --steps 6 \
  --lr 0.01
```

Result:

- 40 tasks.
- Runtime: 481.45 seconds.
- `direct_row`: 70.4% row exact, 52.5% full-task exact.
- `learned_soft_row`: 71.5% row exact, 52.5% full-task exact.
- `init_soft_row`: 71.7% row exact, 50.0% full-task exact.
- `shuffled_soft_row`: 63.5% row exact, 45.0% full-task exact.
- `learned_soft_batch`: 63.8% row exact, 42.5% full-task exact.
- `direct_batch`: 65.0% row exact, 40.0% full-task exact.

Deltas versus `direct_row`:

- `learned_soft_row`: +0.0 full-task points, +1.0 row-exact points; helped 1 task, hurt 1 task, tied 38 tasks.
- `init_soft_row`: -2.5 full-task points, +1.2 row-exact points.
- `shuffled_soft_row`: -7.5 full-task points, -6.9 row-exact points.
- `learned_soft_batch`: -10.0 full-task points.
- `direct_batch`: -12.5 full-task points.

Task-level movement:

- `learned_soft_row` fixed `DateTime.000076`.
- `learned_soft_row` broke `City.000011`.
- Net strict full-task exact was unchanged.

Final diagnosis:

- Neutral for strict full-task exact.
- Weakly positive at row level.
- The learned target matters: shuffled-label prefixes are worse.
- The untrained initialized prefix also changes row behavior, so not all row-level movement should be credited to task learning.
- Episodic soft prompts do not produce a deployable improvement under this configuration, but they are not inert: they can move individual rows and one full task in each direction.

### 2026-06-27 09:25 UTC - Report Hardening

- Added report-only regeneration path to avoid rerunning Qwen when editing the report.
- Regenerated the Markdown and HTML reports from the 40-task CSVs.
- Updated report interpretation to state the neutral full-task result and weak row-level gain explicitly.

Final artifacts:

- `README.md`
- `experiment_log.md`
- `src/qwen_episodic_soft_prompt_task_vectors.py`
- `runs/main_qwen_soft_prompt_40_s6_lr001/task_details.csv`
- `runs/main_qwen_soft_prompt_40_s6_lr001/row_details.csv`
- `runs/main_qwen_soft_prompt_40_s6_lr001/train_log.csv`
- `runs/main_qwen_soft_prompt_40_s6_lr001/summary.csv`
- `runs/main_qwen_soft_prompt_40_s6_lr001/method_deltas.csv`
- `analysis/*.csv`
- `analysis/figures/*.png`
- `reports/qwen_episodic_soft_prompt_task_vectors_report.md`
- `reports/qwen_episodic_soft_prompt_task_vectors_report.html`

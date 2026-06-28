# Qwen Verified Skill Memory RAG Experiment Log

## Objective

Test whether a frozen model becomes more task-consistent when it can retrieve analogous verified transformation skills from a train-only memory.

The primary metric is strict full-task exact on held-out rows: a task is counted correct only when every held-out output is exact.

The experiment is standalone. It stores run-local caches, retrieval records, summaries, charts, Markdown report, and HTML report under this directory. Large benchmark data is referenced through `/workspace/large_artifacts/qwen_verified_skill_memory_rag`.

## Initial Plan

1. Create a fresh experiment directory and separate large-artifact root.
2. Load public text-transformation tasks.
3. Split tasks into an evaluation set and a train-only skill-memory set.
4. Build verified skill cards from memory tasks.
5. Retrieve analogous skill cards for each evaluation task using only the task's training examples.
6. Compare:
   - direct row-by-row Qwen
   - direct batched Qwen
   - retrieved-skill batched Qwen
   - random-skill control
   - corrupted-skill control
7. Generate CSVs, charts, Markdown report, and HTML report.

## Run Notes

### 2026-06-27 08:45 UTC - Scaffold

- Created fresh experiment directory: `/workspace/experiments/qwen_verified_skill_memory_rag`.
- Created separate large-artifact root: `/workspace/large_artifacts/qwen_verified_skill_memory_rag`.
- Added standalone runner: `src/qwen_verified_skill_memory_rag.py`.
- Added persistent log and later added `README.md`.

### 2026-06-27 08:46 UTC - No-Qwen Smoke

Command:

```bash
python -m py_compile /workspace/experiments/qwen_verified_skill_memory_rag/src/qwen_verified_skill_memory_rag.py
python /workspace/experiments/qwen_verified_skill_memory_rag/src/qwen_verified_skill_memory_rag.py \
  --run_name smoke_no_qwen \
  --task_limit 4 \
  --heldout_cap 3 \
  --top_k 2 \
  --no_qwen
```

Result:

- Smoke completed on 4 tasks.
- CSVs, charts, Markdown report, and HTML report were generated.
- Retrieval diagnostics were structurally valid; top retrievals were same-family on the tiny smoke sample.
- Metrics are not interpreted because `--no_qwen` leaves uncached generations blank.

### 2026-06-27 08:47 UTC - Real-Qwen Pilot, Top-3 Retrieval

Command:

```bash
python /workspace/experiments/qwen_verified_skill_memory_rag/src/qwen_verified_skill_memory_rag.py \
  --run_name pilot_qwen_6 \
  --task_limit 6 \
  --heldout_cap 3 \
  --top_k 3
```

Result:

- 6 evaluation tasks.
- `direct_batch`: 83.3% row exact, 66.7% full-task exact.
- `direct_row`: 77.8% row exact, 66.7% full-task exact.
- `skill_rag`: 61.1% row exact, 50.0% full-task exact.
- `random_skill_rag`: 77.8% row exact, 66.7% full-task exact.
- `corrupt_skill_rag`: 55.6% row exact, 33.3% full-task exact.

Diagnosis:

- Top-3 retrieval over-steered the model on a date-format task: retrieved date examples pulled the model into a different format even though target examples were clear.
- Prompt needed to make target examples more authoritative.

### 2026-06-27 08:50 UTC - Prompt Iteration

Changed the memory prompt so the target examples appear first and are explicitly described as authoritative. Reference transformations are labeled optional and ignorable when they conflict with the target.

### 2026-06-27 08:51 UTC - Real-Qwen Pilot, Top-3 Retrieval With Stronger Target Authority

Command:

```bash
python /workspace/experiments/qwen_verified_skill_memory_rag/src/qwen_verified_skill_memory_rag.py \
  --run_name pilot_qwen_6_v2 \
  --task_limit 6 \
  --heldout_cap 3 \
  --top_k 3
```

Result:

- `direct_batch`: 83.3% row exact, 66.7% full-task exact.
- `direct_row`: 77.8% row exact, 66.7% full-task exact.
- `skill_rag`: 83.3% row exact, 50.0% full-task exact.
- `random_skill_rag`: 83.3% row exact, 66.7% full-task exact.
- `corrupt_skill_rag`: 77.8% row exact, 50.0% full-task exact.

Diagnosis:

- The stronger prompt reduced row-level damage but still failed to beat direct baselines.
- Top-3 retrieval still looked too intrusive.

### 2026-06-27 08:52 UTC - Real-Qwen Pilot, Top-1 Retrieval

Command:

```bash
python /workspace/experiments/qwen_verified_skill_memory_rag/src/qwen_verified_skill_memory_rag.py \
  --run_name pilot_qwen_6_top1 \
  --task_limit 6 \
  --heldout_cap 3 \
  --top_k 1
```

Result:

- `direct_batch`: 83.3% row exact, 66.7% full-task exact.
- `direct_row`: 77.8% row exact, 66.7% full-task exact.
- `skill_rag`: 88.9% row exact, 66.7% full-task exact.
- `random_skill_rag`: 83.3% row exact, 66.7% full-task exact.
- `corrupt_skill_rag`: 83.3% row exact, 66.7% full-task exact.

Diagnosis:

- Top-1 retrieval was cleaner than top-3.
- It improved row exact on the pilot but did not improve strict full-task exact, and controls still tied it on full-task exact.
- Chose top-1 for the main run because it was least damaging and still tested the intended mechanism.

### 2026-06-27 08:53 UTC - Main Run

Command:

```bash
python /workspace/experiments/qwen_verified_skill_memory_rag/src/qwen_verified_skill_memory_rag.py \
  --run_name main_qwen_skill_memory_40_top1 \
  --task_limit 40 \
  --heldout_cap 6 \
  --top_k 1
```

Result:

- 40 evaluation tasks.
- 269 train-only memory tasks.
- 364 unique cached generation records.
- No blank generations in the main cache.
- Top-1 retrieved skill was same-family 77.5% of the time, versus 20.0% for random retrieval.

Main metrics:

- `direct_row`: 70.8% row exact, 50.0% full-task exact.
- `skill_rag`: 65.2% row exact, 47.5% full-task exact.
- `corrupt_skill_rag`: 63.1% row exact, 45.0% full-task exact.
- `direct_batch`: 65.2% row exact, 42.5% full-task exact.
- `random_skill_rag`: 67.1% row exact, 42.5% full-task exact.

Deltas versus `direct_row`:

- `skill_rag`: -2.5 full-task points, -5.6 row-exact points; helped 0 tasks, hurt 1 task, tied 39 tasks.
- `corrupt_skill_rag`: -5.0 full-task points.
- `direct_batch`: -7.5 full-task points.
- `random_skill_rag`: -7.5 full-task points.

Final diagnosis:

- Negative for the tested verified skill-memory mechanism.
- The retriever itself was not random: it found same-family memory tasks at a high rate.
- The failure is in converting a retrieved analogous skill card into better target outputs. The card changes the prompt distribution but does not reliably improve task consistency.
- Direct row-by-row inference remains the best method in this experiment.

### 2026-06-27 08:59 UTC - Report Hardening

- Added `analysis/method_deltas.csv`.
- Added `analysis/figures/wins_losses_vs_direct.png`.
- Updated Markdown and HTML reports to state the negative conclusion explicitly.
- Regenerated reports from the cached main run with `--no_qwen`; no additional model calls were required.

Final artifacts:

- `README.md`
- `experiment_log.md`
- `src/qwen_verified_skill_memory_rag.py`
- `runs/main_qwen_skill_memory_40_top1/generations.csv`
- `runs/main_qwen_skill_memory_40_top1/task_details.csv`
- `runs/main_qwen_skill_memory_40_top1/row_details.csv`
- `runs/main_qwen_skill_memory_40_top1/retrieval_details.csv`
- `runs/main_qwen_skill_memory_40_top1/summary.csv`
- `runs/main_qwen_skill_memory_40_top1/retrieval_summary.csv`
- `runs/main_qwen_skill_memory_40_top1/method_deltas.csv`
- `analysis/*.csv`
- `analysis/figures/*.png`
- `reports/qwen_verified_skill_memory_rag_report.md`
- `reports/qwen_verified_skill_memory_rag_report.html`

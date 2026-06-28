# Experiment Log: Qwen3.5-4B Real Sample Verify Commit

Date: 2026-06-25

Experiment directory:
`/workspace/experiments/qwen35_4b_real_sample_verify_commit`

Large artifacts directory:
`/workspace/large_artifacts/qwen35_4b_real_sample_verify_commit`

## Objective

Run the sample -> verify -> commit loop on genuinely sampled Qwen3.5-4B code completions, not mutation-derived candidate pools. The goal is to measure real candidate-pool coverage and selector capture without assuming that a correct candidate is present or that incorrect candidates are easy to reject.

The primary measurement is:

`final pass@1 = candidate coverage ceiling x selector capture`

Where:

- `candidate coverage ceiling` means at least one sampled candidate passes hidden tests.
- `selector capture` means the selected candidate is hidden-correct when the pool contains a hidden-correct candidate.
- Hidden tests are used only for evaluation labels, verifier training labels, and oracle diagnostic arms. Selection inputs at eval time remain leak-free.

## Hypothesis

The candidate distribution itself can dominate the result. If genuine Qwen sampling creates realistic subtle-bug near-misses, selection may collapse even when coverage is high. If coverage is low, the wall is the generator rather than the verifier.

This run is designed to distinguish three outcomes:

1. High coverage and high selector capture: sample + verify + commit is deployable.
2. High coverage and poor selector capture: verifier/selection is the wall.
3. Low coverage: generation is the wall.

## Model And Data

Base model:
`/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`

Datasets:

- MBPP train subset: 40 records.
- MBPP eval subset: 20 records.
- HumanEval eval subset: 30 records.

Candidate generation:

- Direct samples per task: 8.
- Repair attempts per task: 1.
- Temperatures: `0.2,0.7,1.0`.
- Top-p: `0.95`.
- Max new tokens: `220`.
- Seed: `20260625`.
- HumanEval public tests used during candidate filtering: up to 3.
- No mutation-generated candidates.

Candidate pool manifest:

| Split | Records | Mean candidates | Mean parsed/safe | Mean visible-passers | Coverage | Visible coverage |
|---|---:|---:|---:|---:|---:|---:|
| MBPP train | 40 | 6.925 | 6.675 | 4.075 | 77.5% | 77.5% |
| MBPP eval | 20 | 7.800 | 7.500 | 3.200 | 60.0% | 60.0% |
| HumanEval eval | 30 | 5.467 | 5.333 | 4.433 | 96.7% | 96.7% |

## Implementation Notes

The package is intentionally split so the downloadable experiment directory does not contain large checkpoints or adapters:

- Code, reports, CSVs, logs, and JSONL data are under the experiment directory.
- LoRA adapters are under `large_artifacts/qwen35_4b_real_sample_verify_commit/models`.

Core scripts:

- `scripts/sample_candidates.py`: genuine Qwen direct sampling plus Qwen repair sampling.
- `scripts/build_verifier_examples.py`: verifier SFT data from parsed/safe public-failing negatives plus visible-passing positives/negatives.
- `scripts/train_action_sft.py`: shared LoRA trainer for semantic verifier and stop controller.
- `scripts/eval_commit.py`: fixed-budget commit policies.
- `scripts/build_stop_examples.py`: adaptive generation-budget STOP/MORE training examples.
- `scripts/tune_threshold.py`: threshold baseline tuning on MBPP train.
- `scripts/eval_adaptive_budget.py`: adaptive generation-budget policies.
- `scripts/make_report.py`: standalone markdown report, CSV summaries, and figures.
- `scripts/run_evaluation_suite.sh`: reruns fixed-budget, threshold, and adaptive evaluation after adapters exist.

## Debugging And Fixes

This run intentionally used the real model and real Python execution, so several harness issues surfaced and were fixed:

1. MBPP prompt construction initially used the wrong field name (`text` instead of normalized `task_text`). Fixed prompt builders to use normalized task text.
2. The Qwen model implementation rejected `generator=` in `generate`. Fixed sampling to set `torch.manual_seed` per completion instead.
3. Qwen initially emitted thinking text rather than code-only completions. Fixed prompts to use the tokenizer chat template with `enable_thinking=False` where supported.
4. The static safety checker rejected safe typing imports such as `from typing import List`. Fixed import checking to validate the module name, not each imported alias as a module.
5. HumanEval prompts with only a function signature and docstring were being accepted as valid because the entry function existed. Fixed validation to require a non-docstring function body.
6. Verifier training data initially had too few negatives because genuine visible-passers were often correct. Fixed `build_verifier_examples.py` to include parsed/safe public-failing candidates as negative examples by default, while keeping `--visible-only` available.
7. Report generation initially mixed MBPP train scoring files into fixed-budget summaries. Fixed `make_report.py` to exclude `_train_` evaluation files.

## Training

Verifier SFT:

```bash
python scripts/train_action_sft.py \
  --train data/train_verifier_examples.jsonl \
  --output-dir /workspace/large_artifacts/qwen35_4b_real_sample_verify_commit/models/verifier_sft_lora \
  --loss-out reports/verifier_sft_training_losses.json \
  --method sampled_semantic_verifier_sft \
  --max-steps 160 \
  --batch-size 2 \
  --grad-accum 2
```

Verifier training examples:

- Total: 267.
- Positive: 148.
- Negative: 119.
- Includes parsed/safe public-failing negatives.

Stop controller SFT:

```bash
python scripts/train_action_sft.py \
  --train data/train_stop_examples.jsonl \
  --output-dir /workspace/large_artifacts/qwen35_4b_real_sample_verify_commit/models/stop_sft_lora \
  --loss-out reports/stop_sft_training_losses.json \
  --method sampled_generation_budget_stop_sft \
  --max-steps 120 \
  --batch-size 2 \
  --grad-accum 2
```

Stop controller training examples:

- Total: 200.
- STOP: 153.
- MORE: 47.

The threshold baseline was tuned on MBPP train verifier scores. Best threshold: `-19.875`, with MBPP-train selected hidden-pass `77.5%` at `2.75` samples average.

## Fixed-Budget Results

Budgets are candidate-prefix budgets. `max` means all available generated candidates for that record.

| Dataset | Policy | Max-budget coverage | Max-budget selected hidden-pass | Coverage captured |
|---|---|---:|---:|---:|
| MBPP eval | oracle coverage | 60.0% | 60.0% | 100.0% |
| MBPP eval | first visible | 60.0% | 55.0% | 91.7% |
| MBPP eval | public-signature majority | 60.0% | 55.0% | 91.7% |
| MBPP eval | shortest visible | 60.0% | 60.0% | 100.0% |
| MBPP eval | base verifier | 60.0% | 60.0% | 100.0% |
| MBPP eval | SFT verifier | 60.0% | 60.0% | 100.0% |
| HumanEval eval | oracle coverage | 96.7% | 96.7% | 100.0% |
| HumanEval eval | first visible | 96.7% | 96.7% | 100.0% |
| HumanEval eval | public-signature majority | 96.7% | 96.7% | 100.0% |
| HumanEval eval | shortest visible | 96.7% | 96.7% | 100.0% |
| HumanEval eval | base verifier | 96.7% | 96.7% | 100.0% |
| HumanEval eval | SFT verifier | 96.7% | 96.7% | 100.0% |

Coverage curves:

| Dataset | Budget 1 | Budget 2 | Budget 4 | Budget 8 | Max |
|---|---:|---:|---:|---:|---:|
| MBPP eval | 35.0% | 35.0% | 50.0% | 60.0% | 60.0% |
| HumanEval eval | 90.0% | 90.0% | 93.3% | 96.7% | 96.7% |

## Adaptive Generation-Budget Results

The adaptive resource here is number of generated candidates inspected before committing, matching the real-code cost structure more closely than probe/test count.

| Dataset | Policy | Visible coverage | Selected hidden-pass | Coverage captured | Mean samples used |
|---|---|---:|---:|---:|---:|
| MBPP eval | oracle stop | 60.0% | 60.0% | 100.0% | 5.35 |
| MBPP eval | threshold on SFT verifier score | 55.0% | 55.0% | 100.0% | 4.20 |
| MBPP eval | SFT stop controller | 55.0% | 55.0% | 100.0% | 4.20 |
| HumanEval eval | oracle stop | 96.7% | 96.7% | 100.0% | 1.60 |
| HumanEval eval | threshold on SFT verifier score | 96.7% | 96.7% | 100.0% | 1.60 |
| HumanEval eval | SFT stop controller | 96.7% | 96.7% | 100.0% | 1.70 |

## Interpretation

This pilot uses genuine sampled pools and the outcome is clear: on these pools, selection is not the main bottleneck.

For HumanEval, Qwen sampling already creates a correct visible-passing candidate on 96.7% of the sampled tasks, and the first visible-passing candidate is already hidden-correct whenever coverage exists. That leaves almost no selection headroom for either the base verifier or the SFT verifier.

For MBPP, max-budget coverage is only 60.0%. The verifier can capture that ceiling, but it cannot create candidates that sampling missed. This points at generation/repair coverage as the current wall, not verifier ranking.

The adaptive generation-budget result is useful but modest. On HumanEval, the controller matches the 96.7% ceiling after inspecting only about 1.7 samples on average. On MBPP, the SFT stop controller and tuned threshold trade 5 points of accuracy for fewer inspected samples relative to oracle-stop/max-budget selection.

The important fork is therefore:

- Not enough evidence for a selector wall in this sampled-pool pilot.
- Strong evidence that MBPP needs better candidate generation or repair.
- HumanEval at this small scale is too forgiving to stress the verifier.

## Limitations

1. Scale is small because genuine Qwen3.5-4B sampling is expensive on the available hardware: 20 MBPP eval tasks and 30 HumanEval eval tasks.
2. HumanEval appears too easy under this sampling setup because public-visible passers are usually hidden-correct.
3. MBPP coverage is the decisive bottleneck, but this experiment does not yet improve the generator.
4. Verifier SFT data is small and partly trained with public-failing negatives; this is appropriate for a pilot but not a final verifier benchmark.
5. The code execution harness uses subprocess timeouts and static checks, not a full container sandbox.

## Decision

The next high-impact experiment should move upstream to coverage: verifier-guided self-improvement or repair generation on genuine sampled failures, still reported as coverage x selector capture. A larger verifier-only run is lower priority unless the candidate generator can produce harder realistic near-miss pools where coverage is high and first-visible selection fails.

The concrete next step I would run:

1. Generate larger genuine candidate pools on MBPP train/eval.
2. Use the verifier/executor to identify public-failing and hidden-failing candidates.
3. Train a repair generator or rejection-sampling distillation loop on verifier-selected candidates.
4. Re-run this exact coverage x selection report to see whether MBPP coverage rises above 60% without reducing selector capture.

## Reproduction Commands

```bash
python scripts/sample_candidates.py --mbpp-train 40 --mbpp-eval 20 --humaneval-eval 30 --samples-per-task 8 --repair-per-task 1 --max-new-tokens 220 --generation-batch-size 4 --temperatures 0.2,0.7,1.0 --top-p 0.95
python scripts/build_verifier_examples.py
python scripts/train_action_sft.py --train data/train_verifier_examples.jsonl --output-dir /workspace/large_artifacts/qwen35_4b_real_sample_verify_commit/models/verifier_sft_lora --loss-out reports/verifier_sft_training_losses.json --method sampled_semantic_verifier_sft --max-steps 160 --batch-size 2 --grad-accum 2
python scripts/eval_commit.py --records data/mbpp_train_records.jsonl --policy sft_verifier --name sft_verifier --out reports/eval/mbpp_train_sft_verifier.json --budgets 1,2,4,8,max --adapter-dir /workspace/large_artifacts/qwen35_4b_real_sample_verify_commit/models/verifier_sft_lora
python scripts/build_stop_examples.py --scores reports/eval/mbpp_train_sft_verifier.json --budgets 1,2,4,8,max
python scripts/train_action_sft.py --train data/train_stop_examples.jsonl --output-dir /workspace/large_artifacts/qwen35_4b_real_sample_verify_commit/models/stop_sft_lora --loss-out reports/stop_sft_training_losses.json --method sampled_generation_budget_stop_sft --max-steps 120 --batch-size 2 --grad-accum 2
bash scripts/run_evaluation_suite.sh
python scripts/make_report.py
```

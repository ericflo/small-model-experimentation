# Qwen3.5-4B Oracle-Distilled Semantic Verifier Log

## Objective

Train a deployable Qwen3.5-4B semantic verifier for Python candidate programs. Training labels come from hidden-test execution. Inference uses only public task information and candidate code.

## Initial Design

- Training substrate: MBPP.
- Transfer substrate: HumanEval.
- Action interface: `A = candidate will pass hidden tests`, `B = candidate will fail hidden tests`.
- Candidate pool: canonical solution, source mutations, and generic default-return variants.
- Train examples: visible-test-passing candidates only, balanced to keep hard negatives.
- Evaluation metric: selected hidden-pass rate and fraction of candidate-pool coverage captured.
- Large artifacts: LoRA adapter and tokenizer under `/workspace/large_artifacts/qwen35_4b_oracle_distilled_semantic_verifier`.

## Reproduction

```bash
python scripts/build_dataset.py --mbpp-train 90 --mbpp-valid 32 --humaneval-eval 31 --visible-tests 1 --candidate-count 18
python scripts/train_verifier_sft.py --max-steps 220 --batch-size 2 --grad-accum 2
python scripts/make_report.py
```

## Run Notes

- Dataset construction initially targeted 40 HumanEval eval records. With leak-free public tests and a required visible-passing hidden-failing hard negative, only 31 HumanEval tasks were usable. I kept the hard-negative and leak-free constraints and reduced the HumanEval eval target to 31.
- MBPP train and validation targets remained at 90 and 32 records.

## Dataset Manifest

- MBPP train: 90 records, 489 balanced verifier examples, 5.86 visible-passing candidates per task on average.
- MBPP validation: 32 records, 181 verifier examples, 5.66 visible-passing candidates per task on average.
- HumanEval eval: 31 records, 345 verifier examples, 11.13 visible-passing candidates per task on average.
- Candidate-pool coverage is 100% for all retained splits.
- HumanEval skips before reaching 31 usable records: 36 tasks had no public doctest-style tests, 33 had no visible-passing hidden-failing hard negative.

## Evaluation Results

| Dataset | Policy | Selected hidden-pass |
|---|---|---:|
| MBPP validation | First visible-pass | 43.8% |
| MBPP validation | Shortest visible-pass | 25.0% |
| MBPP validation | Random visible-pass | 40.6% |
| MBPP validation | Frozen Qwen verifier | 71.9% |
| MBPP validation | SFT Qwen verifier | 81.2% |
| MBPP validation | Oracle coverage | 100.0% |
| HumanEval | First visible-pass | 77.4% |
| HumanEval | Shortest visible-pass | 32.3% |
| HumanEval | Random visible-pass | 64.5% |
| HumanEval | Frozen Qwen verifier | 93.5% |
| HumanEval | SFT Qwen verifier | 90.3% |
| HumanEval | Oracle coverage | 100.0% |

## Interpretation

The SFT verifier learned a real in-domain selection signal: MBPP validation improved from 71.9% for the frozen Qwen verifier to 81.2% for the adapter. The HumanEval result is weaker. SFT stayed above simple baselines, but it underperformed the frozen Qwen verifier by 3.2 points. This suggests the small MBPP-only oracle-distillation run partly overfit the candidate mutation and task distribution instead of cleanly improving the general code-verification prior.

The next iteration should keep the oracle-labeled verifier idea, but train on a broader code-candidate distribution and include an explicit regularizer or distillation term that preserves the frozen model's strong HumanEval behavior.

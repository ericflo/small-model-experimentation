# Qwen3.5-4B Oracle-Distilled Semantic Verifier

## Objective

Train Qwen3.5-4B as a deployable verifier for Python candidate programs. The training oracle labels visible-test-passing candidates by hidden-test execution. At inference, the model sees only the task, public tests, public-test status, and candidate code; it ranks candidates by the probability that they pass hidden tests.

## Data

- Train records: 90 MBPP tasks.
- Validation records: 32 MBPP tasks.
- Transfer eval records: 31 HumanEval tasks.
- Visible tests per task: 1.
- Candidate implementations per task: up to 18.
- Train verifier examples: 489.

## Key Result

- HumanEval first-visible baseline: 77.4%.
- HumanEval base Qwen verifier: 93.5%.
- HumanEval candidate-pool coverage: 100.0%.
- HumanEval SFT verifier: 90.3%.
- MBPP validation first-visible baseline: 43.8%.
- MBPP validation base Qwen verifier: 71.9%.
- MBPP validation SFT verifier: 81.2%.

## Readout

The SFT verifier improves MBPP validation selection by +9.4 points over the frozen base verifier and +37.5 points over first visible-pass selection. On HumanEval it improves +12.9 points over first visible-pass selection, but trails the frozen base verifier by 3.2 points. This is a useful but not complete transfer result: oracle-distilled posttraining teaches an in-domain candidate verifier, while the frozen model is already a very strong out-of-domain verifier under this candidate-generation setup.

## Overall

| Dataset | Policy | Candidate-pool coverage | Selected hidden-pass | Coverage captured | Visible candidates | Hidden-pass candidates |
|---|---|---:|---:|---:|---:|---:|
| humaneval | First visible-pass | 100.0% | 77.4% | 77.4% | 11.13 | 8.16 |
| humaneval | Shortest visible-pass | 100.0% | 32.3% | 32.3% | 11.13 | 8.16 |
| humaneval | Random visible-pass | 100.0% | 64.5% | 64.5% | 11.13 | 8.16 |
| humaneval | Oracle coverage | 100.0% | 100.0% | 100.0% | 11.13 | 8.16 |
| humaneval | Base Qwen verifier | 100.0% | 93.5% | 93.5% | 11.13 | 8.16 |
| humaneval | SFT Qwen verifier | 100.0% | 90.3% | 90.3% | 11.13 | 8.16 |
| mbpp | First visible-pass | 100.0% | 43.8% | 43.8% | 5.66 | 3.12 |
| mbpp | Shortest visible-pass | 100.0% | 25.0% | 25.0% | 5.66 | 3.12 |
| mbpp | Random visible-pass | 100.0% | 40.6% | 40.6% | 5.66 | 3.12 |
| mbpp | Oracle coverage | 100.0% | 100.0% | 100.0% | 5.66 | 3.12 |
| mbpp | Base Qwen verifier | 100.0% | 71.9% | 71.9% | 5.66 | 3.12 |
| mbpp | SFT Qwen verifier | 100.0% | 81.2% | 81.2% | 5.66 | 3.12 |

## Interpretation

The primary question is whether oracle-labeled posttraining teaches a semantic candidate verifier that captures candidate-pool coverage under leak-free public evidence. The answer here is mixed. The adapter clearly improves in-domain MBPP selection, which means the hidden-test labels provide a learnable signal. The HumanEval result is weaker: SFT remains above simple selection baselines, but the frozen Qwen verifier is better on this eval set. The most likely reading is that this small MBPP-only SFT run partly overfits the mutation and task distribution instead of improving the model's general verifier prior. The next iteration should either train on a larger and more varied code corpus, or distill preferences from a stronger oracle while preserving the frozen model's general-code prior.

## Figures

- `reports/figures/selected_hidden_pass.png`
- `reports/figures/coverage_captured.png`
- `reports/figures/selection_vs_coverage.png`
- `reports/figures/verifier_sft_loss.png`

## Reproduction

```bash
python scripts/build_dataset.py --mbpp-train 90 --mbpp-valid 32 --humaneval-eval 31 --visible-tests 1 --candidate-count 18
python scripts/eval_verifier.py --records data/mbpp_valid_records.jsonl --policy first --name first_visible --out reports/eval/mbpp_first_visible.json
python scripts/eval_verifier.py --records data/mbpp_valid_records.jsonl --policy shortest --name shortest_visible --out reports/eval/mbpp_shortest_visible.json
python scripts/eval_verifier.py --records data/mbpp_valid_records.jsonl --policy random --name random_visible --out reports/eval/mbpp_random_visible.json
python scripts/eval_verifier.py --records data/mbpp_valid_records.jsonl --policy oracle --name oracle_coverage --out reports/eval/mbpp_oracle_coverage.json
python scripts/eval_verifier.py --records data/humaneval_eval_records.jsonl --policy first --name first_visible --out reports/eval/humaneval_first_visible.json
python scripts/eval_verifier.py --records data/humaneval_eval_records.jsonl --policy shortest --name shortest_visible --out reports/eval/humaneval_shortest_visible.json
python scripts/eval_verifier.py --records data/humaneval_eval_records.jsonl --policy random --name random_visible --out reports/eval/humaneval_random_visible.json
python scripts/eval_verifier.py --records data/humaneval_eval_records.jsonl --policy oracle --name oracle_coverage --out reports/eval/humaneval_oracle_coverage.json
python scripts/eval_verifier.py --records data/mbpp_valid_records.jsonl --policy base --name base_verifier --out reports/eval/mbpp_base_verifier.json
python scripts/eval_verifier.py --records data/humaneval_eval_records.jsonl --policy base --name base_verifier --out reports/eval/humaneval_base_verifier.json
python scripts/train_verifier_sft.py --max-steps 220 --batch-size 2 --grad-accum 2
python scripts/eval_verifier.py --records data/mbpp_valid_records.jsonl --policy adapter --name sft_verifier --out reports/eval/mbpp_sft_verifier.json --adapter-dir /workspace/large_artifacts/qwen35_4b_oracle_distilled_semantic_verifier/models/verifier_sft_lora
python scripts/eval_verifier.py --records data/humaneval_eval_records.jsonl --policy adapter --name sft_verifier --out reports/eval/humaneval_sft_verifier.json --adapter-dir /workspace/large_artifacts/qwen35_4b_oracle_distilled_semantic_verifier/models/verifier_sft_lora
python scripts/make_report.py
```

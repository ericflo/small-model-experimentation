# Qwen3.5-4B HumanEval Adaptive Evidence Budget Log

## Objective

Run a standalone external-validity pilot for adaptive evidence budgeting on real executable Python tasks. The model's role is limited to deciding STOP versus MORE. The verifier owns candidate execution, probe ranking, and deterministic commit selection.

## Design Decisions

- Dataset: `openai/openai_humaneval`.
- Visible evidence: public doctest examples parsed from the task prompt.
- Probe evidence: generated inputs are unlabeled at inference time. They are used only to cluster candidate implementations by output agreement.
- Hidden evaluation: official HumanEval `check(candidate)` is reserved for evaluation and for constructing supervised STOP/MORE labels.
- Candidate pool: up to 16 implementations per task from canonical-solution mutations plus generic fallback bodies. The canonical solution is included to measure whether selection can recover a correct candidate when coverage exists.
- Commit rule: choose the first candidate in the largest output-agreement cluster among candidates passing public visible examples.
- Probe rule: greedily choose the unused probe that minimizes expected remaining agreement-cluster size.
- Model action space: `A = STOP`, `B = MORE`.
- Large artifact placement: the QLoRA adapter and tokenizer are stored under `/workspace/large_artifacts/qwen35_4b_humaneval_adaptive_budget/models/budget_sft_lora`.

## Iteration Notes

1. Initial smoke build with 3 train and 2 eval tasks succeeded, validating the executor, timeout path, public doctest parser, state builder, and cheap policy evaluators.
2. A stricter full build with 2 public examples and 32 generated probes plus 32 generated hidden checks accepted too few tasks. The filter was too selective for HumanEval.
3. A second full build with 1 public example and 16 probes plus 16 generated hidden checks still accepted too few tasks.
4. The final pilot uses 1 public visible example, 8 generated unlabeled probes, 0 generated hidden checks, and official HumanEval checks as the hidden target. This yielded 24 train tasks and 12 eval tasks from the first 53 raw tasks scanned.
5. Audit found a public-doctest parser bug: extracting examples from the full function prompt sometimes included the closing triple quote in the expected output. The parser was corrected to extract the function docstring first and skip malformed doctest blocks.
6. The corrected dataset was rebuilt from scratch with the same pilot settings: 24 train tasks, 12 eval tasks, 8 unlabeled probes, and one public visible test.
7. Corrected non-model baselines showed 100.0% candidate-pool coverage on eval but only 16.7% hidden-correct selected accuracy across all fixed budgets. This diagnosed a selection/grounding failure before model training.
8. Base Qwen STOP/MORE evaluation reached the same 16.7% selected accuracy with 0.33 probes on average.
9. QLoRA SFT was retrained for 160 optimizer steps on 216 corrected train states. The loss frequently fell near zero but had intermittent spikes, consistent with the small heterogeneous pilot split.
10. SFT evaluation reached the same 16.7% selected accuracy with 3.83 probes on average.

## Result Summary

| Policy | Hidden-correct selected | Candidate-pool coverage | Avg probes |
|---|---:|---:|---:|
| Fixed budget 0 | 16.7% | 100.0% | 0.00 |
| Fixed budget 8 | 16.7% | 100.0% | 8.00 |
| Oracle stop | 16.7% | 100.0% | 6.67 |
| Base Qwen stop/more | 16.7% | 100.0% | 0.33 |
| SFT Qwen stop/more | 16.7% | 100.0% | 3.83 |

## Read

This pilot is a clean negative for STOP/MORE budget control under the leak-free agreement-only evidence model. The hidden-correct implementation is always in the eval candidate pool, but unlabeled probes do not identify which output-agreement cluster is correct. More probes split candidates but do not move deterministic selection onto the correct candidate, leaving no useful headroom for a stopping controller.

## Reproduction

```bash
python scripts/build_dataset.py --train-tasks 24 --eval-tasks 12 --visible-tests 1 --probe-tests 8 --hidden-tests 0 --candidate-count 16 --max-budget 8
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget0 --fixed-budget 0 --max-budget 8
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget3 --fixed-budget 3 --max-budget 8
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget6 --fixed-budget 6 --max-budget 8
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget8 --fixed-budget 8 --max-budget 8
python scripts/eval_budget_policy.py --policy threshold --name threshold_70 --threshold 70 --max-budget 8
python scripts/eval_budget_policy.py --policy threshold --name threshold_90 --threshold 90 --max-budget 8
python scripts/eval_budget_policy.py --policy oracle_stop --name oracle_stop --max-budget 8
python scripts/eval_budget_policy.py --policy base --name base_budget_policy --max-budget 8
python scripts/train_budget_sft.py --max-steps 160 --batch-size 2 --grad-accum 2
python scripts/eval_budget_policy.py --policy adapter --name sft_budget_policy --adapter-dir /workspace/large_artifacts/qwen35_4b_humaneval_adaptive_budget/models/budget_sft_lora --max-budget 8
python scripts/make_report.py
```

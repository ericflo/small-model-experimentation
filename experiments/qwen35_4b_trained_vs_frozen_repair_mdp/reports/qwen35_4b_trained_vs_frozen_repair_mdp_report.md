# Qwen3.5-4B Trained vs Frozen Repair MDP Report

Date: 2026-06-25

## Summary

This experiment tested whether a trained repair policy can expand held-out coding coverage beyond frozen Qwen self-repair, under a fair comparison against spending the same estimated model-forward-token budget on more direct samples.

The result is negative for trained repair. On 150 held-out MBPP tasks, direct sampling covered 62.0% of tasks, leaving 57 zero-coverage tasks. Frozen repair recovered 3 of those 57 tasks. The SFT repair adapter recovered only 2 of 57 and had a higher false-repair rate. A token-matched sample-more baseline recovered 5 of 57, beating both repair arms at essentially the same estimated model-forward-token cost.

The practical read is: in this setup, the best use of extra model budget was more diverse direct generation, not trained repair. Frozen repair produced a small useful lift, but trained repair did not improve it.

## Main Held-Out Results

| Arm | N | Coverage | Zero-to-one | Zero-to-one rate | False repair rate | Candidates/task | Distinct behavior | Forward tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direct x4 | 150 | 62.0% | 0 / 57 | 0.0% | - | 3.45 | 0.770 | 138,902 |
| Frozen repair | 150 | 64.0% | 3 / 57 | 5.3% | 25.0% | 4.01 | 0.754 | 79,614 |
| SFT repair | 150 | 63.3% | 2 / 57 | 3.5% | 29.2% | 3.98 | 0.766 | 79,325 |
| Sample more | 150 | 65.3% | 5 / 57 | 8.8% | - | 5.18 | 0.708 | 79,861 |

## Zero-To-One Tasks

| Arm | Zero-to-one task IDs |
| --- | --- |
| Direct x4 | - |
| Frozen repair | 112, 137, 147 |
| SFT repair | 112, 147 |
| Sample more | 36, 42, 67, 129, 148 |

## Commit / Selection Summaries

The following table uses the final budget available in each candidate pool. `Oracle coverage` is a ceiling: it selects a hidden-correct candidate if one exists in the pool. Other policies use only public/visible candidate behavior.

| Arm | Commit policy | Budget | Coverage ceiling | Selected hidden-pass | Coverage captured |
| --- | --- | --- | --- | --- | --- |
| Direct x4 | First visible-pass | 8 | 62.0% | 60.0% | 96.8% |
| Direct x4 | Public-signature majority | 8 | 62.0% | 60.0% | 96.8% |
| Direct x4 | Shortest visible-pass | 8 | 62.0% | 61.3% | 98.9% |
| Direct x4 | Oracle coverage | 8 | 62.0% | 62.0% | 100.0% |
| Frozen repair | First visible-pass | 8 | 64.0% | 62.0% | 96.9% |
| Frozen repair | Public-signature majority | 8 | 64.0% | 62.0% | 96.9% |
| Frozen repair | Shortest visible-pass | 8 | 64.0% | 63.3% | 99.0% |
| Frozen repair | Oracle coverage | 8 | 64.0% | 64.0% | 100.0% |
| SFT repair | First visible-pass | 8 | 63.3% | 61.3% | 96.8% |
| SFT repair | Public-signature majority | 8 | 63.3% | 61.3% | 96.8% |
| SFT repair | Shortest visible-pass | 8 | 63.3% | 62.7% | 98.9% |
| SFT repair | Oracle coverage | 8 | 63.3% | 63.3% | 100.0% |
| Sample more | First visible-pass | 8 | 65.3% | 62.0% | 94.9% |
| Sample more | Public-signature majority | 8 | 65.3% | 62.0% | 94.9% |
| Sample more | Shortest visible-pass | 8 | 65.3% | 64.7% | 99.0% |
| Sample more | Oracle coverage | 8 | 65.3% | 65.3% | 100.0% |

## Training Details

- SFT training examples: 17.
- Max steps: 80.
- Batch size / grad accumulation: 1 / 4.
- Learning rate: 0.0001.
- Final logged SFT loss: 0.01351678092032671.
- DPO was skipped because the SFT repair arm failed the held-out gate.

## Figures

- [Coverage by arm](figures/coverage_by_arm.png)
- [Zero-to-one by arm](figures/zero_to_one_by_arm.png)
- [False repair rate](figures/false_repair_by_arm.png)
- [Diversity by arm](figures/diversity_by_arm.png)
- [Tokens vs zero-to-one](figures/tokens_vs_zero_to_one.png)
- [Repair SFT loss](figures/repair_sft_loss.png)

## Interpretation

The headline test was trained repair versus frozen repair on tasks with no hidden-correct direct sample. Trained repair did not pass that test: it recovered fewer zero-base tasks than frozen repair and produced a worse visible-pass-but-hidden-fail profile.

The sample-more baseline is the decisive comparator. It spent approximately the same model-forward-token budget as frozen repair and recovered more zero-base tasks. That means the repair loop did not justify its extra prompt structure or training in this run.

The false-repair rates matter. Frozen repair had 28 visible-passing repair candidates, 7 of which failed hidden tests. SFT repair had 24 visible-passing repair candidates, also with 7 hidden failures. Repair can create plausible candidates that satisfy public evidence but do not generalize, so aggregate visible pass rates would overstate its value.

## Limitations

- This is one held-out MBPP run, not a multi-seed estimate.
- The SFT adapter trained on only 17 mined repair examples, so the trained-arm negative should be read as a result for this small verified-repair recipe, not as a proof that repair training cannot work.
- Repair was conservative: it repaired visible-failing parsed candidates and did not repair candidates that already passed visible tests but failed hidden tests.
- No transfer benchmark was run in this package; the held-out MBPP comparison is the primary readout.
- Hidden tests were used for evaluation and train-side label mining, but not included in repair prompts.

## Conclusion

The experiment does not support trained repair as the next deployable posttraining lever. The best observed intervention was to preserve generation diversity and spend the matched budget on more direct samples. A stronger future repair experiment would need either a much larger verified repair set, a process objective that reduces false repairs, or a repair policy aimed at visible-pass hidden-fail near misses rather than only visible failures.

# Qwen3.5-4B HumanEval Code Confidence Experiment Log

## 2026-07-08 scaffold split

Created as a standalone experiment after the HumanEval replication had been
incorrectly added inside `qwen35_4b_code_confidence`. The harness, HumanEval run
artifacts, and figures were copied into this experiment so C46's MBPP experiment
can remain self-contained and this replication has its own README, report, log,
manifest, site brief, and chart.

## 2026-07-08 smoke

`python scripts/run.py --dataset humaneval --visible-tests 0 --n 4 --k 2
--out-name humaneval_smoke --judge-batch-size 1` completed generation,
execution, mean-logprob, and P(True) population on a small HumanEval subset.

## 2026-07-08 public-probe diagnostic

Command:

```bash
python scripts/run.py --dataset humaneval --visible-tests 1 --n 164 --k 8
```

Only 68 HumanEval tasks had one parseable doctest example usable as a public
probe. The subset is ceiling-limited: random 0.904, mean-logprob 0.926,
public-output majority 0.926, P(True) 0.941, visible-test execution 0.941,
oracle 0.971. P(True) beats random but not public-output majority or
mean-logprob significantly.

## 2026-07-08 all-task no-public-probe run

First all-task attempt reached generation, hidden-test scoring, and mean-logprob
but OOMed during the P(True) judge at batch size 16. The harness now persists
`runs/humaneval_code_conf_novis_logprob.json` before judging and exposes
`--judge-batch-size`.

Successful command:

```bash
python scripts/run.py --dataset humaneval --visible-tests 0 --n 164 --k 8 \
  --out-name humaneval_code_conf_novis \
  --title "HumanEval all tasks, no public probes" \
  --judge-batch-size 1
```

Result: random 0.766, mean-logprob 0.787, P(True) 0.835, oracle pass@8 0.872.
Within-problem AUROC on 51 mixed problems: length 0.573, mean-logprob 0.672,
P(True) 0.779. P(True) beats mean-logprob by +0.049 (p=0.011) and beats random
by +0.069 (p<0.001).

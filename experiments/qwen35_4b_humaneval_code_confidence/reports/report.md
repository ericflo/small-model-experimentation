# Qwen3.5-4B: HumanEval Code Confidence Replication Report

## Summary

The C46 code-confidence result replicates on HumanEval when evaluated as a
standalone, verifier-free experiment. On all 164 HumanEval tasks with no public
probes, P(True) selection reaches 0.835 at k=8, beating sequence mean-logprob
0.787 and random selection 0.766. Oracle pass@8 is 0.872, so P(True) recovers
most of the available selection headroom without executing tests.

The replication preserves the important inversion from MBPP: the useful
program-level confidence signal is not average completion likelihood. It is a
single-token judgment readout, P(A = correct), under a strict A/B self-judge
prompt.

## Research Program Fit

`benchmark_generalization`: tests whether the MBPP real-code confidence result
survives a second benchmark.

`evidence_conditioned_selection`: tests a deployable selector for the no-verifier
regime, where public tests or execution signals are unavailable.

## Method

- Dataset: all 164 HumanEval tasks.
- Main condition: `--visible-tests 0`, so public-output majority and
  visible-test execution are intentionally unavailable.
- Candidates: one greedy completion plus 8 no-think samples per task.
- Generation: temperature 0.7, top_p 0.8, top_k 20, 420-token answer budget.
- Ground truth: hidden tests score `full_pass` only after selection.
- Signals: `mean_logprob`, `p_true`, and `code_len`.
- Analysis: within-problem AUROC on mixed problems only; paired bootstrap over
  tasks for selection deltas.

## Results

### All HumanEval Tasks, No Public Probes

| selector | selection accuracy at k=8 |
|---|---:|
| random pick expectation | 0.766 |
| mean-logprob | 0.787 |
| **P(True) self-judge token** | **0.835** |
| oracle pass@8 | 0.872 |

Paired bootstrap:

- P(True) > mean-logprob: +0.049, CI 0.012 to 0.091, p=0.011.
- P(True) > random: +0.069, CI 0.040 to 0.100, p<0.001.

Within-problem AUROC on 51 mixed problems:

| signal | AUROC |
|---|---:|
| length baseline | 0.573 |
| mean-logprob | 0.672 |
| **P(True)** | **0.779** |

P(True) beats the length baseline by +0.351 (CI 0.229 to 0.475). Greedy
solvability AUROC is 0.862 for P(True), 0.734 for mean-logprob, and 0.612 for
length.

![HumanEval no-public-probe figure](../analysis/humaneval_code_conf_novis.png)

### Public-Probe Diagnostic

The public-probe run uses the 68 HumanEval tasks with one parseable doctest
example. This is not the main endpoint because the subset is much easier:

| selector | selection accuracy at k=8 |
|---|---:|
| random pick expectation | 0.904 |
| mean-logprob | 0.926 |
| public-output majority | 0.926 |
| **P(True)** | **0.941** |
| visible-test execution | 0.941 |
| oracle pass@8 | 0.971 |

P(True) beats random (+0.037, p=0.020), but does not significantly beat
mean-logprob or public-output majority on this ceiling-limited subset.

## Controls

- **No public-probe leakage:** the main run has zero public cases, so
  self-consistency over public behavior and visible execution are undefined
  rather than weak.
- **Verbosity control:** P(True) is compared against code length within each
  mixed problem; its +0.351 AUROC advantage rules out a pure length surface.
- **Difficulty control:** AUROC is reported within mixed problems only.
- **Selection uncertainty:** paired bootstrap over tasks is reported for the
  P(True) deltas.
- **Compute/memory control:** the long HumanEval judge prompts require
  `--judge-batch-size 1` on the 24 GB RTX 4090 used here; logprob artifacts are
  persisted before the judge phase so an OOM does not discard generation work.

## Oracle Versus Deployable Evidence

Deployable, no hidden labels: mean-logprob, P(True), and code length. Oracle-only:
`full_pass` and pass@8. The public-probe diagnostic adds deployable visible-test
execution and public-output majority, but only on the 68-task subset with a
parseable doctest example.

## Interpretation

HumanEval supports the refined C46 law: calibrated uncertainty for long-form code
is best read from a concentrated judgment-token logit. Sequence mean-logprob
still carries signal, but it is weaker because correctness evidence is diluted
across many tokens. This makes P(True) the current default confidence signal for
verifier-free selection and abstention on code.

The result does not imply confidence should replace execution. Where public tests
exist, visible execution remains the stronger deployable selector. The scope is
the no-verifier regime: when execution evidence is absent or too expensive,
P(True) beats sample-more baselines that only count or average candidates.

## Next Experiments

1. HumanEval/MBPP select + abstain + route policy versus matched-compute
   sample-more.
2. Thinking-judge P(True): test whether deliberative judging improves enough to
   justify its token cost.
3. Step-resolved P(True) repair for code, connecting this line to C42.

## Artifact Manifest

`reports/artifact_manifest.yaml` records that all artifacts are checked in. The
full reproduction command is `python scripts/run.py`.

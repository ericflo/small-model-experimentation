# Qwen3.5-4B: Does the Confidence Toolkit Survive on Real Code? Report

## Summary

Yes — with an inversion, and now on two real-code substrates. On 244 MBPP problems (greedy + 8 samples each, 2,196 candidates), the model's own logits discriminate correct from incorrect programs well above a length/verbosity surface baseline, and verification-free confidence-selection beats self-consistency (public-output majority), generalizing C40/C41 off the toy substrate. On all 164 HumanEval tasks with no public probes, the same P(True) signal beats mean-logprob and random selection in the verifier-free regime. The WINNING signal is not the toy's implicit mean-logprob — it is the single-token P(True) judge readout (P of the "A = correct" token under a self-judge prompt, no-think, one forward pass per candidate). Sequence mean-logprob is diluted over hundreds of tokens and does not significantly beat public-output majority on MBPP. Abstention transfers: greedy-solvability AUROC is 0.837 on MBPP and 0.862 on HumanEval vs 0.83 on the toy.

## Research Program Fit

`benchmark_generalization`: the owed generalization test written into C41's caveats ("real code needs a PROGRAM-level confidence — untested, owed"). Verification-free selection is the mission's purest lever: no training, no tools, no external verifier — beat sample-more by reading signal already in the fixed weights.

## Method

- MBPP sanitized test (offline cache), 244 records with executable hidden asserts; 1 test visible in the prompt.
- HumanEval: (a) all 164 tasks with `--visible-tests 0` for the verifier-free replication; (b) 68-task subset with one parseable doctest public probe for public-output majority and visible-execution diagnostics.
- Per problem: greedy + k=8 no-think samples (T=0.7, top_p=0.8, top_k=20), answer budget 420 tokens.
- Per candidate: `full_pass` ground truth by hidden-assert execution; `visible_all_pass` (public test only); `behavior_signature` (outputs on probe inputs — execution-clustering); `mean_logprob` (teacher-forced mean token logprob of the completion, bf16 logits, chunked float32 log-softmax); `p_true` (P(A) after "Answer: " under a strict-reviewer A/B judge prompt, no-think — the C10 readout); `code_len`.
- Analysis (review-hardened): within-problem AUROC on mixed problems only (pooled AUROC is difficulty-inflated); length baseline both signs; cluster bootstrap over problems; paired bootstrap for every selection delta; selection compared against random, self-consistency (behavior-cluster majority), visible-test execution, and oracle pass@k.

## Results

### MBPP

| signal | within-problem AUROC (78 mixed) | greedy problem AUROC | selection at k=8 |
|---|---|---|---|
| random / length | 0.548 (length) | 0.688 (length) | 0.696 |
| mean-logprob (implicit) | 0.693 (CI 0.631–0.751) | 0.760 | 0.730 |
| **P(True) no-think** | **0.738 (CI 0.665–0.808)** | **0.837** | **0.762** |
| self-consistency (public-output majority) | — | — | 0.721 |
| visible-test execution | — | — | 0.816 |
| oracle pass@k | — | — | 0.844 |

Significance (paired bootstrap over 244 problems): P(True) > self-consistency +0.041 (CI 0.004–0.078, p=0.014); P(True) > mean-logprob +0.033 (p=0.034); mean-logprob vs self-consistency n.s. (p=0.37). Within-problem paired diff vs length: logprob +0.242 (CI 0.145–0.338), P(True) +0.287 (CI 0.183–0.386).

Abstention on greedy ranked by P(True): coverage 0.33 → accuracy 0.94; coverage 0.20 → 0.96; unfiltered 0.701. Duplicate rate among samples (public outputs) 0.775. Note: self-consistency clusters on PUBLIC outputs only (visible-test behavior) so the baseline is strictly deployable — hidden-test outputs never inform any selector.

### HumanEval

The all-task HumanEval run uses no public probes, so self-consistency over public outputs and visible-test execution are intentionally unavailable. This is the clean verifier-free replication:

| signal | within-problem AUROC (51 mixed) | greedy problem AUROC | selection at k=8 |
|---|---|---|---|
| random / length | 0.573 (length) | 0.612 (length) | 0.766 |
| mean-logprob (implicit) | 0.672 (CI 0.568-0.767) | 0.734 | 0.787 |
| **P(True) no-think** | **0.779 (CI 0.695-0.857)** | **0.862** | **0.835** |
| oracle pass@k | — | — | 0.872 |

Significance (paired bootstrap over 164 problems): P(True) > mean-logprob +0.049 (CI 0.012-0.091, p=0.011); P(True) > random +0.069 (CI 0.040-0.100, p<0.001). Within-problem paired diff vs length: logprob +0.245 (CI 0.087-0.394), P(True) +0.351 (CI 0.229-0.475).

The HumanEval public-probe subset is an apples-to-apples diagnostic but ceiling-limited: 68 tasks with one parseable doctest example, random 0.904, oracle 0.971. P(True) reaches 0.941, visible-test execution 0.941, mean-logprob 0.926, public-output majority 0.926. P(True) beats random (+0.037, p=0.020) but does not significantly beat public-output majority or mean-logprob on this easier subset.

## Controls

- **Length/verbosity confound:** the central threat on code (verbose solutions could inflate mean-logprob or judge scores). Within-problem AUROC vs length in both directions: confidence wins by +0.24/+0.29 with CIs far from zero. Confidence is not verbosity.
- **Difficulty confound:** headline AUROC is within-problem (mixed problems only); pooled numbers reported separately.
- **Selection deltas:** all paired over problems, bootstrap CIs; the honest negative (mean-logprob n.s. vs self-consistency) is reported alongside the positive.

## Oracle Versus Deployable Evidence

Deployable (no hidden labels): mean-logprob, P(True), behavior-signature majority, visible-test execution (public test only). Oracle-only (scoring): `full_pass`, pass@k. The abstention curve uses only P(True) ranks; its y-axis is oracle-scored.

## Interpretation

The C40 law refines rather than breaks: calibrated uncertainty lives in CONCENTRATED SINGLE-TOKEN logit readouts — the answer digit's P on the toy, the judgment token's P on code. Sequence-averaged logprob dilutes it on both MBPP and HumanEval. C40's "read logits, not self-report" survives — P(True) IS a logit read, not a sampled verbalization — but the toy's implicit-beats-explicit hierarchy inverts on long-form outputs where no single answer token exists. C41's recipe updates for code: sample k, self-judge each candidate (one no-think forward pass; batchable but memory-sensitive on long prompts), pick argmax P(True), abstain on low max; execution still king when any test exists (MBPP visible-test execution 0.816 vs P(True) 0.762) — confidence is for the verifier-free regime.

## Next Experiments

1. Thinking-judge P(True) (judge_think exists in gen_lib; does deliberation improve the readout enough to pay for its tokens?).
2. Select + abstain + route policy vs matched-compute sample-more (the C41 "owed" compute-optimal curve).
3. Step-resolved P(True) on code for targeted repair (C42 bridge): resample only low-confidence regions.
4. Next training slot: compositional-grammar induction, with combinations / productions / depth as separate endpoints, execute-ceiling and token-budget gates, and branch-coverage/value-fill checks in episode generation.

## Artifact Manifest

`reports/artifact_manifest.yaml` — no external artifacts; everything regenerates from `scripts/run.py` (HF cache/network required for MBPP and HumanEval). The all-task HumanEval P(True) judge should use `--judge-batch-size 1` on this 24 GB GPU.

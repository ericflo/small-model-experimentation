# Adversarial design review — self-vetted (workflow agent produced no verdict)

The review-workflow agent returned no verdict; the design was self-vetted against the failure modes that burned earlier confidence claims (C40's review history), and each concern was built into the analysis as a mandatory control.

## Confounds

1. **Verbosity/length**: on real code, verbose commented solutions could inflate mean-logprob or judge scores. Control: code-length surface baseline (both signs) in every AUROC comparison, paired per-problem. Outcome: confidence beats length by +0.24/+0.29 (CIs exclude 0).
2. **Problem difficulty**: pooled AUROC mixes "which problems are hard" with "which samples are wrong". Control: headline AUROC is WITHIN-problem, on mixed problems only (78/244); pooled reported separately and labeled.
3. **Small paired deltas**: selection differences of a few points on 244 problems can be noise. Control: paired bootstrap over problems for every selection delta; the honest negative (mean-logprob n.s. vs self-consistency, p=0.28) is reported in the verdict.
4. **Selection floor/ceiling**: random-pick expectation (0.696) and oracle pass@k (0.844) bracket every method; headroom capture is stated against that 0.148 gap, not in absolute points.

## Must-fix

- ~~mean_logprobs OOM (full-vocab float32 log-softmax)~~ — fixed: bf16 logits, chunked float32 log-softmax, batch 4.
- ~~Abstention ranked only by mean-logprob~~ — fixed: P(True)-ranked curve added (it is the better signal, AUROC 0.837 vs 0.760).

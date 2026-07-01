# Qwen3.5-4B Generator-Verifier Gap Experiment Log

## Scaffold

Attacks the C2 selection bottleneck at its root: is the wall a verification-CAPABILITY limit or a
selection-PLUMBING limit? Measures the frozen 4B's intrinsic black-box verification skill vs its
generation skill on the same MBPP items, at no-think and thinking-on. Also inverts C9 (thinking helps
generation content; does it help verification asymmetrically?).

## Method

- k=8 no-think candidates per task (execution-labeled). Verification judge: task spec + candidate code ->
  read A/B logit -> P(correct); A=32/B=33 single tokens. No-think = one forward; thinking = generate up to
  1024 thinking tokens, force </think>, then read the A/B logit at "Answer: ".
- Metrics: generation pass@1/pass@k; verification balanced-accuracy + AUROC; verifier-selected best-of-k
  vs pass@1 vs oracle (gap closed); thinking asymmetry. Controls: foreign-solution judge, say-A rate.

## Smoke

4 tasks x k=4: pipeline validated. P(A) ~0.65-0.88 on own (correct) candidates, ~0.01 on FOREIGN
candidates (verifier correctly rejects a different task's solution) -- the judge reads the task. Thinking
raised P(A) on correct candidates (0.7 -> 0.9+); need incorrect candidates (full run) to see discrimination.

## Results (see reports/report.md)

Generation pass@1 0.771, oracle pass@8 0.890. Verification balanced-acc/AUROC: no_think 0.627/0.773
(say-A 0.91, heavy yes-bias), think 0.827/0.926 (say-A 0.83). Verifier-selected best-of-8: no_think 0.800
(+24% of the pass@1->oracle gap), think 0.860 (+75%). Foreign reject 1.00 both. Findings: checking is
easier than doing but only WITH thinking; C2 is plumbing not capability (deployable thinking-verifier closes
75% of the oracle gap, no training/execution); thinking helps verification (+0.20 bal-acc) >= generation
(C9 inversion). Claim C10.

Note: judge_think initially generated thinking for the whole 800-item list in one _gen call (fine for the
16-item smoke, would OOM at 800) -- fixed with chunked OOM-resilient processing before the full run.

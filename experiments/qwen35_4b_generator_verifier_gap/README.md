# Qwen3.5-4B Generator-Verifier Gap

**Status:** finished

## Research Program

- Program: `evidence_conditioned_selection` (bridges to `test_time_reasoning_budget` via the thinking axis).
- Program question: is the C2 selection wall (coverage ≫ deployable selection) a **capability** limit
  (the model can't tell right from wrong any better than it can produce right) or a **plumbing** limit
  (verification is good, but current selectors waste it)?
- Prior anchors: claim C2 (`qwen35_4b_retrieval_adapt_verify_scale`, `qwen35_4b_real_sample_verify_commit`);
  the whole selection program builds trained selectors and reports SELECTED accuracy but never isolates the
  upstream black-box primitive of self-verification. Claim C9 (thinking helps generation content).

## Question

Is checking easier than doing for a frozen 4B? Measure its intrinsic **verification** skill (given a
candidate solution, judge correct/incorrect as a black box — no execution, no hidden tests) against its
**generation** skill (pass@1) on the same MBPP items, at no-think and thinking-on. And can the model's
own verifier close the pass@1 → oracle-pass@k gap — i.e. is C2 fixable? Does thinking help *verification*
asymmetrically more than it helped *generation* (checking is often the easier reasoning problem)?

## Hypothesis

If verification balanced-accuracy ≫ generation pass@1 and verifier-selected accuracy approaches oracle
pass@k, the selection program has real headroom (C2 is plumbing). If verification ≈ generation and
verifier-selection ≈ pass@1, C2 is a capability limit and the portfolio should pivot to coverage.

## Setup

- Model Qwen3.5-4B frozen (bf16, fast path). MBPP sanitized `test`, 100 tasks, k=8 no-think candidates,
  execution-labeled.
- **Verification judge:** present the task spec + one example assert + the candidate code; read the A/B
  logit at a forced answer position → P(correct) (A=32 correct, B=33 incorrect, single tokens). No-think:
  one forward. Thinking: generate up to a 1024-token budget, force `</think>`, then read the A/B logit.
- **Metrics:** generation pass@1 / oracle pass@k; verification balanced-accuracy + AUROC (discriminate the
  model's OWN correct vs incorrect candidates) at no-think and think; verifier-selected best-of-k accuracy
  vs pass@1 vs oracle (the fraction of the pass@1→oracle gap the verifier closes); the thinking asymmetry.
- **Controls:** foreign-solution judge (a different task's candidate → should be rejected, P(A) low) tests
  that the verifier actually reads the task; say-A rate + balanced accuracy control the yes-bias.

## Run

```bash
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ../../.venv/bin/python scripts/run.py --tasks 100 --k 8 --budget 1024
../../.venv/bin/python scripts/verify.py      # execution labels
../../.venv/bin/python analysis/analyze.py     # metrics + figure
```

## Results

Full results in [reports/report.md](reports/report.md).

| quantity | no-think | thinking |
| --- | ---: | ---: |
| verification balanced accuracy | 0.627 | **0.827** |
| verification AUROC | 0.773 | **0.926** |
| verifier-selected best-of-8 (deployable) | 0.800 | **0.860** |
| pass@1(0.771)→oracle(0.890) gap closed | +24% | **+75%** |
| foreign reject rate | 1.00 | 1.00 |

- **Checking is easier than doing — but only with thinking.** No-think self-verification is weak/yes-biased
  (AUROC 0.77, says "correct" 91%); thinking makes it a real critic (AUROC **0.93**).
- **C2 is plumbing, not capability.** The model's own black-box, training-free, deployable thinking-verifier
  closes **75%** of the pass@1→oracle gap. The selection program has real headroom.
- **C9 inversion:** thinking helps *verification* (+0.20 balanced-acc) at least as much as generation — its
  deepest value may be helping the model *know* which answer is right.

## Interpretation

The selection bottleneck isn't that the model can't tell right from wrong — it can, well, once it thinks.
The highest-leverage "plumbing" for C2 is thinking-augmented self-verification (cheaper and stronger than the
trained selectors the corpus favored). Thinking (C9) and selection (C2) meet in thinking-verification. See
claim C10.

## Artifacts

- `src/judge_lib.py` (generation + A/B-logit verification judge), `src/tasks.py`. `scripts/run.py`,
  `scripts/verify.py`, `analysis/analyze.py`. `data/records.jsonl`, `data/labels.jsonl`, `data/tasks.json`
  (small, in-repo). No external artifacts (behavioral + logit-read; no activation caching).

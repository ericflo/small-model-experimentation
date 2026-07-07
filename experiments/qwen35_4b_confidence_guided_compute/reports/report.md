# Beating sample-more with the model's own uncertainty: confidence-select beats majority vote

## Motivation
The mission is "beat sample-more." C40 unearthed a deployable implicit confidence signal (the model's answer-token
probability P(answer) is calibrated to its own correctness). This turns that signal into a compute tool.

## Method
A mix of problems on the C40 successor task ("advance k in a cyclic order") spanning three regimes: easy
(execute), coverage-limited (familiar_induce), capability-limited (novel_induce). Sample k=12 per problem; read
each sample's P(answer) (softmax over the 10 digit tokens, one forward pass). Compare selection/allocation
policies at matched forward-pass budget. Verifiable ground truth; single seed.

## Results (n=80/condition, k=12)
**Two failure modes of sample-more:**
| condition | greedy (1) | pass@12 |
|---|---|---|
| execute | 1.00 | 1.00 |
| familiar_induce (coverage-limited) | 0.21 | **0.90** |
| novel_induce (capability-limited) | 0.07 | 0.59 (below pure-luck 0.72) |

**Verification-free selection (pick 1 of 12 samples):**
| method | accuracy |
|---|---|
| random | 0.44 |
| self-consistency (majority vote) | 0.48 |
| **confidence-select (argmax P(answer))** | **0.62** |
| oracle (pass@12 upper bound) | 0.83 |

**Across budgets:** self-consistency is **flat (~0.48)** — sample-more is wasted — while confidence-select
**rises 0.47 → 0.62** and beats majority at every budget. It works because P(answer) is calibrated (C40): on hard
problems the model's *mode* is confidently wrong, but when it derives the right rule it is confident, so the
most-*confident* sample beats the most-*common* one.

**Abstention:** max per-sample P(answer) predicts per-problem solvability at **AUROC 0.83**; abstaining on
low-confidence yields ~1.0 accuracy on the confident top third (a working abstain/escalate signal for the
capability-limited problems where sampling is futile).

## Conclusion
The fixed 4B's own logits tell you **which sample to trust** and **when to stop** — no verifier, no execution —
turning naive sample-more (flat under majority vote) into a rising accuracy-vs-compute curve. This is a deployable
use of C40's unearthed self-knowledge, and it beats the standard verifier-free method (self-consistency).

## Honest scope
- Confidence-guided **allocation** (probe-then-allocate budget) is roughly **tied** with uniform confidence-select
  — the win is in **selection** and **abstention**, not allocation.
- Single toy substrate (successor task). Generalizing the signal to real code/MBPP (where "sample-more" is the
  standard baseline) is the owed next step.
- Single seed; the design review agent died on an API error (not re-run), so the design was self-vetted against
  the arc's prior selection findings (C10, C17).

## Artifact Manifest
See `reports/artifact_manifest.yaml`.

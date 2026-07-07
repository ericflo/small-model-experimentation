# Does the model know when it will fail? Yes implicitly (answer-token probability), no explicitly (self-report)

## Motivation
The arc pinned a **verified** competence boundary (C39): on "advance k in a cyclic order" the model executes
near-perfectly, induces a familiar rule partly (~0.45), and induces a novel rule at chance (~0.10). Unlike normal
calibration work, we know exactly which tasks the model gets right and *why*. So: does the model's own
confidence/uncertainty track that boundary — does it know when it's guessing?

## Method (review-hardened)
Format-equalized single-value task (every condition shows an order block, so block-presence isn't a cue).
Conditions: familiar_execute (anchor ~1.0), familiar_induce (**headline** — intermediate acc, surface-matched),
reversal_induce (intended dissociation), novel_induce (chance). Verbalized 0–100 confidence is a degenerate
constant 100, so we use **two non-degenerate logit signals**:
- **Implicit — P(answer):** the model's probability on the digit it emits (softmax over the 10 digit tokens at the
  `Answer: ` position, one forward pass) + entropy + top-2 margin.
- **Explicit — P(True):** Kadavath-style self-verification ("is your answer correct? A/B"), read P(A).

The clean self-knowledge test is **within-condition item-level AUROC in familiar_induce** (surface matched, both
classes present), compared against an **external surface-feature baseline** (logistic regression on
{k, gap-to-seen, n-distinct-seen, query}) — a signal is self-knowledge only if it *beats* surface.

## Results (n=150/condition)
**Condition-level calibration:**
| condition | acc | mean P(answer) | mean P(True) |
|---|---|---|---|
| familiar_execute | 1.00 | 1.00 | 0.42 |
| familiar_induce | 0.40 | 0.44 | 0.35 |
| reversal_induce | 0.19 | 0.29 | 0.38 |
| novel_induce | 0.10 | 0.15 | 0.38 |

Implicit P(answer) tracks accuracy almost perfectly; explicit P(True) is **flat (~0.4)** and even *underconfident*
on the perfect execute cell.

**Headline — within familiar_induce (surface-matched, acc 0.40), AUROC predicting per-item correctness:**
| signal | AUROC |
|---|---|
| **P(answer) (implicit)** | **0.95** (95% CI 0.90–0.99) |
| margin / −entropy | ~0.90 |
| external surface baseline | 0.61 |
| **P(True) (explicit)** | **0.46** (= chance) |

The implicit signal predicts *which specific items* the model gets right, far beyond surface features — genuine
item-level self-knowledge. The explicit signal is at chance.

**Deployable:** selective prediction by low P(answer) lifts accuracy on attempted from 0.23 to ~1.0.

**Mechanism:** on novel_induce only 0.11 of wrong answers are the natural-successor intrusion → failures are
high-entropy scatter (the model *is* uncertain there), not a confident consistent wrong-rule.

## Conclusion
**The model knows when it will fail — but only in its output distribution, not in anything it can say.** Implicit
metacognition (answer-token probability) is excellent and beats both surface and explicit self-report; explicit
self-assessment (P(True), verbalized confidence) is broken. For deployment: read the answer-token probability as a
confidence/abstain signal; never trust the model's explicit self-assessment. This is a latent capability
*unearthed* (a usable self-knowledge signal exists in the fixed weights) with a sharp caveat on where to read it.

## Honest caveats
- P(True) at chance may partly reflect elicitation, but the independently-degenerate verbalized-100 confirms
  explicit metacognition is broken regardless.
- reversal_induce turned out genuinely hard (0.19), not the intended "scrambled-looking-but-easy" dissociation, so
  the surface-vs-competence contrast rests on the external surface-baseline comparison (0.61 ≪ 0.95), not reversal.
- Single seed; no-think answer channel (think triggers code-mode — C39).

## Artifact Manifest
See `reports/artifact_manifest.yaml`.

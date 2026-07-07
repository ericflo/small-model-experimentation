# Can SFT install the skill of induction? Partially — the wall is neither a hard bound nor cleanly liftable

## Motivation
The arc's central law: the fixed 4B is an executor/retriever, not an inducer of novel structure (C38/C39). The
mission-core question: can QLoRA SFT *install* the induction skill so it generalizes to held-out rules — lifting
the wall — or is induction fundamentally un-installable?

## Method (review-hardened)
Each episode = a random **scrambled digit order** (stated) + a hidden rule + 6 examples + a query; the model must
infer the rule and apply. Base fails this at chance (per C39). Rule families over positions in the order: **shift**
`f(order[i])=order[(i+k)%10]` (train) and **affine** `order[(a·i+b)%10]`, a∈{3,7,9} (out-of-family). Random
orders/params → held-out episodes are genuinely novel. Answer-only QLoRA (r32/α64). **The review's mandatory
gate:** measure the base **execute** ceiling (rule stated) per family — an induction failure is only meaningful if
the base *can* execute the rule. Eval = forced `Answer: ` argmax over the 10 digit tokens (fair for base + SFT).

## Results
| | base | SFT-4k | SFT-8k | execute ceiling |
|---|---|---|---|---|
| **shift induce** (in-family) | 0.087 | 0.35 | **0.40** | 0.72 |
| **affine induce** (out-of-family) | 0.213 | 0.267 | 0.297 | 0.457 |
| shift **execute** | 0.72 | **0.093** | — | — |

- **SFT partially lifts the wall, data-limited:** shift induction 0.087 (chance) → 0.35 → 0.40, ~4.6× chance and
  still rising with data — but it **plateaus well below the execute ceiling (0.72)**, so the skill is only
  partially installed.
- **Weak out-of-family transfer:** affine induction barely moves (0.21 → 0.30, far below in-family and its 0.46
  ceiling) — the model learned a **shift-specific procedure**, not general induction.
- **Two costs:** (1) **catastrophic forgetting** — answer-only SFT crashed execute from 0.72 → 0.09; (2) a
  default-fallback digit **bias** that shrinks with data (37% → 20%).

## Conclusion
The induction wall is **neither a hard architectural bound** (SFT lifts it several-fold, scaling with data) **nor
cleanly liftable** (partial, procedure-specific, forgets execution). This is exactly what the arc predicts: trained
to induce, the fixed 4B learns a *specific procedure*, not the *general skill* — it remains an executor at heart.

## Honest scope
- Single seed; answer-only SFT (a **reasoning-SFT** arm and a **multi-family leave-one-out** design for *general*
  induction are the owed next steps, per the review — this run tests shift→affine procedure-specificity, a weaker
  probe).
- Affine OOF is partly **execution-limited** (0.46 base ceiling), so its ceiling is lower than shift's.
- The '8'-bias caveat: SFT-8k accuracy (0.40) far exceeds the frequency baseline (~0.10); most is real induction.

## Artifact Manifest
See `reports/artifact_manifest.yaml`.

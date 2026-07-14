# Paired Power and Attainability

The primary qualification and confirmation blocks each contain 144 paired tasks,
48 per family. The preregistered capability gate requires both an observed
correct-reflection improvement of at least 0.10 and a positive 95% paired-bootstrap
lower bound, independently against shuffled reflection and frozen sample-more.

For paired binary coverage, let `d = p10 + p01` be the discordant-pair rate and
`delta = p10 - p01` the true paired gain. The large-sample standard error is
`sqrt((d - delta^2) / 144)`. A two-sided-normal approximation gives:

| True gain | Discordance | Approx. power for lower 95% > 0 |
|---:|---:|---:|
| 0.10 | 0.20 | 0.79 |
| 0.10 | 0.30 | 0.61 |
| 0.10 | 0.40 | 0.48 |
| 0.15 | 0.30 | 0.93 |
| 0.15 | 0.40 | 0.83 |
| 0.20 | 0.40 | 0.98 |

This is intentionally a large-effect screen. It has good power for the 0.15–0.20
gains that would justify an eight-adapter/two-seed program, but it is not designed to
promote a fragile five-point effect. The 0.10 point threshold and positive bootstrap
bound are conjunctive; at high discordance the realized effect must exceed 0.10 to
pass. Family breadth uses a point threshold of 0.05 at `n=48` rather than a separate
family-level confidence interval, but all three families must pass and no pooled
result can rescue a miss.

The same complete gate is required on both training seeds and again on the untouched
confirmation block. These repeated gates reduce the probability that one favorable
adapter seed or split promotes a false capability claim. They also make the protocol
conservative: a real effect smaller than the operationally useful threshold is an
honest negative for this branch, not evidence that reflection transfer is impossible.

The direct plan-plus-answer positive control has a deliberately higher sanity bar:
coverage@16 at least 0.50, improvement over frozen at least 0.20, and a positive
paired lower bound. Failure means the training/generalization setup is not sensitive
enough to adjudicate the treatment and stops the branch.

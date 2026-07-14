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

## End-to-End Matched-Compute Promotion

The ordinary qualification and confirmation decisions compare all arms at 16
candidates to determine whether training produced a broad, replicated distributional
shift. That is not the final sample-more comparison. After both confirmation decisions
pass, frozen Qwen receives fixed-seed blocks of 16 candidates per task on the same
persistent vLLM engine. Block generation stops using compute fields only—never labels,
scores, or correctness—at the first completed block satisfying both:

- token-forward equivalents at least the larger seed's full training charge
  (`forward_tokens × 3`) plus its correct-model confirmation inference tokens; and
- wall time at least the larger seed's training GPU phase plus correct-model load and
  confirmation generation, versus one frozen load plus every completed block.

The complete 36-step training cost is charged to each 144-task confirmation split;
there is no cross-task, cross-seed, or hypothetical deployment amortization. The
reservoir is capped at 16 blocks (256 candidates/task). Failure to reach both units by
that cap is a protocol failure. Each correct-reflection seed must independently have
a strictly positive mean coverage@16 advantage over the resulting frozen coverage,
a positive paired-bootstrap lower 95% bound, and a nonnegative point delta in every
family. Thus a larger trained-vs-frozen effect at equal candidate count cannot by
itself support the capability claim.

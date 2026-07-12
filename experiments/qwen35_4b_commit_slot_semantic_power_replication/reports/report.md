# Qwen3.5-4B Commit-Slot Semantic Power Replication Report

## Status

Terminal seam result `POWERED_COMMIT_SLOT_SEAM_REPLICATED`. Qualification and
untouched confirmation independently passed every frozen gate. J stages remain
sealed pending a new outcome-blind implementation and adversarial audit.

## Purpose

The parent fixed slot repaired answer mode and showed a +8.33pp ordered-over-
shuffled hint at cap 1,024, but it missed the mixed-task gate and task-level
uncertainty crossed zero. This experiment fixes that one cap/interface and tests
the hint with 113 fresh task units per seam stage plus bootstrap and diversity
gates.

## Results

CPU only: 322/322 unique exact-depth tasks, zero overlap with five parents,
balanced 10--11 target tasks per seam split, exact lens hash, reachable gates,
and seven passing unit tests. Parent-effect planning requires and assigns 113
tasks per stage for approximate power 0.802745; the actual decision uses a
10,000-resample task bootstrap.

No correctness row, scientific trace, or scientific summary exists yet.

After the immutable design boundary, outcome-blind model smoke passed the exact
revision, 32-layer/2,560-wide architecture, five rank-24 lens matrices, 12
distinct leading-space aliases, fixed slot tokens `[271, 5170, 25]`, finite
logits, and native/free-form cache contracts. Peak allocation was 8,514,319,872
bytes. The receipt stores no correctness, chosen alias, or trace text. A final
implementation audit then verified task bootstrap, diversity, exact cardinality,
and confirmation hash locks before any scientific run.

## Qualification result

The fixed-cap scientific stage completed exactly 339 native traces, 339 real
slots, 339 exact-token-multiset shuffled slots, 339 close-only free-form
controls, and 113 no-thought slots. Every native path contacted cap 1,024. All
cache, finite, exact-shuffle, cardinality, data, design, and power contracts
passed.

| metric | observed | frozen requirement | pass |
| --- | ---: | ---: | --- |
| real slot accuracy | 92/339 (0.271386) | 0.20--0.70 | yes |
| no-thought accuracy | 11/113 (0.097345) | real minus >=0.03 | yes (+0.174041) |
| shuffled accuracy | 46/339 (0.135693) | real minus >=0.05 | yes (+0.135693) |
| one-sided 95% task lower, real−shuffle | 0.088496 | >0 | yes |
| mixed real tasks | 32/113 | >=28 | yes |
| correct-alias support | 11 | >=8 | yes |
| chosen-alias support | 12 | >=8 | yes |
| unmasked top-is-alias | 0.882006 | >=0.75 | yes |
| mean total alias mass | 0.667938 | >=0.50 | yes |
| finite real rows | 1.0 | 1.0 | yes |

The no-thought task-bootstrap lower diagnostic was also positive (0.120944),
though it was not a powered primary gate. Correct-alias probability averaged
0.266564 under real thought versus 0.164722 shuffled. Alias mentions were not
required: only 16.2% of real prefixes contained the correct alias token.

The interface itself remains valid without relying mainly on masking: an alias
was already the unmasked top token on 299/339 rows and the 12 aliases held 66.8%
mean full-vocabulary probability. Close-only free-form remained much worse:
57/339 parsed, 20/339 were correct, and 310/339 exhausted 16 answer tokens.

Qualification sampled 347,136 native thought tokens in 11,669.621 seconds and
processed 492,435 real plus 492,435 shuffled slot-prefill tokens. Peak allocated
GPU memory was 8,706,993,152 bytes.

The automatic qualification decision was `POWERED_COMMIT_SLOT_SEAM_QUALIFIED`
at the only registered cap 1,024. It opened exactly one hash-locked confirmation
and no J stage by itself.

## Independent confirmation result

The untouched confirmation then completed exactly the same 339 native traces,
339 real slots, 339 exact-token-multiset shuffled slots, 339 close-only controls,
and 113 no-thought slots. Every path again contacted cap 1,024. No selection row
was pooled into a confirmation decision.

| metric | confirmation | frozen requirement | pass |
| --- | ---: | ---: | --- |
| real slot accuracy | 98/339 (0.289086) | 0.20--0.70 | yes |
| no-thought accuracy | 8/113 (0.070796) | real minus >=0.03 | yes (+0.218289) |
| shuffled accuracy | 47/339 (0.138643) | real minus >=0.05 | yes (+0.150442) |
| one-sided 95% task lower, real−shuffle | 0.094395 | >0 | yes |
| mixed real tasks | 31/113 | >=28 | yes |
| correct-alias support | 10 | >=8 | yes |
| chosen-alias support | 12 | >=8 | yes |
| unmasked top-is-alias | 0.876106 | >=0.75 | yes |
| mean total alias mass | 0.663490 | >=0.50 | yes |
| finite real rows | 1.0 | 1.0 | yes |

The no-thought task lower diagnostic was 0.165192. Correct-alias probability
averaged 0.239230 under ordered thought versus 0.153041 shuffled. Only 19.8% of
ordered prefixes contained the correct alias token, and the post-decision
mention strata were not favorable to copying: success was 26.9% with a mention
versus 29.4% without one. The unrestricted next token was already an alias on
297/339 rows, and aliases carried 66.35% mean full-vocabulary mass. Close-only
free-form again remained unusable: 55/339 parsed, 20/339 succeeded, and 316/339
exhausted its answer cap.

Confirmation sampled 347,136 native thought tokens in 11,690.539 seconds and
processed 491,457 real plus 491,457 shuffled slot-prefill tokens. Peak allocated
GPU memory was 8,704,452,608 bytes. The automatic terminal seam decision is
`POWERED_COMMIT_SLOT_SEAM_REPLICATED`.

## Post-decision replication audit

The deterministic audit preserves the two stage decisions and adds two-sided
task-bootstrap diagnostics. Ordered-minus-shuffled was 0.135693 [0.079646,
0.191740] in qualification and 0.150442 [0.082596, 0.218289] in confirmation.
The difference between those two independent effects was 0.014749 with interval
[-0.073746, 0.103245], providing no evidence of stage drift. At the paired-path
level, ordered-only wins versus shuffled-only wins were 60:14 and 64:13. Task
effects were positive/zero/negative on 35/72/6 and 34/71/8 tasks.

The pooled 226-task effect, explicitly diagnostic and unnecessary for either
pass, was 0.143068 [0.098820, 0.187316]. It cannot rescue a failed stage.

Alias identity remains an important nuisance. Confirmation successes spanned
10 of 11 target aliases, but `horse` had 0/30 ordered successes. Shuffle beat
ordered thought for `tiger` (28/30 versus 21/30) and `river` (13/30 versus
11/30), while most other targets favored ordered thought. This heterogeneity
does not defeat the registered breadth/task gates, but it forbids treating raw
alias logits or identity as a certainty coordinate. Any value model must use
task-held-out evaluation and demonstrate incremental signal over correct-alias
activity, slot margin, and alias identity.

## Boundary

This is replicated constrained semantic elicitation: ordered native thought
changes the fixed semantic commit choice beyond syntax and identical token-bag
controls. It is not autonomous termination, free-form capability, J certainty,
or installed capability. Gold labels evaluate the seam. J/value/control/causal
commands still fail closed until a separately committed outcome-blind audit and
implementation boundary.

## Prefix-value implementation boundary

After the seam decision, a new preregistration and 30-point adversarial design
review froze a prospective rather than merely endpoint readout: midpoint
coordinates must rank each path's later full-cap correct-alias probability, and
endpoint signal cannot rescue a midpoint miss. The primary 120 J features use
all 24 coordinates at layers 4--8 with task-held-out, within-task/fraction
centered ridge evaluation. Mandatory matched pipelines use five gold-alias J
activities, ordinary slot margin, alias identity, and 120 layer-matched random
coordinates orthogonal to the complete J span. Task bootstrap and 32 within-
group shuffled refits are load-bearing.

Implementation is complete and 16 outcome-blind tests pass, including exact
three-path groups, future-label timing, whole-task folds, train-only scaling,
non-J projection <=1e-5, reserved-data loader isolation, and pending-boundary
failure before model load. The exact code/audit is anchored to pushed commit
`ddbc1969`; only the outcome-blind value-model smoke is now authorized. No
`value_fit` or `causal_confirmation` row had been opened at that boundary.

The subsequent one outcome-blind model smoke passed at 8,510,865,408 peak
allocated bytes. All five J dictionaries retained rank 24; J and non-J feature
widths were each 120 and finite; non-J projection into J-space was at most
`2.67e-7`. The live feature sequence was 384 tokens with no close/slot, while
the separate slot prefill was exactly four tokens longer. The receipt stores no
outcome, correctness, chosen alias, probability, or trace text, and both
reserved splits remained unopened. The single scientific prefix-value run is
now authorized. Control calibration and causal confirmation remain fatal-
unavailable.

## Artifact Manifest

See `artifact_manifest.yaml`.

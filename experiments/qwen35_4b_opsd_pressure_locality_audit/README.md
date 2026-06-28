# qwen35_4b_opsd_pressure_locality_audit

Standalone no-training audit for positive-only on-policy self-distillation.

The audit asks whether a privileged hinted teacher provides dense token-level signal at the exact places where hidden-correct code diverges from visible-pass hidden-wrong near-misses. It is a gate before any OPSD/OPD training.

## Primary Gate

For matched hidden-correct and visible-pass hidden-wrong candidates on the same task, find same-prefix code forks. At each fork, score:

`log p_teacher(correct_branch | shared_prefix, weak_hint) - log p_teacher(wrong_branch | shared_prefix, weak_hint)`

The weak hint is the retrieved verified algorithm associated with the hidden-correct adaptation. Full reference-code hints are included only as a leakage ceiling, not as success evidence.

## Supporting Analysis

The package also reports token-level student/teacher gaps for full correct and wrong rollouts:

`gap_t = log p_teacher(token_t | prefix, hint) - log p_student(token_t | prefix)`

Token buckets split shared boilerplate, discriminating correct chunks, discriminating wrong chunks, parse/format tokens, and other tokens. Discriminating forks are further stratified by whether their branch text overlaps with the retrieved hint.

## Decision Rule

Run training only if weak-hint teacher preference is positive on task-specific forks, above shuffled-hint control, and not merely a full-reference leakage effect. If weak hints only upweight shared or hint-overlap tokens, this audit kills the OPSD training run.

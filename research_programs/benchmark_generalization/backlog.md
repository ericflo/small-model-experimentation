# Backlog

## Next Experiments

- Active cross-program test: `qwen35_4b_specialist_policy_integration` uses
  no-new-exposure primitive families plus fully held-out pairwise and
  three-primitive compounds to distinguish capability union from composition.
  Its confirmatory distribution is frozen after level-only calibration; items
  may not be filtered against model outputs.
- Build a common shift taxonomy: length, family, primitive, composition, prompt, format, and real-task shift.
- Re-run top mechanisms on at least one non-original substrate.
- Add bridge-composition and held-out-primitive splits to new experiments by default.
- Create tiny smoke suites for fast sanity and larger challenge suites for claims.
- Track which claims are single-substrate versus cross-substrate.
- Run the C45 follow-up as compositional-grammar induction, not a flat non-affine menu: train reasoning-SFT on condition x action depth-1 rules, then test held-out combinations, held-out productions, and held-out composition-depth (depth-2 nested/two-action chains) as separate endpoints. Gate every family with execute-given-rule ceilings, a token-budget/truncation curve, and automated example-set sufficiency checks so failures distinguish search/composition limits from non-executable or underdetermined episodes.

## Required Controls

- IID split.
- At least one held-out split.
- Baseline repeated on every split.
- Report both row-level and task-level metrics when applicable.

## Stop Conditions

Do not promote a result to shared strategy if it has only been shown on a single easy or IID split.

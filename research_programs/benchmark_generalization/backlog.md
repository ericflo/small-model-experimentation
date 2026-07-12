# Backlog

## Next Experiments

- Stopped cross-program test: `qwen35_4b_specialist_policy_integration` kept its
  no-new-exposure compound and confirmatory distributions sealed, but stopped
  before training because one mandatory specialist target exceeded the score
  ceiling. A new experiment may reuse the generalization design only with a
  harder independently calibrated tools/provenance core and fresh frozen
  confirmatory seeds.
- Completed proxy-transport negative: `qwen35_4b_pareto_policy_integration`
  kept every benchmark seed sealed and tested C54's external quick/medium
  labels on a fresh procedural quick/deep proxy. The required crossover did not
  reproduce: `blend` lost both quick blocks. Future distillation work must
  distinguish instrument-specific ranking from teacher advantage on the
  training-state distribution; neither can stand in for the other.
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

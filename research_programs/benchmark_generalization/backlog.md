# Backlog

## Next Experiments

- Build a common shift taxonomy: length, family, primitive, composition, prompt, format, and real-task shift.
- Re-run top mechanisms on at least one non-original substrate.
- Add bridge-composition and held-out-primitive splits to new experiments by default.
- Create tiny smoke suites for fast sanity and larger challenge suites for claims.
- Track which claims are single-substrate versus cross-substrate.

## Required Controls

- IID split.
- At least one held-out split.
- Baseline repeated on every split.
- Report both row-level and task-level metrics when applicable.

## Stop Conditions

Do not promote a result to shared strategy if it has only been shown on a single easy or IID split.

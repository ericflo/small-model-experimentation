# Post-Model-Smoke 001 Audit: Invalid Receipt

## Verdict

`INVALID_NUMERIC_RECEIPT`. This is an implementation finding only. It neither
passes nor fails native J controllability, and mechanics remains unavailable.

## Evidence

- Exact model/revision/lens loaded; prompt 386 tokens; generated live prefix 32
  tokens; peak allocated memory 9,112,936,960 bytes.
- All five J and non-J hooks reported one application and finite downstream
  logits.
- Every realized delta was recorded as zero while every requested J norm error
  was exactly 1.0—physically incompatible with the requested nonzero branches.
- `numeric.passed` was nevertheless true because requested J norm error was not
  included in its conjunctive expression.
- No branch probabilities, choices, task labels, correct alias, or scientific
  outcome were written.

## Root cause

The hook bound `current` as a view into `patched`. After calculating `changed`,
it assigned `changed` into `patched` before subtracting `current`; the view now
contained `changed`, so subtraction returned zero. The input activation receipt
was corrupted by the same aliasing.

## Repair boundary

- clone `current.float()` before any assignment;
- calculate both input receipt and realized delta against that clone;
- make maximum J requested-norm relative error <=1e-5 mandatory;
- assert the unit-test patcher receipt equals the injected branch matrix; and
- re-anchor all changed implementation hashes before smoke 002.

Do not overwrite the invalid receipt, weaken numeric tolerances, or proceed to
mechanics based on its false pass.

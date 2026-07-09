# Interface attempt 3 failure: syntax repaired, exact alias fidelity failed

Date: 2026-07-09. Status: final interface-gate failure; registered stop rule fired.

## Scope and boundary

Interface attempt 3 repeated only the four task-independent, train-only, plan-given transcription
records from attempt 2. The verified primitive plan was supplied directly, so this gate measured
whether `Qwen/Qwen3.5-4B` could rewrite it as an exact shortest macro surface. It did not measure
program induction, visible-only selection, or held-out task solving.

Every completion used the copied vLLM runner in `thinking: off` mode. The pipeline stopped at this
gate. No fresh `smoke-v2-*` induction prompt and no full-evaluation prompt was ever generated or
shown to the model.

## Registered gate result

A record succeeded only if at least one completion:

- passed the strict one-line program parser;
- used a supplied macro;
- had the optimal surface length; and
- expanded exactly to the supplied primitive plan.

| Metric | Attempt-3 result | Gate |
| --- | ---: | ---: |
| Records | 4 | frozen |
| Samples | 16 | frozen |
| Strictly valid samples | 16 | diagnostic |
| Macro-using samples | 16 | diagnostic |
| Answer truncation | 0/16 = 0.0000 | below 0.05: pass |
| Successful records | 1/4 (`00`) | at least 3/4: **fail** |

No-think removed the answer-boundary failure: every sample was syntactically valid, every sample
called at least one macro, and none truncated. Exact semantic transcription nevertheless remained
unreliable.

## Error pattern

The committed post-gate audit in `analysis/interface_v3_audit.json` regenerates the failure
taxonomy directly from the raw vLLM rows. Three of 16 samples expanded exactly, all on record
`00`; the other 13 all used multiple macros and expanded to 6--10 primitives, beyond the stated
limit of five. Ten of those 13 included the correct designated alias but appended unrelated
aliases, while three omitted the designated alias entirely. The resulting programs were short
and parseable, but their literal expansions differed from the supplied plans.

## Interpretation

Attempts 2 and 3 separate two interface failures:

- budgeted thinking caused answer spill and truncation in attempt 2;
- no-think repaired syntax, termination, and macro calling in attempt 3, but not faithful alias
  substitution.

Parse rate and raw macro-use rate would have made attempt 3 look perfect. Exact expansion checking
correctly rejected those superficially compliant outputs. This is a reusable lesson for any
learned operator surface: invocation is not evidence of correct abstraction use unless the call
expands to the intended behavior.

## Stop decision

Amendment 2 precommitted that another failure ends the experiment before fresh induction smoke.
That condition is met. There will be no amendment 4 and no full run in this experiment.

The verified-macro invention hypothesis remains unresolved because its fresh scientific smoke was
never run. This result warrants no claim-ledger entry. Any further attempt must be a new material
follow-up with its own intake, adversarial review, preregistration, fresh interface design, and
matched-compute controls; it must not overwrite or extend this stopped experiment.

The exact attempt-3 config, raw vLLM outputs/sidecar, verdict, and source are preserved under the
paths recorded in `artifact_manifest.yaml`.

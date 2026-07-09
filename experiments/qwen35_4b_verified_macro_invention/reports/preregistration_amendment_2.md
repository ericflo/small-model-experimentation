# Preregistration amendment 2: no-think plan-transcription retry

Date frozen: 2026-07-09, after interface attempt 2 and before any interface-attempt-3 GPU call.

This amendment changes only the non-scored, train-only, plan-given interface gate introduced in
[`preregistration_amendment_1.md`](preregistration_amendment_1.md). It does not change the macro
proposal stage, fresh induction smoke, full evaluation, or any scientific decision rule.

## Attempt-2 result and stopping boundary

Interface attempt 2 contained four task-independent records with four vLLM samples each. A record
counted as successful only if at least one strictly parsed completion used a supplied macro, had
the optimal surface length, and expanded exactly to the supplied verified primitive plan.

| Attempt-2 metric | Result | Gate |
| --- | ---: | ---: |
| Records | 4 | frozen |
| Samples | 16 | frozen |
| Successful records | 2 (`00`, `02`) | at least 3/4 |
| Strictly valid samples | 4 | diagnostic |
| Macro-using samples | 4 | diagnostic |
| Answer truncation | 12/16 = 0.75 | below 0.05 |

The gate failed both registered requirements. All four valid samples used a macro, so the
remaining failure is not evidence that the alias notation itself is unusable; it is primarily a
reliability and answer-boundary failure under budgeted thinking. The pipeline stopped immediately.
No fresh `smoke-v2-*` induction prompt was generated or shown to the model, and no fresh smoke or
full output was inspected.

The exact failed attempt is preserved at:

- `configs/interface_v2.yaml`;
- `runs/interface_v2_failed/`;
- `analysis/interface_v2_gate_failed.json`; and
- `archive/interface_v2_source/`.

## Attempt-3 scope

Retry **only** the same task-independent, train-only, plan-given transcription gate. Interface
attempt 3 uses the experiment-local vLLM runner with:

- `thinking: off` (the runner's exact no-think mode);
- `n: 4` per record;
- `answer_max_tokens: 128`; and
- the same explicit temperature, top-p, top-k, parent seed family, four
  prompt contents, supplied target plans, designed library, parser, executor, and success rule.

Attempt-qualified record ids and output paths may differ only to prevent accidental cache reuse;
the underlying prompt content, targets, and parent run seed remain fixed. No prompt wording,
demonstration, macro definition, target, or threshold may be selected from attempt-3 output.

## Unchanged attempt-3 gate

Attempt 3 passes only if both original interface requirements hold:

1. at least 3 of the 4 records have a strictly parsed, macro-using, optimal-surface completion
   whose expansion exactly equals the supplied plan; and
2. answer truncation is below 0.05 across all 16 samples.

There is no relaxed parser, candidate salvage, best-of-retry threshold, or scientific metric at
this stage. Failure stops the experiment before any fresh induction smoke prompt.

## Rationale for no-think

The plan is supplied verbatim, so this gate measures exact alias transcription and one-line
formatting, not rule induction, program search, or hidden-task reasoning. Budgeted thinking made
all 16 attempt-2 samples exhaust their 768-token thinking allowance, after which 12 spilled into
the 128-token answer cap. No-think isolates the syntax/transcription capability the gate was
created to test and removes the force-close boundary that dominated attempt 2.

This is not permission to turn thinking off for the scientific task. If attempt 3 passes, the
still-unseen fresh induction smoke uses the unchanged amendment-1 protocol: vLLM budgeted thinking
at 768 tokens, answer cap 128, matched K=12 base/designed arms, the same frozen
`smoke-v2-reuse-NNN` and `smoke-v2-no-reuse-NNN` tasks, and the same smoke gates. The full protocol
and every confirmatory rule remain unchanged.

# Adversarial Design Review

## Status

`HOLD_LIVE_CALLS`. Independent adversarial review of scaffold commit
`56b0b67b0b054610f783d6b6107a0e4c32b5c95e` returned `HOLD_DESIGN`.
The remediation is prospective and must receive a new independent review at an
exact pushed commit before construction or live calls.

## Self-review before delegation

- The experiment is a fresh result-bearing successor, not an in-place parser
  repair.
- Tokenizer EOS applies only to answer stages; thought semantics stay fixed.
- First-stop geometry and every pre-commit token are authenticated.
- HF EOS is a matched boundary control rather than historical prose.
- The two no-think prefix cells are paired and not called replications.
- Fresh task/function/request/seed identities are mandatory.
- Calibration remains known-answer interface measurement, not capability.
- Mechanics, protected labels, and benchmarks remain behind staged locks.
- The branch terminates if no tokenizer-EOS arm independently qualifies.

Independent review must inspect construction identity, boundary pairing,
malformed-stop controls, transaction order, matched-compute accounting, and the
hidden-label firewall before any model/GPU call.

## Independent review: first pass

The reviewer found four scientific blockers.

1. Boundary causality was not fail-closed. The phrase "matching sampled
   prefixes whenever both traces reach 248046" excluded precisely the early-HF
   and cap cases that could invalidate the comparison.
2. Thinking was not isolated across the four thinking cells. Matching numeric
   seeds cannot substitute for a single persisted shared-thought transaction
   because vLLM sampling is not batch-invariant.
3. The grammar froze two aliases while requiring both arity-two and
   arity-three blocks, and the unquoted YAML scalar `PROGRAM:` parsed as a
   mapping.
4. Conditional mechanics left task counts, strata, direct-pool bounds, resource
   matching, selector/inference, effect floors, terminal outcomes, and lock
   order adaptable after calibration.

The separate implementation review also noted that the scaffold runner is
still HF-EOS-only and that the smoke checker does not yet bind live stop/
finish/cap metadata, shared thought rows, transaction durability, or semantic
grammar. Those are expected scaffold limitations but independently block a
live lock.

## Prospective remediation

- All 192 tokenizer/HF pairs now fail closed on identical token prefixes
  through the earliest registered stop or cap, plus identical prompts, seeds,
  shared thought, injected prefix, adjacency, and batch geometry. Any mismatch
  terminates `BOUNDARY_PAIRING_INVALID` before cell metrics are used.
- Exactly one persisted thought row per task feeds all four thinking answers.
  Content from the first natural `</think>` onward is discarded, and every
  answer is rebuilt behind exactly one injected close.
- The grammar is arity-parametric for `k in {2,3}`, and `"PROGRAM:"` is quoted
  in configuration.
- Fresh seeds/namespaces, 24-task 8/8/4/4 strata, 24 candidates, 96 direct rows,
  24-row transport, two first-over resource matches, selector, inference,
  large-effect floors, terminal outcomes, and staged hidden-lock order are now
  frozen in the preregistration and config.

This document remains `HOLD_LIVE_CALLS` until the exact remediation commit is
pushed, both workflows are green, and the independent reviewer returns a
design pass. A design pass will not authorize model calls by itself; a later
implementation review and committed-green implementation lock are also
required.

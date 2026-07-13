# Mechanics result adversarial review

**Review date:** 2026-07-13

Three independent read-only audits reviewed the completed mechanics run before
publication. They accessed no benchmark, hidden, qualification, or confirmation
content and made no model call, GPU call, or repository edit.

## Authenticated boundary

- Summary SHA-256:
  `14cd0ba20521c4f57eb267becd612794fe4ed6105cd26d8a22a5577cb577209c`
- Authentication receipt SHA-256:
  `45d72fe5aa8c997e472525aafb46cf5479fcc2621e83dff1f7de3f26d459133a`
- Live preflight SHA-256:
  `1877da9976e38f347f96248ab0fe16830710b799be65dc405a47910f51401cb4`
- Implementation lock SHA-256:
  `c1d1c25c3989ad9a748720ee733fc59245c08b7a4e9a88663d52fb5dd460f6e9`
- V2 preoutcome SHA-256:
  `04d8ba59d212adac3193d88c19a38f58298fa18cbdd41321bf9e312bea72fe72`

All nine invocations are exact canonical `COMPLETE` transactions. Each chain
authenticates its prepared request bytes, lock, preflight, runtime, prompts,
stage seeds, generated bundle, `GENERATED` receipt, predecessor completion, EOS
handling, cap flags, token accounting, and requested log probabilities. The
audits reauthenticated all 1,984 rows, 2,304 ranking rows, and 4,032 requested
finite raw-logprob values. Request IDs equal their canonical seed-key hashes;
all 676 stage-one and 676 stage-two seeds are unique and cross-stage disjoint.

Offline analysis recomputed all nine scored JSONL files byte-for-byte and
reproduced the summary. A guarded restart made runner construction and any new
durable write fatal, recovered every completed invocation, reproduced the same
summary, and changed no artifact byte. Stale-schema, wrong-preoutcome,
mixed-source, ambiguous-transaction, and critical-hash mutations failed closed.

## Forced decisions

- Mechanics A: `MECHANICS_INTERFACE_INVALID`
- Mechanics B: `CHEAP_SIBLING_RANKING_FAIL`
- Qualification authorized: `false`
- Top-four secondary authorized: `false`

The generation interface failure is not borderline. The materialized,
name-only, shuffled, echo, and direct arms parsed 12/52, 7/52, 12/52, 20/52,
and 7/24 rows, while their answer caps were contacted 37/52, 42/52, 40/52,
28/52, and 17/24 times. Registered minima were 47/52 suffix parses and 22/24
direct parses; maxima were two and one cap contacts. All 208 suffix thoughts
hit 512 tokens and all 24 direct thoughts hit 1,024 tokens. Every cap-contact
answer was unparsable. Echo succeeded on 20/52 overall and 20/24 non-cap rows,
validating parsing and partial copy compliance, not residual problem solving.

The parse-immune ranking result is a clean negative for the registered cheap
materialized viability score. Its recall@4 was 0.25694, below name-only
0.28125, shuffled 0.32292, listwise 0.27083, and surface 0.37500. It beat the
realized random arm by 0.14931, narrowly below the registered +0.15 margin, but
also missed absolute recall, hit, task-support, and operation-support gates by
large margins. No threshold adjustment can rescue the result.

## Interpretation boundary

The generation stage establishes no positive materialized-residual evidence,
but its invalid ABI prevents a broad mechanism refutation. The descriptive
zero successes among 12 parseable materialized rows are weak negative evidence
only because conditioning on parsing/non-cap contact is outcome-selected. The
ranking stage does cleanly retire this specific cheap behavioral viability
ranker and the top-four branch.

The warranted successor is a separately registered, fresh-identity,
echo-gated answer-seam study. It should select among the current protocol,
short no-think structured emission, and an autonomous commit-slot emission on
known-answer calibration tasks; require at least 90% exact echo/parse and at
most 5% cap contact; freeze the interface; and only then compare materialized,
name-only, and shuffled residual prompts on disjoint mechanics tasks. Raising
the cap, relaxing the parser, tuning ranking thresholds, or opening downstream
splits in this experiment is forbidden.

**Final verdict:** `PASS_FOR_RESULT_PUBLICATION`

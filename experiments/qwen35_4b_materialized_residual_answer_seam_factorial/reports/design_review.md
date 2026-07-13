# Adversarial design review

Status: `HOLD_LIVE_CALLS`; model-free v2 remediation may continue.

Three independent adversaries reviewed committed scaffold `c95a49f1` without
touching repository files, a model, GPU, or any benchmark content:

- scientific falsifiability and causal isolation;
- identity, freshness, contamination, transactions, and recovery; and
- exact Qwen/vLLM chat-template, close/EOS, cap, seed, and slot feasibility.

All three rejected live execution. The scaffold reservation was valid, but it
did not yet define a falsifiable or transaction-safe experiment.

## Findings that changed the design

1. The proposed three policies were an L-shaped design that confounded
   reasoning policy with answer syntax. V2 uses a complete 2x2.
2. The generic runner could not inject `PROGRAM:` after a retained sampled
   thought. V2 adds a token-ID-native continuation primitive, forces the same
   answer seam even after a natural stage-one close, forbids decode/re-tokenize,
   authenticates injected versus sampled tokens, and pairs an explicit answer
   seed domain.
3. The vendored runner was stale relative to the authenticated parent: it
   lacked raw-logprob mode authentication and its fake tokenizer encoded the
   close newline incorrectly. Those defects are repaired and covered by tests.
4. Rate thresholds lacked integer denominators, arity breadth, and a winner
   rule. V2 freezes 44/48, 22/24, 2/48, and 1/24 boundaries plus a fixed
   intervention priority.
5. Hidden-correct proposal coverage was incorrectly described as deployable.
   V2 headlines hidden accuracy of a pre-hidden visible-only selector and
   labels coverage oracle.
6. Matched sampling was unspecified. V2 freezes an independent 96-row direct
   pool per task and mandatory first-over sampled/logical prefixes.
7. A calibration pass could not directly authorize mechanics. V2 requires a
   committed winner receipt, a second lock, a disjoint transport gate, and a
   committed visible-selection receipt before hidden reads.
8. Twenty-four tasks are not confirmatory for modest effects. V2 is explicitly
   a large-effect pilot with exact inference report-only.

## Model-free evidence now present

- Fresh construction completed twice byte-identically.
- 48 calibration and 24 mechanics task IDs have exact frozen namespaces.
- All 72 public-instance fingerprints are unique and have zero overlap with
  264 authenticated parent instances.
- Exact strata are 24 single, 24 double, 12 triple, and 12 quad overall.
- Every alias A-X occurs exactly once in every calibration answer position.
- 4,104 prepared rows contain 2,952 canonical unique request IDs; the three
  suffix arms share exact IDs/order, and parent request-ID, seed-key, and
  user-prompt overlap is zero.
- Construction summary SHA-256:
  `b39e0ad1ccf49503eb48353eac118500432953f32ad27ae2acc1448ed99f622d`.
- Preoutcome receipt SHA-256:
  `a73b5a0a8fa65700a5ddc8e4a4aa7a50355d7e1826ee63d27a0f790a2c8b350e`.
- The experiment suite currently has 61 passing model-free tests.

## Remaining blockers before a calibration lock

1. **Completed model-free:** append-only real-tokenizer receipts `61ff7292...`
   and current-runner receipt `c1a7c6b...` authenticate thinking/no-thinking
   suffixes, close `[248069,271]`,
   `PROGRAM:` `[78041,25]`, model/tokenizer EOS, A-X tokenizations, all 14,400
   canonical lines/tails, context fit, and zero rendered parent overlap.
2. **Core completed model-free; live integration remains:** the append-only
   `STARTED -> bundle -> GENERATED -> COMPLETE` state machine has mutation tests
   for every crash boundary, zero-call recovery, terminal STARTED-only refusal,
   symlinks, unknown/partial inventory, tampering, predecessor chains, and
   incomplete authentication. Completed-prefix reads now revalidate every
   STARTED input hash, prepared ID/order, model/revision, runner metadata, and
   transaction link even against self-consistently rehashed forged receipts.
   The calibration stage runner must still bind the exact lock/preflight and
   pass end-to-end review.
3. **Completed model-free:** the calibration loader has an exact four-artifact
   read allowlist (config, preoutcome, calibration requests, tokenizer receipt).
   A relocated test proves byte-identical plans with mechanics public, audit,
   gold, transport/direct/suffix requests, and every forbidden directory
   absent.
4. **Completed model-free:** the runner
   samples one thought transaction and forks both think512 continuations from
   its exact authenticated retained token IDs. Tests cover natural close/EOS,
   cap termination, JSON persistence, runner/prompt/order/seed drift,
   prefix-only continuation differences, and separate physical versus reused
   token accounting. The calibration stage now freezes and fake-executes one
   48-row batch each in this order: thought, think/freeform, think/slot,
   no-think/freeform, no-think/slot. Full restart performs zero generation
   calls and analysis authenticates exact thought/answer pairing.
5. Add calibration and mechanics implementation locks, live preflight, separate
   authorization receipts, and a hidden-read firewall.
6. Re-run independent implementation review over exact committed hashes.

Verdict: no live model or GPU request is authorized. Proceed only with the
remaining model-free implementation and review.

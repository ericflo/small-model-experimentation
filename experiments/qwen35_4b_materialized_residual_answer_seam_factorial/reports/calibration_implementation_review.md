# Calibration implementation adversarial review

Status: `HOLD_LIVE_CALLS`; remediation is model-free and requires a fresh
independent verdict over the final committed implementation.

The independent review of commit `5b33f01eecc30d925d908acc060f897212b73fda`
found four decisive blockers. This report preserves those findings rather than
silently replacing them with the remediation.

## Findings at reviewed commit

1. The exact-answer parser split on the last `</think>`, allowing no-think
   outputs or double-close tails to discard unregistered text and falsely score
   an exact echo.
2. The calibration lock did not freeze `plans.py`, the mechanics prepared
   pools, mechanics execution/second-lock code, or their tests before the
   interface outcome became known.
3. The live calibration entry point imported local critical modules before
   verifying their hashes and lacked a process-level sealed-path audit hook.
4. Completed transaction authentication was not rebound to the exact frozen
   prepared table, row count, sampling plan, and runner metadata.

The review separately found the token-native shared-thought fork sound: one
sampled thought is persisted once, both thinking arms continue from the exact
retained token IDs, answer seeds are paired, and logical plus physical/reused
sampled-token accounting is correct. It requested explicit physical model-input
totals and tests for compute-prefix plans.

## Model-free remediation under review

- Arm-aware parsing now requires exactly one registered close for thinking
  outputs and none for no-think outputs; multiple closes always fail.
- Every completed prefix and chain is rebound to exact frozen registrations,
  including prepared bytes, row count, lock/preflight/runner hashes, sampling,
  output metadata, completion count, and row metadata.
- Runner receipts expose logical, physical, and reused prompt, sampled, and
  total model tokens. First-over sampled/logical plans have exact-boundary,
  overshoot, exhaustion, deterministic-order, and invalid-cost tests.
- The calibration entry point now has a standard-library pre-import verifier
  and process audit hook. It rejects local bytecode caches and blocks every
  unregistered repository path and all benchmark paths during live calibration.
- The calibration lock inventory now binds the full mechanics implementation
  and tests and records sealed Git blob identities for public/audit/gold data
  and every prepared mechanics pool. Live calibration checks Git tree identity
  without opening sealed mechanics files.
- A separate mechanics lock, transport gate, exact transaction order,
  visible-only selection/resource-plan receipt, and committed-green pre-hidden
  authorization path are implemented before calibration.

These bullets are implementation claims, not a PASS verdict. No calibration
lock may be minted and no model/GPU call may run until a fresh adversarial
review inspects the final committed hashes and explicitly releases this HOLD.

## Second review at `88c3c9e2`

A fresh science adversary found the original four blockers materially repaired
but kept `HOLD_LIVE_CALLS` for seven additional issues:

1. analysis still inferred the thinking parser from mutable output metadata, so
   a self-consistently rewritten no-think row could masquerade as thought;
2. the mechanics lock trusted a shallow, metricless calibration-winner JSON
   instead of recomputing the authenticated calibration analysis;
3. downstream transactions did not durably bind the transport-decision hash;
4. the pre-hidden gate checked shallow receipt fields instead of recomputing the
   complete visible selection;
5. duplicate proposal multiplicity affected the selector's within-cluster tie;
6. `run_mechanics.py --stage lock` imported local mechanics code before checking
   the calibration lock's frozen Git blobs; and
7. the promised report-only exact inference, bootstrap intervals, and exhaustive
   CPU ceiling were absent.

The remediation after that review reconstructs prompt channels and token/text
semantics from registered arm identity, exact-compares calibration and transport
analyses, binds authorization-file hashes into `STARTED` receipts, recomputes
visible selection before hidden authorization, uses hash-only unique-program
ties, bootstraps the mechanics lock stage from the calibration freeze, and adds
deterministic paired inference plus the frozen 13,824-program CPU ceiling.

These are again implementation claims, not a PASS. The HOLD remains until an
independent adversary reviews the final pushed commit and explicitly releases
live calls.

## Third review at `263046c0`

The exact pushed commit and both green workflows were independently verified.
The reviewer confirmed every earlier HOLD item materially repaired, then found
two further decisive gaps:

1. transport and mechanics generation authenticated transaction identity but
   did not reconstruct prompt, seed, token/text, seam, or cost semantics before
   using outputs for a gate, selector, or matched-compute plan; and
2. calibration required seed equality across arms but did not recompute the
   registered numeric seed from run seed, request ID, sample index, and domain.

A synthetic self-consistently receipted transport bundle with unrelated prompt
semantics passed the generic chain and scored 24/24. The committed calibration
fake also demonstrated that arbitrary `2000+index`/`3000+index` seeds passed.
The resulting verdict was `HOLD_RELEASE_LIVE_CALLS`.

The current model-free remediation runs every transport and generation bundle
through tokenizer-based reconstruction before any scoring, cost plan, or
selection. It exact-checks rendered/effective prompts, prompt channel, stable
numeric seeds, seam/prefix/injected tokens, EOS trimming, token/text agreement,
all token counts, generation mode, and complete summary counters. Calibration
now recomputes the same stable seeds. Regressions reject prompt, seed, text, and
cost forgeries, including a rehashed transport chain that generic transaction
authentication alone accepts. This remediation still requires a new exact-hash
review; the HOLD remains.

## Fourth review at `c7bea55d`

Both third-review blockers and both green workflows passed independent audit,
but the reviewer reproduced one last gate-relevant termination gap. Coherent
bundles could exceed the registered 512/24-token caps, and a short stopped
answer could be relabeled `length`; `finish_reason == "length"` directly enters
cap-contact scoring. The exact verdict remained `HOLD_RELEASE_LIVE_CALLS`.

Current remediation binds every stage to its registered cap. A `length` finish
requires exactly the cap and no stop reason/EOS; a `stop` finish requires the
registered model-EOS stop reason and terminal model-EOS token. Values over cap
always fail. Changing `length` to `stop` at exactly the cap cannot change cap
contact because sampled-token count already triggers it. Regression cases now
cover 513/512 thought tokens, 25/24 thinking-answer tokens, 25/24 no-think
tokens, and short `length` rewrites across calibration and mechanics. A fifth
exact-hash review is required; the HOLD remains.

## Fifth review at `f542730b`

The exact pushed commit and both green workflows passed the review preconditions.
The reviewer then constructed a stopped sample whose token sequence contained
the registered model terminator internally and again at the final position.
It passed because authentication required a terminal terminator but did not
forbid an earlier one. The pinned runner registers this token as an explicit
vLLM stop token, so sampling must halt at its first occurrence; tokens after it
are impossible. Accepting the forged sequence could preserve decoded text while
inflating sampled-token and matched-compute accounting. The exact verdict
remained `HOLD_RELEASE_LIVE_CALLS`.

Current remediation requires the registered terminator to occur exactly once,
as the final sampled token, on every `stop` path. Calibration and mechanics
regressions now reject an internal-terminator-plus-final-terminator trace. A
sixth exact-hash review is required; the HOLD remains.

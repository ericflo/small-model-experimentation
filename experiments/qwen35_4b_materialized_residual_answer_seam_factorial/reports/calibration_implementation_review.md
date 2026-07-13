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

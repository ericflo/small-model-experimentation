# Adversarial design review

## Verdict

Proceed after the mitigations below. The design tests a concrete deployability
bottleneck with fixed weights and matched controls; it does not lower the
predecessor's bars after failure.

## 1. Is this post-hoc rescue?

**Attack.** The new answer budget and two-turn metric were chosen after λ=.18
failed the predecessor.

**Mitigation.** The predecessor remains permanently stopped and documented.
This is a new experiment with a new question, artifact root, locality block,
inference budget, metrics, and preregistration. Its transfer seeds were never
opened. The new choices are pinned to machine receipts: 24/24 invalids hit the
answer cap, and 30/30 rejected cases changed within two turns and solved.

## 2. Is more compute being mistaken for a weight capability?

**Attack.** Doubling answer capacity raises per-call reservation from 768 to
1,024 tokens.

**Mitigation.** Base, happy, action, candidate, scaffold, and sample-more all
receive the same per-call budget. Sample-more has the same total call-token
reservation as deep. A candidate gain must therefore be relative under the new
compute regime. Reports must say the harness *elicits* fixed-weight capability,
not that the budget itself installs capability.

## 3. Why keep 512 thinking instead of reallocating a fixed 768?

**Attack.** A 256-thinking/512-answer split would preserve total compute.

**Mitigation.** More than half of valid patch turns at λ=.18 used over 256
thinking tokens, so that reallocation would introduce a second truncation
mechanism while testing the first. Keeping 512 thinking isolates answer-payload
capacity. Matched controls adjudicate the additional compute.

## 4. Could 512 answer tokens still truncate exact-replacement JSON?

**Attack.** Whole-file old/new payloads can exceed 512.

**Mitigation.** This is measured directly with cap-hit rate for every arm and is
a hard retention gate. GPU smoke exercises the exact candidate/tool interface.
If the candidate remains cap-bound, the mechanism fails; the experiment does
not silently raise the cap again.

## 5. Does changed-within-two reward dithering?

**Attack.** Any wasted first turn followed by a patch could pass.

**Mitigation.** The primary metric permits only immediate `PATCH` or
`INSPECT→PATCH`. `INVALID→PATCH`, `VERIFY→PATCH`, and other paths do not qualify.
Immediate rate and the full first-two operator census remain diagnostic, and
invalid actions have an independent hard gate.

## 6. Does an inspection merely exploit more calls than immediate patching?

**Attack.** The new metric may favor a four-step solve over a terse three-step
solve.

**Mitigation.** Every arm has the same six-call limit and token reservation.
Success, sampled tokens, turns, cap hits, verification, and commit are all
reported. The goal is robust recovery in a real loop, where re-reading after a
rejected edit is legitimate evidence acquisition.

## 7. Is calibration reused for yet another selection?

**Attack.** Repeated familiar-family measurement can overfit strategy.

**Mitigation.** There is no model or budget selection: λ=.18 and 512/512 are
fixed before new evaluation. Calibration is only a mechanism/feasibility gate.
Claim-grade evidence begins on the two untouched family blocks.

## 8. Could the candidate win only because controls are infeasible to beat?

**Attack.** Larger answer slots may saturate happy/action or scaffold controls,
making +3pp impossible.

**Mitigation.** Calibration and each transfer block run controls first and write
a hard-range feasibility receipt before candidate evaluation. An impossible
bar stops the experiment without lowering it.

## 9. Is the new locality block truly independent?

**Attack.** Prior integration smoke consumed two contexts from the predecessor's
confirmation file.

**Mitigation.** This experiment commits a third set of 48 entirely new stems and
prefixes. CPU smoke verifies no content-hash overlap with either predecessor
file. The fixed candidate faces it before any new-budget behavior.

## 10. Backend and sampling parity

**Attack.** Same seeds do not make mixed backends or different call geometry
comparable.

**Mitigation.** All behavioral arms use the copied parent vLLM runner, same
engine geometry and model revision. HF Transformers is restricted to the exact
locality measurement. Deep/sample reservations are asserted in CPU smoke.

## 11. Hidden-test and benchmark leakage

**Attack.** Generated hidden code or benchmark items could enter the model or
agent context.

**Mitigation.** The copied firewall serializes public manifests and host
booleans only; hidden executable text/output never enters messages or committed
receipts. No `benchmarks/` file/module is read/imported. Menagerie remains a
public CLI score call after both white-box transfer gates.

## 12. What result would actually matter?

Calibration success with fewer invalids confirms the payload mechanism but is
not enough. The candidate must beat full action, happy, scaffold, and matched
sampling on two fresh family blocks while retaining normal verify/commit loops.
Only a subsequent paired Menagerie improvement supports capability elicitation.

## Required pre-run checks

- Focused tests cover `INSPECT→PATCH`, invalid exclusion, payload telemetry,
  compute matching, and firewall invariants.
- CPU and GPU smokes pass.
- Fresh-locality and model hashes match config.
- `make check` passes; design is rebased and pushed directly to `main`.
- No 512-answer result is observed before that push.

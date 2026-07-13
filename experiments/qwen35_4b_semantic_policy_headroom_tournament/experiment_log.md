# Experiment log

## 2026-07-13 — intake

- Created immediately after the validation-policy curriculum stopped at parent
  48/48 calibration saturation.
- Attached to `agentic_breadth_installation`; no new program or claim.
- Copied the exact procedural repo, loop agent, and vLLM runner into a new
  self-contained experiment; no result-bearing predecessor was modified.

## 2026-07-13 — substrate design

- Added negative, non-integer, and blank-resource conflicts across bundle,
  record, and tuple representations.
- Inferred families state the valid domain but leave malformed-input exception
  behavior to public tests; explicit controls state it verbatim.
- Every partial is correct on copy, ordinary rejection, atomic application,
  and input nonmutation. Only the selected malformed-input behavior differs.
- Frozen two disjoint seeds, a 15–80% replicated eligibility band, two-of-three
  shape support, explicit-control and interface gates, and zero benchmark
  authorization.

## 2026-07-13 — deterministic preflight

- 19/19 tests and CPU smoke pass.
- Headroom A/B each contain 36/36 unique public contents with zero overlap.
- Initial and partial fail visible+hidden; oracle passes visible+hidden for all
  12 family cells.
- No Qwen output or training exists before the design boundary.

## 2026-07-13 — immutable design boundary

- Rebased and pushed the complete design directly to `main` at `391dadc1`.
- Wrote `runs/preregistration_receipt.json` over nine design-critical files;
  it records `model_output_precedes_lock: false` and fails closed on digest or
  ancestry drift.

## 2026-07-13 — GPU smoke

- Exercised all 12 family/state cells through the exact merged parent and vLLM
  looping harness: 22/24 terminal success, 12/12 failed-test success, and 10/12
  rejected-patch success.
- The smoke already showed a 6.96% answer-cap rate, above the later full-run
  gate; this remained diagnostic until the frozen blocks ran.

## 2026-07-13 — full qualification and stop

- Generated both locked 72-case parent blocks and wrote
  `analysis/qualification.json`.
- Formal verdict: `INSTRUMENT_FAIL`. Answer-cap contacts were 43/356 turns
  (12.08%) in A and 46/363 (12.67%) in B versus the frozen 5% ceiling.
  Explicit controls, invalid actions, and content disjointness passed.
- No inferred axis qualified independently. Negative and non-integer
  failed-test recovery were 9/9 in both blocks. Blank was 8/9 and 7/9, with
  only one shape inside the band per block and a different supported shape in
  each.
- The registered exit code 4 is a scientific stop, not a runtime crash. No
  training, checkpoint, benchmark, or Menagerie seed was exposed.

## 2026-07-13 — trajectory forensics and routing lesson

- Of 89 capped turns, 78 contained a valid tool call and 77 then ran on after
  it. All 12 terminal failures contacted the cap, so the frozen interface stop
  cannot be waived even though most cap events remained parseable.
- Every failed-test case reached a fully correct patch; four later regressed.
  Across 54 inferred-contract rejected-patch cases, first-patch correctness was
  0/54, and zero of all 72 rejected trajectories inspected visible tests before
  first patching. Sixty-four eventually became correct after verifier evidence.
- Route the successor earlier: qualify and train ambiguous-state → discriminating
  evidence inspection → evidence-faithful first patch, with complete transition
  replay. Use entropy/varentropy only to mine and stratify ambiguous forks, not
  as correctness labels or token weights.

# Qwen3.5-4B Context-Local Jacobian Clamp Experiment Log

## Scaffold

Created as a new experiment scaffold.

## 2026-07-12 — intake and adversarial design

- Named `qwen35_4b_jacobian_value_transport` as the closest near-duplicate.
- Registered the material correction: selected-token position, direct-concept
  pullbacks, fixed donor coordinate clamps, and exact realized-norm controls.
- Completed the adversarial design review before implementation or scientific
  GPU inference.
- Prohibited target-digit gradients from intervention construction and band
  selection; they are diagnostic-only after confirmation is frozen.
- Registered a full-activation donor site gate before any J conclusion.

## 2026-07-12 — immutable design boundary

- Pushed design commit `c1f06c035404bde62303439daa66dba3c1f026f9` to
  `origin/main` before any result-bearing model call.
- Recorded exact SHA-256 values for the frozen README and preregistration in the
  config and `runs/design_boundary_receipt.json`.

## 2026-07-12 — cache-free model plumbing

- Implemented selected-token discovery, context-local direct-logit pullback
  fitting, fixed full-activation donor patching, fixed coordinate clamping, and
  additive control patching under batch-one `use_cache=False` forwards.
- Added full-rank SVD diagnostics, exact coordinate/idempotence tests, and
  row-wise span-orthogonal norm-control tests.
- Moved `Key:` / `Value: ` into the assistant response prefix so direct concepts
  and bare digits obey the preregistered one-token contracts.
- CPU suite passes 22 tests plus 24 subtests. No model result has been observed.

## 2026-07-12 — model-smoke batch preflight correction

- The first plumbing-only smoke failed solely because equal-length, unpadded
  batch-two clean logits differed from separate batch-one calls by max 0.21875,
  above the descriptive 0.05 tolerance.
- This is the Qwen hybrid batch-equivalence hazard the design intended to detect.
  The frozen scientific path is already batch-one, so batch equivalence is now a
  recorded diagnostic rather than a blocker. Causal antecedent activations were
  exactly suffix-invariant (max difference 0), all token/position contracts
  passed, all three smoke dictionaries had rank 4, and both patch deltas were
  finite/nonzero.
- No target-answer outcome was inspected or used.

## 2026-07-12 — model smoke passed

- Cache-free batch-one plumbing passed on the pinned Qwen3.5-4B revision.
- All 24 concept tokens and all 10 bare digit tokens are single-token; source,
  target, direct, and consequence selected positions agreed at index 62.
- Causal antecedent activations were exactly suffix-invariant at layers 4, 16,
  and 28. Equal-length batch-two top IDs agreed with batch-one even though full
  logits did not, confirming the registered batch-one policy.
- Small context-local dictionaries were full rank at all three smoke layers;
  coordinate and full-donor patch deltas were finite and nonzero.
- Peak allocated GPU memory was 9.68 GB. This was plumbing-only and did not
  inspect target-answer success.

## 2026-07-12 — full context-local lens fit

- Fitted the registered 24 direct-concept pullbacks at every layer 4–28 on all
  48 disjoint lens prompts.
- Every layer passed full effective rank 24 at SVD rtol 1e-5; condition numbers
  ranged from about 3.28 to 6.43. All directions were finite and nonzero.
- The 3,080,243-byte lens has SHA-256
  `e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`.
- Fit time was 27.7 seconds and peak allocated memory was 11.08 GB. Band-selection
  and confirmation outcomes remained unopened.

## 2026-07-12 — donor gate implementation

- Implemented the registered selection-only full-activation donor sweep over six
  five-layer bands under cache-free batch-one inference.
- The code captures source, target, and wrong-donor states under both suffixes,
  enforces equal positions/lengths and causal activation equality at all 25
  fitted layers, and selects the earliest passing band.
- The selection stage contains no J-coordinate intervention and writes
  `j_outcomes_observed: false`; confirmation remains inaccessible unless the
  stored donor gate passes.

## 2026-07-12 — full-activation donor gate passed

- Clean direct and consequence accuracy were both 24/24 with parse rate 1.0.
- Target-donor clamps at bands 4–8, 8–12, 12–16, and 16–20 redirected both the
  direct key and mapped digit on 24/24 items. Wrong donors redirected 24/24 to
  their own key/digit and 0/24 to the registered target.
- Bands 20–24 and 24–28 failed both endpoints, exposing a sharp early causal
  transport window rather than generic activation replacement.
- The frozen earliest-passing rule selected band `[4,5,6,7,8]`. Causal suffix
  invariance remained exact at all 25 layers. Runtime was 44.2 seconds, peak
  allocated memory 8.44 GB, and the stage stored 624 batch-one rows.
- The receipt confirms `j_outcomes_observed: false`; the 48-item confirmation
  split remains unopened.

## 2026-07-12 — untouched confirmation implementation

- Implemented the seven frozen arms at selected band 4–8: baseline, full donor,
  all-24 J clamp, exact norm-matched span-orthogonal random, wrong-donor J,
  source/target pair J, and all-24 concept logit lens.
- Primary J deltas are measured after bf16 application. Random controls are then
  executed, measured, rescaled, and rerun up to 24 times until every item/layer
  is within the registered 1e-5 realized-norm tolerance; any miss yields
  `INVALID_CONTROL`.
- Added paired 10,000-resample confirmation bootstrap, wrong-donor own-digit
  specificity, causal-invariance audit, and a hard assertion that no target
  digit gradient is used. Confirmation has not yet been opened.

## 2026-07-12 — exact bf16 norm-control preflight

- The first norm smoke correctly failed: repeated global rescaling bottomed out
  at relative error 2.64e-5, above the frozen 1e-5 threshold. The failed receipt
  is preserved as `runs/model_smoke/failed_norm_preflight.json`.
- Replaced global rescaling with an in-hook bf16-realized norm search. Each layer
  receives 24 independently generated vectors orthogonal to the full J
  dictionary; the hook selects solely by realized norm error and binary-searches
  its scale using the current residual. No logits or answers enter selection.
- The rerun matched the smoke J delta with relative error exactly 0.0. Requested
  span projection was 5.2e-8; bf16-realized projection was 9.1e-4 and is recorded
  rather than hidden. The untouched confirmation split remains unopened.

## 2026-07-12 — terminal confirmation: invalid control

- Opened the 48-item confirmation split once at frozen band 4–8.
- All-24 J changed direct key and mapped digit on 48/48; pair J changed direct
  48/48 and consequence 47/48; wrong-donor J changed 48/48 to its own key/digit.
  Full donor was 48/48, while logit lens and random were 0/48 target and retained
  the source on 48/48.
- Every scientific endpoint and specificity threshold passed with paired
  J-minus-random bootstrap interval [1.00,1.00].
- One of 96 random rows (`confirm-0046`, consequence) had max per-layer realized
  norm error 1.155e-5, above the frozen 1e-5 limit. The terminal verdict is
  therefore `INVALID_CONTROL`, not `J_TRANSPORT`.
- Requested random vectors were orthogonal to 2.34e-7 maximum projection, but
  bf16-realized deltas reached 5.71% J-span projection. This is recorded as an
  additional replication requirement.
- Runtime was 268.4 seconds, peak allocated memory 8.44 GB, 672 rows. Native
  thinking remains ineligible; next work is a separate fresh control replication.

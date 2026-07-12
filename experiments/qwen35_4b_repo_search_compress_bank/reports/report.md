# Report — repository search-compress-bank curriculum

Status: **negative; primary gate failed and all downstream work stopped.** Menagerie remained sealed.

## Outcome

The registered recipe installed a fast, exact policy for the six training families but damaged the broader repository agent. Compact improved the trained-family block from 40/48 to 48/48 (+16.7 points, paired 95% CI [+6.3, +27.1]), then fell on four wholly held-out algorithm families from 49/72 to 25/72 (−33.3 points, CI [−44.4, −22.2]). Matched-compute sampling reached 38/72, so compact also lost that comparator by 18.1 points (CI [−27.8, −8.3]).

| Frozen block | Apex replay | Compact | Compact delta |
|---|---:|---:|---:|
| Six trained families, deep | 40/48 (0.833) | 48/48 (1.000) | +0.167 |
| Four held-out families, deep | 49/72 (0.681) | 25/72 (0.347) | −0.333 |
| Four held-out families, matched sampling | 38/72 (0.528) | — | compact −0.181 |

Held-out family success exposes the narrow transfer:

| Family | Apex replay | Compact |
|---|---:|---:|
| `dependency_order` | 13/18 | 0/18 |
| `recursive_overlay` | 0/18 | 0/18 |
| `retry_schedule` | 18/18 | 18/18 |
| `ttl_cache` | 18/18 | 7/18 |

Compact verification given success was 0.88 versus 1.00 for apex, invalid actions rose from 0.093 to 0.260 per turn, and submit rate fell from 0.889 to 0.306. Commit given verification was preserved (1.00 versus 0.939), so this is not loss of the terminal submit operator after a genuinely successful verification.

The unrelated-context locality guard also failed: median centered non-target logit drift was 0.386 against the frozen 0.15 ceiling. Mean entropy changed only −0.0094, so entropy did not reveal the collateral that the centered-logit audit caught.

## Training and artifact validity

Both licensed arms completed the exact registered 584 optimizer steps from pinned `Qwen/Qwen3.5-4B` revision `851bf6e8`, at effective batch 16. Apex ran for 10,007 seconds and compact for 9,633 seconds. The exact checkpointed full-vocabulary loss peaked at 48.58 GB in both arms and crossed the former dense-loss OOM path without approximation. Every one of 128 LoRA modules was nonzero and explicitly merged into each composite checkpoint before vLLM evaluation.

The frozen C53 search policy had covered 129/144 fresh training-family repositories and supplied 376 conjunctive visible+private successes. Replay minimization admitted all 129 covered tasks, collapsed every per-file edit to one patch, and produced 516 rows—exactly 129 each for `INSPECT`, `PATCH`, `VERIFY`, and `COMMIT`. Weighted action-token mass was exactly 36,110 per operator and compact-plan mass 3,125.8 per operator. All 129 canonical task traces replayed before training. The negative therefore survives the intended data, weighting, padding, merge, and inference-backend checks.

## Mechanism diagnosis

The bank compressed away the policy states that matter after something goes wrong.

- On trained tasks, compact produced exactly `INSPECT→PATCH→VERIFY→COMMIT` on all 48/48 examples, with zero invalid calls, four turns, and only 402 sampled tokens on average. This is a strong family-specific transducer.
- Among failed-test observations with another turn available, the apex control chose another `PATCH` on 24/26 next transitions (92.3%). Compact chose another `PATCH` on 0/48; it chose `VERIFY` 20 times, `INSPECT` 17 times, and an invalid output 11 times.
- After a passing test, both policies retained commit: apex committed on 64/65 next transitions and compact on 22/22.
- Compact encountered rejected exact patches on all 18 `recursive_overlay` tasks, re-read the file, then repeated the byte-identical rejected patch multiple times on all 18. It had learned the nominal edit shape without learning how to repair an exact-match failure.
- On `ttl_cache`, compact made a plausible partial fix, observed a visible-test failure, then mostly re-tested or re-inspected instead of revising. `retry_schedule`, whose repair matched the single local happy-path pattern, stayed perfect.

This sharpens the prior semantic-operator lesson. The interactive-policy curriculum failed because rare verify/commit operators disappeared. This experiment balanced those operator marginals exactly and preserved commit after pass, yet still failed because marginal counts do not specify conditional transitions. A looping agent needs a contingency policy: failed patch → re-inspect and change the edit; failed test → diagnose and revise; passed test → submit. Success-only minimization retained only the last of those.

The locality failure is consistent with additional shared-weight collateral, but it is not a complete causal attribution. Action-only training was intentionally conditional on the necessary gate and did not run. The candidate also displaced some C54 replay examples under the fixed 584-step budget (1.801 mixed-data epochs versus 2.0 apex epochs). The supported conclusion is that the registered compact plan-plus-action recipe fails, not that compact plans alone are harmful.

## Decision and learned lessons

The primary gate failed 7 of 11 checks: held-out transfer, both paired-bootstrap lower bounds, matched sampling, invalid actions, verification retention, and locality. Trained-family retention, absolute verification, and both commit checks passed. The orchestrator recorded `stop_before_action_only_confirmation_and_menagerie`; action-only, confirmation, and Menagerie did not run, and zero benchmark seeds were consumed.

Do not repeat this success-only one-patch bank at another dose. A defensible successor requires a new experiment and fresh procedural splits, with one of two strategies:

1. Bank a verifier-conditioned state machine rather than a minimal success trace. Preserve rejected patches, failed tests, changed second edits, and recovery transitions; balance conditional transitions such as `failed_test→PATCH`, not only operator totals. Require a frozen perturbation/recovery gate before training.
2. Keep tool-found scaffolds external and retrieve/execute them conditionally, avoiding a broad shared-weight policy update. Compare this against matched compute and the same held-out algorithm-family transfer.

Either strategy must pass locality and family-disjoint recovery before action-only attribution, confirmation, or Menagerie. The consumed `trained_dev` and `transfer_dev` seeds are now analysis data and cannot become future training material or be reused as fresh capability evidence.

The compact machine-readable record is [result_receipt.json](result_receipt.json); paired decisions are in [repo_primary_gate.json](../analysis/repo_primary_gate.json).

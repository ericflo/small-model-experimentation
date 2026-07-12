# Report — repository search-compress-bank curriculum

Status: **pre-training mechanism gate passed; training authorized.** Menagerie remains sealed and no capability result is available yet.

The frozen C53 search policy covered 129/144 fresh training-family repositories (89.6%; weakest family 58.3%) across four trajectories per task. It yielded 376 visible+private-test successes. Replay minimization admitted all 129 covered tasks: seven two-patch traces collapsed to one per-file edit, all 129 canonical traces replayed, and the bank contains exactly 129 targets for each of INSPECT/PATCH/VERIFY/COMMIT.

Exact tokenizer calibration equalized action loss mass at 36,110 per operator and compact-plan mass at 3,125.8 per operator. The longest bank row is 879 tokens versus the frozen 4,096 limit. Compact and action-only have identical contexts, invariant text, actions, and action weights; only compact has plan-span gradient. All pre-training gates pass. See [the compact receipt](harvest_bank_receipt.json).

The first apex training preflight stopped at step 52 when dense cross-entropy on a long batch requested a 9.54 GiB vocabulary temporary; no checkpoint was saved. A 2 × 8 fallback was safe but slow. The final recovery restores the registered 4 × 4 geometry and computes mathematically identical weighted loss in checkpointed 128-position chunks. See [amendments 2](preregistration_amendment_2.md) and [3](preregistration_amendment_3.md).

The final report will add trained-family retention, unseen-family transfer, matched-call sampling, locality, and—only if licensed—paired aggregate Menagerie outcomes.

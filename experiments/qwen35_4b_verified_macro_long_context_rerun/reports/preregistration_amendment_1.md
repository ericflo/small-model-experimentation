# Preregistration amendment 1: deterministic vLLM scheduling

Historical note: amendment 2 later showed that the same prefix effect occurs with asynchronous
scheduling disabled. This amendment is preserved as the decision record that motivated the clean
rerun; its causal attribution is superseded by amendment 2.

Date: 2026-07-09. Frozen after the train-only 8,192-token calibration call failed its token-prefix
audit and before any rerun or scientific prompt.

## Observation

vLLM 0.24 auto-enabled asynchronous scheduling even though the copied runner already disabled V1
multiprocessing for offline reproducibility. The 2,048- and 4,096-token calibration tiers both
force-closed all 64 completions and preserved 64/64 sampled-token prefixes. At 8,192, some of the
first 32 logical sequences terminated before the cap. The remaining two records then entered the
scheduler under a different batching trajectory, and all 32 of their samples diverged from the
4,096-token prefix. The first two records still matched exactly. Full evidence and hashes are in
`analysis/async_scheduler_prefix_audit.json`.

No generated text, parser result, macro correctness, fresh smoke prompt, or full-evaluation prompt
was inspected. The observation uses engine logs, token ids, termination metadata, and record ids
only.

## Repair

Set `async_scheduling=False` explicitly in the single-file vLLM runner and bump its artifact schema
from 2 to 3. Apply the same repair to the repository experiment template. Archive all three
auto-async calibration tiers with their exact old runner and exclude them from budget selection.
Rerun the complete calibration ladder from its first rung under the repaired runner.

The prefix audit remains a hard infrastructure check: a mismatch stops the run. Scientific arms,
prompts, data, seeds, budgets, K values, termination thresholds, and decision rules do not change.
The scheduler repair can change sampled trajectories, so no auto-async row may be pooled with or
substituted into the deterministic rerun.

## Interpretation boundary

This is an inference-protocol failure, not evidence about macro usability or model capability. The
only reusable lesson is operational: in vLLM 0.24, in-process V1 plus explicit sampling seeds is not
sufficient for cross-call prefix reproducibility when asynchronous scheduling remains on.

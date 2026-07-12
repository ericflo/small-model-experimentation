# State-Carry Versus State-Bag Experiment Log

## 2026-07-12 — Intake and Design Freeze

- Attached the new experiment to `structured_execution_and_compilers`; `test_time_reasoning_budget` was rejected as primary because its charter excludes new architectures.
- Named `qwen_fastweight_hook` as the closest negative near-duplicate.
- Reconstructed repository evidence C11–C54 and current recurrence/latent-state literature as a failure map.
- Froze the central contrast: one inherited state versus equal-compute independent reset states.
- Froze Qwen layers 12–19 as two complete hybrid motifs, eight state slots, K=4 training, K=5–12 extrapolation, three seeds, and fail-closed verdicts.

No model was loaded or called.

## 2026-07-12 — Implementation

- Removed the scaffold's vLLM runner because hidden-state intervention requires Transformers and backend mixing would invalidate comparisons.
- Added deterministic random-world substrate, three transition families, three renderings, exact trajectories, structural fingerprints, held-out depth/family/template splits, and matched counterfactual pairs.
- Added a manual pinned-Qwen forward with untouched K=1, recurrence-only loop LoRA, state-only cross-loop communication, Carry/Bag edge switch, state sufficiency heads, fixed-point loss, and optional semantic echo.
- Added model identity/layer/tokenizer/LoRA-locality/parity/gradient gates.
- Added paired pilot/full training, matched-depth evaluation, K curves, edge cuts, donor swaps, explicit textual trace training, compute-matched sample-more, paired bootstrap, and terminal verdict assignment.
- Hardened minimum-depth generation after adversarial review: both full joint-state repeats and earlier occurrences of the terminal queried field are rejected, including in donor-swap pairs.
- Added research handoff, literature map, architecture contract, GPU runbook, agent goal, preregistration, and adversarial design review.

## 2026-07-12 — Local Validation

- CPU smoke: `CPU_SMOKE_PASS`; three families at depths 1/4/8; counterfactual pair has distinct consequences; Carry/Bag compute receipts identical; no benchmark files read.
- Deterministic smoke data build: zero structural cross-split duplicates; exact row counts and hashes.
- Unit suite: 25 tests pass after initial implementation and adversarial hardening.
- Python compilation: every experiment source and script compiles without importing unavailable GPU packages.

No Qwen model was loaded or called. Live model smoke remains the first task on the 48 GiB Ada environment.

## Review Revisions

- Avoided duplicate registration of the PEFT base model inside the recurrence wrapper.
- Removed latent workspace placeholders from the explicit-text comparator.
- Made answer tokens context-prefix-stable rather than assuming standalone tokenization.
- Corrected the explicit-CoT target to close Qwen's think channel before the final answer.
- Isolated analysis by config hash so mixed echo cannot pool with continuous results.
- Reallocated evaluation compute toward full matched-depth and K=4 comparisons; nonprimary K curves are smaller diagnostics.
- Preserved both composite and text-only Qwen3.5 config identifiers while keeping model ID/revision absolute.
- Strengthened counterfactual swaps so paired prompts share world, label mapping, table order, query, and choice order; only the initial state and consequence differ.
- Replaced flat item bootstraps with hierarchical seed-then-task resampling, machine-enforced positive breadth on six of eight depths, and an explicit state-sufficiency verdict gate.
- Made sample-more fail closed on compute overspend and reject truncated thoughts that never naturally reach the answer channel.
- Corrected auxiliary node supervision from an unobservable random generator ID to the node's visible table-row position; counterfactual pairs share that coordinate system.
- Made gzip archives byte-reproducible and removed the counterfactual exception from the global structural-duplicate firewall; the complete default-size corpus now builds cleanly.
- Fixed portable row receipts, pilot/full isolation, primary-cell completeness enforcement, and edge-cut analysis; corrupted rows, datasets, adapters, and loop states now fail hash checks.
- Added initial-value and cumulative training-compute receipts that analysis enforces for every Carry/Bag seed pair.
- Matched the explicit-CoT optimizer schedule and upgraded deployment analysis to a three-seed task-paired hierarchical comparison against oracle `pass@N`, with actual sampled-token and synchronized timing receipts.

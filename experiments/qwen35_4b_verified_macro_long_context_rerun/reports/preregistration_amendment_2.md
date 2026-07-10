# Preregistration amendment 2: cross-budget prefixes are diagnostic only

Date: 2026-07-09. Frozen after the repaired non-async 4,096-token train-only calibration call and
before resuming at 8,192 or rendering any scientific prompt.

## Observation

Amendment 1 correctly separated scheduler modes and conservatively reran from scratch, but its
causal attribution was too strong. With `async_scheduling=False` explicitly recorded, think@2,048
to think@4,096 again produced 32/64 prefix mismatches, exactly on the two records admitted after
the first `max_num_seqs=32` wave. Therefore asynchronous scheduling was not necessary for the
effect. Exact evidence is frozen in `analysis/nonasync_prefix_audit.json`.

vLLM 0.24 has a separate batch-invariant execution mode, but its installed source states that it
requires NVIDIA compute capability at least 9.0. This box's RTX 6000 Ada is capability 8.9. On this
hardware, changing `max_tokens` can change dynamic batch composition and floating-point sampling
trajectories even with the same prompts, explicit seeds, in-process V1, and synchronous scheduling.

## Repair

Cross-budget prefix equality is removed as a pass/fail condition and retained as a recorded
diagnostic. The original preregistered selector already uses only per-tier termination metadata:
cap rate, answer-truncation rate, p99 thinking use, and p99 answer use. Those quantities do not
require common random numbers, and each tier remains fully frozen and reproducible as its own
protocol. Lower tiers remain excluded from scoring and are never pooled.

The repaired synchronous runner, runner schema, all completed non-async rows, budget ladder,
thresholds, prompts, seeds, K values, and scientific decision rules remain unchanged. The run
resumes at think@8,192 without regenerating the valid non-async 2,048/4,096 tiers.

## Interpretation boundary

Neither prefix failure is model evidence. It is a limitation on paired *cross-budget* randomness.
Within each accepted scientific matrix, every arm still uses the same frozen budget and inference
protocol; inference arms were never preregistered to share token-level random trajectories.

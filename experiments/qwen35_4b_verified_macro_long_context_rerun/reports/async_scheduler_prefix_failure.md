# Auto-async vLLM calibration failure

Historical note: the non-async replication reproduced the same second-wave prefix divergence.
Amendment 2 therefore supersedes this report's async-specific causal attribution; the underlying
limitation is lack of batch-invariant execution on the Ada GPU.

The first long-context calibration attempt was excluded before science because vLLM 0.24 silently
auto-enabled asynchronous scheduling. Fixed seeds did not make later queued requests independent
of earlier requests' termination lengths.

At think@2,048 and think@4,096, every completion hit the cap and all 64 longer-tier prefixes
matched. At think@8,192, 52 completions ended before the cap; the first 32 scheduled samples kept
their prior prefixes, while the other 32—exactly the two records admitted later—diverged, often
within the first few hundred tokens. This scheduler-dependent RNG path would make a sample depend
on batch neighbors and invalidate the wrapper's offline reproducibility contract.

The artifacts live under
`/workspace/large_artifacts/qwen35_4b_verified_macro_long_context_rerun/diagnostic_archives/async_scheduler_calibration/`;
their deterministic tree checksum and inspection-only status are recorded in
`reports/artifact_manifest.yaml`. The repair explicitly passes `async_scheduling=False`, records it
in runner metadata, and reruns calibration from scratch. No fresh smoke or full task was exposed in
this failed attempt.

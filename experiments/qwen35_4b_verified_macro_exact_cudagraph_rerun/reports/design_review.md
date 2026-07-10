# Adversarial design review: exact CUDA-graph verified-macro rerun

Status: **implementation review and independent prelaunch review complete. The reviewer gave GO
for the registered 49k K=4 probe before engine construction.**

## 1. Calling `max_cudagraph_capture_size=19` does not prove a graph at 19

Pinned vLLM generates a sparse default list. The predecessor's maximum-only request can therefore
resolve below the requested active width.

**Resolution:** pass explicit lists ending at 19 or 15. Read the resolved compilation config after
construction, record it, and abort on any difference. Also require `decode_mode=FULL` and
`has_full_cudagraphs=true`; vLLM may otherwise retain sizes while downgrading dispatch to eager or
piecewise-only execution. CPU tests reject truncated lists plus resolved `NONE` and piecewise-only
modes.

## 2. A custom list could exceed scheduler or GDN limits

A capture maximum larger than `max_num_seqs` would add memory and could revive the cache-line
assertion the shared runner originally avoided.

**Resolution:** `EngineConfig.validate()` requires the largest capture size to equal
`max_num_seqs`. The frozen lists contain no larger width.

## 3. CUDA-graph coverage does not prove KV-cache fit

Compilation geometry and sequence-cache capacity are independent. Fixing one can leave scheduler
preemption/recomputation in place.

**Resolution:** retain the live block-rounded KV check and require both audits in every preflight,
receipt verification, catalog state, and test fixture.

## 4. More graph coverage is not guaranteed to be faster

Warmup, capture memory, padding, GDN behavior, or termination mix can erase the expected gain.

**Resolution:** state a falsifiable throughput hypothesis, preserve a tie/slowdown, record load and
generation time separately, and avoid calling the cross-protocol delta a causal benchmark.

## 5. Scheduler changes can change samples on Ada

Explicit seeds do not give batch-invariant common random numbers on compute capability 8.9.

**Resolution:** fresh samples, fresh external root, exact prompt-order binding, no row reuse or
pairwise token claims, and same backend/config within each selectable K=12 matrix.

## 6. Reusing the 49k negative would adaptively skip a changed protocol

The old 49k result was produced with implicit capture geometry. Treating it as this experiment's
lower rung would violate first-adequate selection.

**Resolution:** run a fresh exact-geometry K=4 probe at 49k. Only its registered rejection can
authorize the separate 61k probe.

## 7. K=4 could leak into K=12 evidence

Probe output has lower compute and a different batch population.

**Resolution:** distinct namespaces, roles, K checks, receipt geometry, and tests make probe
promotion impossible.

## 8. Throughput tuning could become outcome shopping

Inspecting decoded answers before choosing geometry or budget would make infrastructure tuning an
accuracy hyperparameter.

**Resolution:** this protocol is frozen before GPU work. Rung choice is termination-only and the
analyzer remains three-pass: receipts/history, token-only termination, then semantics.

## 9. A resolved metadata field could be fabricated or drift from preflight

Recording only requested engine arguments would miss vLLM normalization; recording resolved data
in one place would permit cross-file disagreement.

**Resolution:** runner metadata records resolved sizes and semantic graph modes, preflight records
requested and resolved geometry, receipts bind both files, and validators recompute and cross-check
the identities.

## 10. A partial or predecessor directory could be resumed accidentally

That would pool protocols or make an interrupted call look complete.

**Resolution:** new fail-closed namespace, both predecessor roots forbidden, fixed external lock,
preflight-only as the only resumable state, receipt written last, and checksum catalog.

## 11. A positive smoke could be overclaimed

The designed library is a ceiling and there is no matched-compute sample-more baseline here.

**Resolution:** describe every outcome as inference/termination/interface evidence. Any macro
capability claim requires a separate contamination-controlled matched-compute experiment.

## Review verdict

The implementation has the required model/backend lock, exact list and full-decode-mode checks,
independent live-KV gate, fresh artifact boundary, one-engine phases, K4 nonpromotion, content-blind
selection, receipt-last storage, and CPU regression tests. Launch only after model-free validation
passes at the frozen commit and a separate read-only reviewer confirms GO.

## 2026-07-10 — independent prelaunch disposition

A separate read-only reviewer completed the required prelaunch audit against frozen protocol
binding `9d2692c6acad35d3b7ab56ddf368c9974c1ddaf6e0a06997b01015c0de397158` before the first engine was
constructed. The reviewer independently confirmed the sole-model/backend lock, explicit capture
lists and resolved full-decode assertions, live block-rounded KV-capacity check, fresh and forbidden
artifact roots, one-engine/one-phase execution, receipt-last commits, K4 nonpromotion,
content-blind branch rule, and stop conditions. Model-free validation was already green.

Verdict: **GO for only the fresh 49,152-token K=4 probe.** This disposition did not authorize
automatic advancement, any K=12 arm, decoded-output inspection, scoring, or a macro claim.

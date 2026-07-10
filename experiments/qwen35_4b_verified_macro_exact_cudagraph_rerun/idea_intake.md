# Idea intake: verified-macro exact CUDA-graph vLLM rerun

## Program fit

- Primary program: `operator_and_skill_inventories`.
- Secondary connections: `structured_execution_and_compilers` and
  `test_time_reasoning_budget`.
- Closest near-duplicate: `qwen35_4b_verified_macro_capacity_fit_rerun`.
- Related-work query: `make related QUERY="verified macro exact CUDA graph capture geometry long context vLLM capacity fit"`.
- Prior anchors:
  `qwen35_4b_verified_macro_capacity_fit_rerun` (live-KV-safe concurrency),
  `qwen35_4b_verified_macro_long_context_rerun` (long-context termination and overcommit
  diagnosis), and `qwen35_4b_verified_macro_invention` (original stopped interface prototype).

This belongs in a new experiment because CUDA-graph capture geometry is part of the inference
protocol on Ada. Results from the near-duplicate cannot be patched in place, pooled, or treated as
common-random-number continuations.

## Unresolved uncertainty

The capacity-fit follow-up corrected KV-cache overcommit by setting `max_num_seqs=19` at think@49k
and 15 at think@61k. Its runner also set `max_cudagraph_capture_size=max_num_seqs`, but left the
capture-size list implicit. In pinned vLLM 0.24 the default list uses 1, 2, 4 and then multiples of
8. The requested maxima therefore resolve to 16 and 8, leaving steady-state active widths 17--19
or 9--15 outside captured CUDA graphs. The 49k capacity-fit probe was slower than the earlier
overcommitted diagnostic, so this compilation geometry is a plausible contributor.

No semantic conclusion follows from that throughput observation. Lower concurrency, termination
mix, recomputation, graph coverage, and Ada batch-shape sensitivity remain entangled.

## Novelty claim

No completed verified-macro experiment has simultaneously required live block-rounded KV fit and
an explicit CUDA-graph list whose resolved maximum exactly equals each capacity-fit active batch
width.

## Mechanism and falsifier

Explicit lists `[1,2,4,8,16,19]` at think@49k and `[1,2,4,8,15]` at think@61k should keep every
decode width through the active maximum eligible for graph dispatch, rather than silently falling
back above 16 or 8. The runner must read vLLM's resolved compilation config after engine creation
and abort before generation if either list or maximum differs, or if the resolved mode does not
provide full decode CUDA graphs.

The throughput explanation is weakened if a fresh exact-geometry probe remains no faster than its
same-budget capacity-fit predecessor. The scientific macro question remains unresolved unless a
fresh same-protocol K=12 base/designed matrix clears the content-blind termination gate.

## Controls

- Only `Qwen/Qwen3.5-4B` at the pinned revision and only vLLM inference.
- Exact copied tasks, demonstrations, libraries, prompts, sampling law, context, and KV-fit rule.
- Explicit capture lists and their resolved values are recorded in metadata, preflight, receipt,
  and protocol binding.
- Fresh K=4 probes and fresh K=12 arms in a new external namespace; no predecessor output imported.
- 49k and 61k are separate one-engine invocations. A rejected rung cannot contribute rows to the
  next rung.
- Termination selection remains content-blind. Semantic access remains impossible until a complete
  same-rung K=12 matrix passes.

## Evidence output

- Model-free validation of exact source hashes, capture mappings, and state-machine invariants.
- Receipt-bound live KV and resolved CUDA-graph geometry for each invocation.
- Aggregate sampled-token throughput and content-blind termination audit for each completed probe.
- Semantic smoke output only if the existing preregistered eligibility gate clears.

## Decision

Prepare and freeze the experiment now. Do not launch a GPU process until the current GPU owner has
coordinated the handoff and an independent design audit gives GO.

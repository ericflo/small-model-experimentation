# Qwen3.5-4B verified-macro exact CUDA-graph vLLM rerun report

## Summary

**Termination result:** both fresh exact-capture K=4 probes passed their infrastructure gates but
failed all three registered content-blind termination thresholds. The 61k rung is terminal:
selection records `pass=false` with no selected budget. No K=12 arm, decoded or scored inspection,
semantic analysis, or macro result is eligible.

The experiment isolates a concrete inference defect: the capacity-fit predecessor requested CUDA
graph maxima 19 and 15 but relied on vLLM's implicit sparse shape list, which resolved only through
16 and 8. This follow-up supplies explicit lists ending at the active widths and fails unless the
constructed engine resolves them exactly under a full-decode CUDA-graph mode.

## Research program fit

The scientific target remains reusable composite operators, so the primary program is
`operator_and_skill_inventories`. This variant repairs the inference protocol needed to make the
verified-macro smoke interpretable; it is not a new macro mechanism.

## Method

- Only pinned `Qwen/Qwen3.5-4B` through vLLM.
- Frozen 12-task smoke-v2 inputs and identical sampling/interface rules.
- Live KV-safe max-seqs 19 at 49k and 15 at 61k.
- Explicit CUDA-graph lists `[1,2,4,8,16,19]` and `[1,2,4,8,15]`.
- Post-construction equality check against vLLM's resolved compilation config, including
  `decode_mode=FULL` and full CUDA graphs enabled.
- Fresh K4 termination probe at each reached rung; no probe is semantic evidence.
- Fresh same-rung base/designed K12 matrix only after a passing probe.
- New fail-closed external namespace, receipt-last writes, and content-blind selection.

## Results

### Infrastructure gates

The independent reviewer gave prelaunch GO against frozen binding
`9d2692c6acad35d3b7ab56ddf368c9974c1ddaf6e0a06997b01015c0de397158` before engine construction.
The exact 49k invocation then passed both live checks:

- 996,864 live KV tokens with 528-token cache blocks;
- 963,072 block-rounded tokens required by max-seqs 19, leaving 33,792 tokens;
- resolved `decode_mode=FULL`, full CUDA graphs enabled; and
- requested and resolved capture sizes exactly `[1,2,4,8,16,19]`.

The receipt SHA-256 is
`61da6f616365bf080e97f341bd0c2305b889998c4d161c61e63d06e5dfb5923c`. The complete external
four-file tree is 30,653,162 bytes with SHA-256
`654d44119fc46fe83428c154680ee502073c00022ccb9cdb0922c1896ca20685`.

After the content-blind 49k rejection, the separately fresh 61k invocation passed the same gates:

- 997,888 live KV tokens with 528-token cache blocks;
- 950,400 block-rounded tokens required by max-seqs 15, leaving 47,488 tokens; and
- requested and resolved full-decode capture sizes exactly `[1,2,4,8,15]`.

Its receipt SHA-256 is
`8f00535a773e347ec4f90a48eb6b00960935d7010c3500af7ccdf57fedd6f2e1`. The final external
eight-file tree is 70,191,578 bytes with SHA-256
`4aa311ab579c301f2b2d7383591e3e68ed66035d184bbaaa2659d59bd95542d3`.

### Termination result

All 48 samples ended at the frozen reasoning boundary and required force-close. The content-blind
token-ID audit found 38 exact periodic loops, with periods recorded in
`analysis/scientific_smoke_49k_termination_audit.json`; the remaining 10 contacts were unresolved.
Six answer stages reached the 512-token limit. Loop (79.17%), unresolved (20.83%), and answer-limit
(12.50%) rates each fail the registered threshold. Therefore 49k was rejected before decoding or
scoring.

At 61k, all 48 fresh samples likewise ended at the reasoning boundary and required force-close.
The audit found 40 exact periodic loops, eight unresolved cap contacts, and four answer-limit
contacts: rates of 83.33%, 16.67%, and 8.33%. All three fail their thresholds. The terminal
selection SHA-256 is
`acbaf7cdb84ee5633e4f86b0716360c382f2262eb50e79ca92ce02b6e157fb07`; it records `pass=false`
and `selected_thinking_budget=null`, so no K=12 matrix or semantic analysis is eligible.

### Operational throughput

The run sampled 2,363,163 tokens in 4,809.081014 generation seconds, or 491.395964 sampled tokens/s.
Late JIT warnings are included in this elapsed time. Exact active-width capture was 4.1636% faster
than the otherwise closest implicit-capture capacity-fit probe at 471.753824 tokens/s, consistent
with the operational diagnosis that uncaptured widths 17--19 imposed avoidable overhead. It remained
16.2114% slower than the invalidly overcommitted max-seqs-64 diagnostic at 586.471154 tokens/s.

These are descriptive cross-protocol comparisons, not a clean causal benchmark: scheduler geometry
can change trajectories, and sampled-token throughput does not count recomputed prefix work.

The exact-capture 61k probe sampled 2,951,995 tokens in 7,422.885983 generation seconds, or
397.688312 sampled tokens/s, after 100.293580 seconds of model loading. Its lower aggregate rate is
also descriptive: the budget, active width, and generated trajectories differ from 49k.

The experiment-local CPU suite had already passed 42/42 tests and `scripts/run.py --validate` with
both frozen record hashes and capture mappings. No decoded, semantic, oracle, or deployable metric
is available.

## Controls

The runner rejects an explicit list whose maximum differs from `max_num_seqs`. The live preflight
rejects vLLM normalization/truncation, eager or piecewise-only decode, lack of active-width coverage,
and insufficient block-rounded KV capacity. Storage validators bind both requested and resolved
geometry into preflight, metadata, receipt, and catalog. Predecessor output and external roots are
ineligible.

## Interpretation

Exact graph coverage improved 49k aggregate sampled-token throughput modestly relative to the
closest implicit-capture run, so active-width capture matters, but it did not recover the
overcommitted diagnostic's speed. The stronger scientific obstacle is now terminal for this ladder:
simply increasing the reasoning allowance from 49k to 61k left every probe sample at the boundary,
mostly in exact periodic loops. This is a useful negative about the setup, not a negative macro
result, because the protocol never reached a termination-adequate K12 matrix.

No outcome in this experiment by itself establishes verified-macro capability. A complete
termination-adequate K12 matrix is required even for semantic smoke, and a separate matched-compute
experiment is required for a capability claim.

## Next action

Stop this ladder and preserve both rejected probes. Do not decode either probe or launch a K12 arm.
Any future attempt to address periodic reasoning loops must be preregistered as a separate design
variant rather than extending this result-bearing experiment.

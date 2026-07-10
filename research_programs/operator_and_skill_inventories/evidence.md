# Evidence

## Seed Experiments

- [qwen35_4b_operator_inventory_search_pilot](../../experiments/qwen35_4b_operator_inventory_search_pilot/reports/qwen35_4b_operator_inventory_search_pilot_report.md)
- [qwen35_4b_operator_inventory_scaling_stress](../../experiments/qwen35_4b_operator_inventory_scaling_stress/reports/qwen35_4b_operator_inventory_scaling_stress_report.md)
- [qwen35_4b_inventory_shortlister_training](../../experiments/qwen35_4b_inventory_shortlister_training/README.md)
- [qwen35_4b_joint_shortlister_ladder](../../experiments/qwen35_4b_joint_shortlister_ladder/reports/qwen35_4b_joint_shortlister_ladder_report.md)

## Stopped Prototype and Corrected Interface

- [qwen35_4b_verified_macro_invention](../../experiments/qwen35_4b_verified_macro_invention/reports/report.md)

The verified-macro experiment stopped before its fresh induction smoke and full comparison. Smoke
v1 used an under-sized 192-token thinking budget: all 1,440 solver samples force-closed and 607
answer stages truncated, so it measured an unusable generation interface rather than macro quality.

Task-independent, plan-given probes then separated surface use from induction. With budgeted
thinking, all 16 samples again force-closed and 12/16 spilled into the answer cap. With thinking
off, truncation disappeared and 16/16 outputs were syntactically valid and macro-using, but only
3/16 samples were exact and only 1/4 supplied plans had even one exact optimal transcription. All
13 failures over-aliased beyond depth five. Those probes established only that the low-compute
interfaces failed; they did not establish a model-level alias-placement limit.

The independent long-context follow-up
[`qwen35_4b_verified_macro_long_context_rerun`](../../experiments/qwen35_4b_verified_macro_long_context_rerun/)
corrected that diagnosis before scoring induction. Under vLLM with think@16,384, the disjoint
plan-given gate passed 16/16 records: 63/64 samples were strict valid macro-using rewrites, all 12
cap contacts were classified by the frozen exact-token periodic-tail detector, no cap remained
unresolved, and no answer truncated. At the registered K=4 record-level gate, adequately budgeted
reasoning therefore made the existing free-form alias interface usable when the primitive plan was
supplied; this is not a per-sample-perfect or induction claim. The fresh induction workload is much
harder to provision. Its 16,384-token base arm had 131/144 unresolved contacts, 13/144 exact loops,
and 60/144 answer-limit contacts. At 32,768 tokens, 63/144 remained unresolved, 81/144 were exact
loops, and 37/144 reached the answer limit. Both rungs were excluded before any decoded output or
score was inspected. The later max-seqs-64 K=4 probe at think@49,152 also force-closed all 48
samples: 34 exact loops, 14 unresolved contacts, and 13 answer-limit contacts over 2,366,620 sampled
tokens in 4,035.356 seconds (586.47 tokens/s). Amendment 12 had already made it diagnostic-only
before its receipt: its 48 admitted block-rounded contexts could demand 2,433,024 cache tokens from
the measured 995,328-token cache. No decoded output or score was inspected.

The independent capacity-fit follow-up
[`qwen35_4b_verified_macro_capacity_fit_rerun`](../../experiments/qwen35_4b_verified_macro_capacity_fit_rerun/)
completed a fresh base-only K=4 probe at think@49,152 with Qwen/Qwen3.5-4B on vLLM. Its live
997,888-token cache, 528-token blocks, max-seqs 19, and 50,688-token rounded worst sequence gave
963,072 tokens of demand and 34,816 of headroom. Despite valid capacity geometry, all 48 samples
contacted the reasoning boundary: 37 were exact token-ID loops, 11 remained unresolved, and 9
answers hit the limit. The rung was rejected before decoded or scored content was inspected, and a
fresh K=4 think@61,440 attempt began at max-seqs 15. It was stopped before a receipt after a
source/runtime audit showed that the runner's implicit CUDA-graph list covered only through width 8,
not the active width 15; it left no reusable rows. The strict capacity-fit 49k run generated
2,364,643 sampled tokens in 5,012.451 seconds (471.754 tokens/s), 19.6% slower than the predecessor's
max-seqs-64 diagnostic (586.471 tokens/s).

The separate exact-capture follow-up
[`qwen35_4b_verified_macro_exact_cudagraph_rerun`](../../experiments/qwen35_4b_verified_macro_exact_cudagraph_rerun/)
then froze explicit capture lists and fresh artifacts. Its 49k preflight fit 963,072 required cache
tokens into 996,864 live tokens and vLLM resolved `[1, 2, 4, 8, 16, 19]` exactly. The fresh K=4
probe still failed all three content-blind termination thresholds: all 48 samples contacted the
boundary, 38 were exact loops, 10 remained unresolved, and 6 answers hit the limit. It generated
2,363,163 sampled tokens in 4,809.081 seconds (491.396 tokens/s), descriptively 4.16% faster than
the closest implicit-capture capacity-fit probe. The terminal 61k probe also passed both runtime
gates: its live audit fit 950,400 required tokens into 997,888 with 47,488 headroom, and vLLM
resolved FULL decode graphs at `[1, 2, 4, 8, 15]` exactly. It still failed every termination
threshold: all 48 samples contacted the boundary, 40 were exact loops, 8 remained unresolved, and
4 answers hit the limit. It generated 2,951,995 sampled tokens in 7,422.886 seconds (397.688
tokens/s). The terminal selector records `pass=false` and no selected budget. Cache-safe
concurrency and active-width graph coverage are both part of a valid high-throughput envelope, but
neither guarantees termination. This remains provisioning/termination evidence: no K=12 arm,
semantic analysis, or mined-versus-random-versus-designed capability result was authorized, and no
decoded or scored content informed escalation.

## Current Read

Inventory search can recover held-out targets when the right primitives exist. The next bottleneck
is scalable shortlisting and disambiguation. Composite aliases are no longer blocked at the
registered plan-given K=4 record-level gate. The open question is now the intended one—whether a
mined inventory helps *induce* fresh programs beyond matched-compute base sampling, literal hints,
random inventories, and a designed ceiling—but simply increasing context is no longer the right
way to reach it. The exact-capture ladder terminated without selecting a budget, with exact loops
dominating both long rungs. The next bridge should be a separately preregistered symmetric
loop-control protocol that preserves the unresolved-contact and answer-limit gates before any
semantic comparison is exposed.

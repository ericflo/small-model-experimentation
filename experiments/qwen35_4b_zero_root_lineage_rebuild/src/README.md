# Source — intentionally empty (gateway-only measurement rule)

This experiment deliberately has NO experiment-local model EVAL engine.
The scaffold's `src/vllm_runner.py` (and its test) was removed, not left
to rot: every MEASUREMENT in this cell goes through the trusted
aggregate gateway `scripts/run_benchmark_aggregate.py` at the repo root,
which is sha256-pinned by the design receipt, the seed-consuming runner,
and the harness. Nothing under `benchmarks/` is ever read from this
experiment — only gateway receipts.

The only model code this cell runs itself is the byte-identical COPIED
lineage package (`scripts/lineage_trainers/`, `scripts/merge_adapter.py`
— HF training and merging, never evaluation), replayed by
`scripts/rebuild_zero_root.py` under the frozen recorded recipe.

If you are tempted to add eval engine code here, stop: a local engine
would bypass the benchmark firewall (receipts-only, no benchmark data
read) and break the implementation-signature anchoring to the reference
events (all three receipts must carry the pinned discovery/confirmation
benchmark-implementation signature, fail closed).

# Experiment-local payload harness

This directory freezes the procedural repositories, public tool loop, firewall,
and pinned vLLM backend copied from the predecessor. The only semantic harness
addition is explicit two-turn rejected-patch transition telemetry; inference
budget changes live in `configs/default.yaml`.

No module reads/imports `benchmarks/`. Hidden generated tests remain host-side
and only booleans enter evaluation receipts.

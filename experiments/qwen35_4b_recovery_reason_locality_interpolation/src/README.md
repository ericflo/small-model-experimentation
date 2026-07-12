# Experiment-local harness

This directory freezes the parent procedural repository generator, public
tool-loop harness, firewall assertions, and pinned vLLM runner. They are copied
rather than imported from a result-bearing predecessor so later edits cannot
change this experiment retroactively.

No module reads or imports `benchmarks/`. Hidden generated tests remain
host-side; only final booleans enter serialized evaluation receipts.

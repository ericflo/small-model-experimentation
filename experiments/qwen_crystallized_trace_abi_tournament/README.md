# Qwen Crystallized Trace ABI Tournament

**Status:** finished

This standalone experiment tests whether dense executable traces help a local
4B model learn compact crystallized procedures better than answer-only
supervision, and whether the output representation itself is a load-bearing
choice.

The experiment generates deterministic tasks from several practical families,
trains small QLoRA adapters for multiple output ABIs, evaluates held-out
examples, and writes Markdown and HTML reports with charts.

Large adapter checkpoints are stored outside this directory under:

`/workspace/large_artifacts/qwen_crystallized_trace_abi_tournament`


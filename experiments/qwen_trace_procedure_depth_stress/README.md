# Qwen Trace Procedure Depth Stress

This standalone experiment tests whether a local 4B model can compile natural
language tasks into explicit executable procedures over a fixed stack ABI.

The primary metric for procedure arms is external execution of the generated
procedure. The model's emitted final answer is measured separately and is not
trusted as the procedure score.

Large adapter checkpoints are stored outside this experiment directory:

```text
/workspace/large_artifacts/qwen_trace_procedure_depth_stress
```


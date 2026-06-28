# Qwen Iterative Repair Policy

Standalone experiment for an iterative hidden-program repair policy attached to
a frozen Qwen numeric compiler. The policy receives the current compiled program
trace, predicts sparse slot edits, and is evaluated over multiple repair
iterations.

Small outputs live in this experiment directory. Large checkpoints live under:

```text
large_artifacts/qwen_iterative_repair_policy/checkpoints/
```


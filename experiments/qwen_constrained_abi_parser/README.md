# Qwen Constrained ABI Parser

**Status:** finished

This standalone experiment tests whether a finite-state stack-ABI decoder and
a canonical parse stage improve a local 4B model's reliability as a compiler
from natural language into executable procedures.

The headline metric is external execution accuracy. Valid-program rate is
tracked, but validity alone is not a success criterion.

Large adapter checkpoints are stored outside this experiment directory:

```text
/workspace/large_artifacts/qwen_constrained_abi_parser
```


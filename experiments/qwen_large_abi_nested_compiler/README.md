# Qwen Large ABI Nested Compiler

**Status:** finished

Standalone experiment testing whether a constrained stack-ABI compiler still works when the primitive library becomes large and when tasks require nested sub-procedures rather than only linear chains.

## Question

The compiler can compose a small known ABI into long linear procedures. This experiment tests two new bottlenecks before scaling to a real crystallized-skill corpus:

- Operation selection at larger ABI size: does moving from 32 to 128 unary operations degrade linear chain compilation?
- Nested structure: does adding shallow nested supervision let the model compile branch/sub-procedure tasks at larger held-out nesting widths?

## Training Targets

- `abi32_chain_d3`: 32 unary operations, chain tasks only, depths 1 to 3.
- `abi128_chain_d3`: 128 unary operations, chain tasks only, depths 1 to 3.
- `abi32_nested_d3`: 32 unary operations, chain depths 1 to 3 plus nested tasks with 2 to 3 branches.
- `abi128_nested_d3`: 128 unary operations, chain depths 1 to 3 plus nested tasks with 2 to 3 branches.

## Evaluation

- Chain depth sweep: depths 3, 8, and 16.
- Template-shifted chain endpoint: depth 16.
- Nested branch sweep: 2, 3, 4, and 8 branches.
- Template-shifted nested endpoint: 8 branches.
- Decoder arms: free greedy stack generation and finite-state constrained stack generation.
- Gold ABI sanity arm: execute the reference program through the interpreter.

## Primary Metrics

- Constrained external execution accuracy.
- Correct-given-valid accuracy, since constrained decoding should keep validity near 100%.
- Failure taxonomy on chain depth 16 and nested 8-branch outputs.
- Free versus constrained execution to separate syntax effects from semantic operation/structure selection.

## Artifacts

- Source: `src/qwen_large_abi_nested_compiler.py`
- Metrics and details: `analysis/`
- Reports: `reports/`
- Large checkpoints: `/workspace/large_artifacts/qwen_large_abi_nested_compiler/checkpoints`

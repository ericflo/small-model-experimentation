# Qwen Extrapolation-Bound ABI

**Status:** finished

Standalone Track 1 experiment measuring how far a constrained stack-ABI compiler extrapolates past its maximum supervised composition depth.

## Question

The compiler can learn composed procedures, but a large ABI corpus needs a practical curriculum rule. This experiment tests whether training up to depth 3 is enough for depth 12 and 16, or whether the curriculum must include deeper composed procedures such as depth 6 or 8.

## Curricula

- `atomic_d1`: one-operation tasks only.
- `mix_d1_d2_d3`: depths 1, 2, and 3.
- `mix_d1_to_d6`: depths 1, 2, 3, 4, and 6.
- `mix_d1_to_d8`: depths 1, 2, 3, 4, 6, and 8.

## Evaluation

- Standard depth sweep: 1, 3, 6, 8, 12, and 16.
- Wording-shift sweep: 8, 12, and 16.
- Decoder arms: free greedy stack generation and finite-state constrained stack generation.
- Gold ABI sanity arm: execute the reference program through the interpreter.

## Primary Metrics

- Constrained external execution accuracy at depths 12 and 16.
- Correct-given-valid accuracy, since constrained decoding should keep validity near 100%.
- Failure taxonomy on depth-12/depth-16 constrained outputs.
- Free versus constrained execution to separate syntax/format effects from semantic composition effects.

## Artifacts

- Source: `src/qwen_extrapolation_bound_abi.py`
- Metrics and details: `analysis/`
- Reports: `reports/`
- Large checkpoints: `/workspace/large_artifacts/qwen_extrapolation_bound_abi/checkpoints`

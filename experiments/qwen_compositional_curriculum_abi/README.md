# Qwen Compositional Curriculum ABI

Standalone Track 1 experiment testing whether shallow composed-procedure supervision improves a Qwen stack-ABI compiler.

## Question

Constrained ABI decoding can enforce valid syntax, but the remaining hard errors are valid programs that execute to the wrong result. This experiment tests whether those errors shrink when training includes a small amount of composed procedures instead of only atomic one-operation tasks.

## Arms

- `atomic_d1`: train only on one-operation procedures.
- `mix_d1_d2`: train on a balanced mix of one- and two-operation procedures.
- `mix_d1_d2_d3`: train on a balanced mix of one-, two-, and three-operation procedures.

Each trained adapter is evaluated with:

- `program_stack_free`: greedy raw ABI generation.
- `program_stack_constrained`: finite-state constrained ABI decoding.
- `program_stack_resample_valid`: retry free decoding until the ABI parses, within a fixed attempt budget.
- `gold_abi_constrained`: interpreter sanity check using the gold ABI.

## Primary Metrics

- External execution accuracy at held-out depths 4, 6, and 8.
- Correct-given-valid accuracy, to distinguish semantic gains from syntax-only gains.
- Template-shift execution accuracy at depths 6 and 8.
- Failure taxonomy over valid/wrong, invalid, wrong operation, and wrong argument cases.

## Artifacts

- Source: `src/qwen_compositional_curriculum_abi.py`
- Metrics and details: `analysis/`
- Reports: `reports/`
- Large checkpoints: `/workspace/large_artifacts/qwen_compositional_curriculum_abi/checkpoints`

# Qwen Real Task ABI Coverage Gate

Standalone experiment testing whether a fixed office-data ABI covers real-style deterministic tasks that were not generated from the ABI.

## Question

The compiler recipe only matters for real capability if realistic tasks decompose into a reusable operation library. This experiment freezes a general-purpose office ABI first, then evaluates oracle coverage on hand-curated deterministic tasks from contact cleanup, dates, money, URLs/files, product codes, addresses, and small tables.

## Method

- Define ABI primitives before task definitions in `src/qwen_real_task_abi_coverage_gate.py`.
- Write task references as ordinary Python functions over examples, not as ABI programs.
- Use an enumerative oracle synthesizer to search ABI expressions from visible fields and constants.
- Score candidates on train examples and held-out examples for each task.
- Report coverage by ABI tier, split, family, search depth, and failure mode.

## Primary Metrics

- `heldout_covered`: a candidate matches both train and held-out examples.
- `train_match_only`: a candidate fits train examples but fails held-out examples.
- `no_train_match`: the ABI/search could not even fit the visible examples.
- Coverage by split and task family under the strongest fixed ABI.

## Artifacts

- Source: `src/qwen_real_task_abi_coverage_gate.py`
- Metrics/details: `analysis/`
- Reports: `reports/`
- Large artifact directory: `/workspace/large_artifacts/qwen_real_task_abi_coverage_gate`

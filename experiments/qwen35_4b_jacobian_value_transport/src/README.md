# Source layout

- `task_data.py`: fresh procedural split generation and exact verifiers.
- `jacobian.py`: averaged/targeted Jacobian estimators, sparse coordinate reads,
  coordinate swaps, and norm-matched controls.
- `model_ops.py`: pinned Qwen3.5-4B loading, residual hooks, prefix continuation,
  and exact forward-token accounting.
- `stats.py`: within-task value metrics and paired bootstrap gates.
- `io_utils.py`: atomic artifact writes and immutable design receipts.

The implementation is experiment-local. It follows the equations from the 2026
Jacobian-lens paper but defines source/target position weighting explicitly and tests
the orientation on a tiny differentiable decoder before any Qwen run.

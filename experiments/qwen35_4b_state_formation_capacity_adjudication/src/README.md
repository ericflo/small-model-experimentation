# Source Layout

- `config.py`: exact model/config validation and the versioned source/test contract.
- `design_boundary.py`: clean-at-HEAD scientific-design freeze and implementation-GO gate.
- `substrate.py`: fresh deterministic finite-state tasks and exact trajectory verification.
- `data_pipeline.py`: six disjoint splits, tiered sealed validation, and contrast access ledger.
- `initialization.py`: per-seed common-state bundles plus external and tracked receipts.
- `adaptation.py`: common LoRA/direct-full-shape extra-call hooks and deterministic dropout.
- `state_loop_model.py`: the single carried-state recurrence, dense heads, and answer-graph control.
- `optimizer_receipts.py`: complete finite FP32 Adam-state auditing.
- `mechanics.py`: compute accounting and deterministic crossed bootstrap helpers.
- `gpu_runner.py`: setup gates, fixed-final training, durable checkpoints, and intact/disabled
  evaluation.
- `analysis.py`: exact evaluation validation, sequential branch authorization, sealed contrasts,
  and terminal taxonomy.

There is one model path: the pinned `Qwen/Qwen3.5-4B` Transformers revision. PEFT is instantiated
only as a small deterministic tensor reference inside LoRA G0; it is not another model or result
backend. The executable experiment has no alternate recurrence arm, swap, edge-cut, benchmark,
sample-more, or intermediate-checkpoint stage.

Normal execution uses `scripts/run.py`, which enforces canonical output paths. Every model-bearing or
verdict stage requires the exact confirmatory config, an implementation review with exact `GO`, and
the canonical scientific-design receipt. Follow `docs/gpu_runbook.md`; conditional stages must
consume their exact upstream authorization rather than a manually interpreted score.

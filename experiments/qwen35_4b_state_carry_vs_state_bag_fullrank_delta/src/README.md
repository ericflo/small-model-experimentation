# Source Layout

- `config.py`: frozen full-rank, parent-trigger, seed, gate, and source contracts.
- `substrate.py`: unchanged deterministic finite-world task and paired swaps.
- `data_pipeline.py`: split generation plus canonical parent-row parity receipt.
- `mechanics.py`: CPU recurrence references, compute accounting, and bootstrap.
- `state_loop_model.py`: manual Qwen recurrence and direct FP32 delta hooks.
- `gpu_runner.py`: trigger-gated G0, training, checkpointing, evaluation, swaps.
- `analysis.py`: parent-equivalent pilot/full causal ladder with G4 deferred.

There is no fallback model and no PEFT execution path. Use only the root pinned
environment and `Qwen/Qwen3.5-4B`. Model-bearing stages require the exact frozen
confirmatory config.


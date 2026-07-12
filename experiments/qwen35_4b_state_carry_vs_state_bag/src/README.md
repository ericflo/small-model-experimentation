# Source Layout

- `config.py`: pinned contracts and config hashing.
- `substrate.py`: exact procedural worlds and counterfactual pairs.
- `data_pipeline.py`: split construction and manifests.
- `mechanics.py`: CPU reference semantics, compute accounting, bootstrap.
- `state_loop_model.py`: manual Qwen recurrent-middle-block forward.
- `gpu_runner.py`: live smoke, training, evaluation, swaps, text comparator.
- `analysis.py`: paired metrics and terminal verdict ladder.

GPU modules intentionally have no fallback model. Use the root pinned training environment and only `Qwen/Qwen3.5-4B`.

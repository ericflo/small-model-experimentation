# Large Artifacts Manifest

No large artifacts are required for the initial coverage probe.

If later iterations add model adapters or checkpoints, they should be stored under:

`/workspace/large_artifacts/qwen35_4b_sketch_coverage_shift_probe`

Final audit:

- Experiment directory size: `4.1M`.
- Large artifact directory size: `0`.
- No files larger than `50M` are stored inside the experiment directory.
- No model adapters or checkpoints were produced in this run.

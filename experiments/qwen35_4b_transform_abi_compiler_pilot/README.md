# qwen35_4b_transform_abi_compiler_pilot

**Status:** finished

Standalone compiler pilot for deterministic transformation tasks.

The experiment tests whether Qwen3.5-4B can choose executable ABI programs for a frozen transformation library. It uses constrained candidate scoring rather than free-form JSON decoding: candidate ABI programs are enumerated, the model scores each candidate under the task prompt, and the selected program is executed by a deterministic interpreter.

The headline metrics separate:

- depth-1 operation selection
- depth-2/3 composition
- raw example accuracy
- counterexample-filtered accuracy

Large model adapters are written under `/workspace/large_artifacts/qwen35_4b_transform_abi_compiler_pilot/`.

## Reproduce

```bash
python scripts/build_data.py
python scripts/eval_constrained.py --arm base
python scripts/train_lora.py
python scripts/eval_constrained.py --arm lora --adapter /workspace/large_artifacts/qwen35_4b_transform_abi_compiler_pilot/lora
python scripts/make_report.py
```

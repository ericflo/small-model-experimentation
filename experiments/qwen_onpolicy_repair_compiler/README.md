# Qwen On-Policy Repair-to-Compiler

This experiment tests whether verified local program repairs can be distilled into the Qwen-attached compiler policy itself.

The compiler emits an executable modular-arithmetic program from a prompt. For each on-policy compiler output, the training loop enumerates nearby edits, keeps locally verified repaired programs, and fine-tunes the same QLoRA/compiler head toward those targets. The central measurement is whether the deployable compiler improves on fresh prompts, not whether target-aware repair search has headroom.

## Layout

- `src/qwen_onpolicy_repair_compiler_experiment.py`: training and evaluation entrypoint.
- `src/qwen_onpolicy_repair_compiler_core.py`: compiler, executor, task generation, and local verifier utilities.
- `analysis/analyze_qwen_onpolicy_repair_compiler.py`: aggregates runs, writes figures, markdown, HTML, and summary files.
- `runs/`: per-run CSV and JSON outputs.
- `reports/`: standalone writeups.
- `checkpoint_manifest.csv`: generated manifest for large checkpoint artifacts.

Large model artifacts are stored under:

```text
large_artifacts/qwen_onpolicy_repair_compiler/checkpoints/
```

## Primary Question

Can target-aware local repair headroom become a policy-weight improvement after one or more on-policy fine-tuning rounds?

The strongest positive signal is a fresh paired compiler accuracy gain after training, with a remaining local repair ceiling that explains the available headroom.

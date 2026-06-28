# Qwen Fast-Weight Hook Experiment

This experiment tests whether a frozen Qwen3.5-4B model gains accuracy from an inserted invisible recurrent fast-weight runtime.

## Contents

- `src/latent_qwen_fastweight_experiment.py`: training and evaluation script.
- `src/analyze_latent_results.py`: regenerates analysis CSVs and figures from run metadata.
- `reports/latent_fastweight_qwen_paper.md`: standalone paper-style report.
- `reports/latent_fastweight_qwen_paper.html`: HTML version of the report.
- `reports/experiment_log.md`: chronological run log.
- `analysis/`: generated figures, summary Markdown, and analysis CSVs.
- `runs/`: small JSON run outputs. Checkpoint `.pt` files are not stored here.
- `checkpoint_manifest.csv`: list of saved checkpoints stored outside this directory.

## Large Files

Adapter checkpoints are stored at:

```text
../../large_artifacts/qwen_fastweight_hook/checkpoints/
```

Download that directory only if you need to load saved adapter weights. The experiment reports and analysis figures do not require it.

## Useful Commands

Regenerate analysis outputs from the stored run metadata:

```bash
python experiments/qwen_fastweight_hook/src/analyze_latent_results.py
```

Run a new experiment from this experiment directory or from the workspace root, passing an explicit `--output_dir` if you want a named run.


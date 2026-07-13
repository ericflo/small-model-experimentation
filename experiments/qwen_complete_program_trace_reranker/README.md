# Qwen Complete-Program Trace Reranker

**Status:** finished

This standalone experiment tests whether a learned verifier can better select
hidden-VM candidate traces when it receives prompt-conditioned Qwen hidden-state
context in addition to candidate execution features.

Large checkpoints are stored outside this directory:

- `/workspace/large_artifacts/qwen_complete_program_trace_reranker/checkpoints`

Local structure:

- `src/qwen_complete_program_trace_reranker_experiment.py`: run generator, training, and evaluation
- `src/analyze_qwen_complete_program_trace_reranker.py`: regenerate aggregate CSVs, figures, reports, and checkpoint manifest
- `runs/`: per-run logs and metrics
- `analysis/`: aggregate metrics and figures
- `reports/`: standalone Markdown and HTML report
- `checkpoint_manifest.csv`: checkpoint and artifact index

Regenerate analysis artifacts from completed runs:

```bash
python experiments/qwen_complete_program_trace_reranker/src/analyze_qwen_complete_program_trace_reranker.py
```

# Qwen Mixed-Domain Trace Verifier

This standalone experiment tests whether a learned candidate-trace verifier can
select better hidden-VM programs from a frozen Qwen-attached mixed-domain
compiler. The compiler proposes an executable VM trace; the verifier reranks
local candidate traces using only prompt/compiler/execution-derived features at
test time.

Large checkpoints are stored outside this directory:

- `/workspace/large_artifacts/qwen_mixed_domain_trace_verifier/checkpoints`

Expected local structure:

- `src/`: experiment and analysis code
- `runs/`: per-run metrics and logs
- `analysis/`: aggregate CSVs, summaries, and figures
- `reports/`: standalone Markdown and HTML reports
- `checkpoint_manifest.csv`: checkpoint/artifact index


# Adaptive Cognitive Kernel

**Status:** finished

This experiment tests whether a task-conditioned recurrent runtime with temporary adapter-style weight edits can learn compositional execution better than ordinary fixed-weight controllers.

The experiment is standalone. Small artifacts live in this directory. Large checkpoints live under:

```text
/workspace/large_artifacts/adaptive_cognitive_kernel/checkpoints/
```

## Layout

```text
src/       runner and report generation code
runs/      run metrics and logs
analysis/  aggregate CSVs and figures
reports/   Markdown and HTML reports
```

## Main Question

Can a prompt-conditioned generator select temporary computation kernels that run recurrently and generalize to longer or held-out operation compositions?

## Controls

- Direct transformer over the full prompt.
- Fixed recurrent controller with ordinary op-token inputs.
- ACK runtime with dynamic low-rank weight edits disabled.
- ACK shuffled-code evaluation, which preserves runtime capacity while breaking ordered task-conditioned computation.

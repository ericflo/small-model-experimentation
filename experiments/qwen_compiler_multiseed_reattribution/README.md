# Qwen Compiler Multi-Seed Reattribution

**Status:** finished

This standalone experiment tests whether a one-shot executable latent compiler reliably learns length-24 modular programs across random seeds.

The result-bearing arms are:

- `max24_curriculum`: a max-24 compiler trained from the start with staged train lengths.
- `expand_copy`: an 8 -> 16 -> 24 expanding compiler where new slots copy the last learned slot.
- `max24_no_curriculum`: a max-24 compiler trained on the full length range immediately.

All result-bearing runs use the same seed set across arms. The main report aggregates final length-24 executable accuracy with mean, standard deviation, min, and max across seeds.

Large checkpoints are stored outside this directory:

`/workspace/large_artifacts/qwen_compiler_multiseed_reattribution/checkpoints`

Primary reports:

- `reports/qwen_compiler_multiseed_reattribution_report.md`
- `reports/qwen_compiler_multiseed_reattribution_report.html`


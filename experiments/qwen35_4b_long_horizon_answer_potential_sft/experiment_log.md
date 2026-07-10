# Qwen3.5-4B Long-Horizon Answer-Potential SFT Experiment Log

## Scaffold

Created as a new experiment after C51 showed real signal under a 99.37% force-close regime but stopped
before SFT. The follow-up plan removes the 512-token cutoff, scales to about 95k candidates, retains full
natural traces, and commits to the complete training matrix without an effectiveness gate.

No scientific GPU work had run at this boundary. No benchmark content was read.

## 2026-07-10 — Pre-GPU Implementation Boundary

- Design committed at `6f97f0ce`; config now verifies the committed README, preregistration, and design
  review hashes rather than the mutable worktree.
- Generated all seven fresh procedural splits and passed ID/prompt/digest/family-seed disjointness.
- Added atomic gzip shard storage, natural-thought continuation, sampled-trace prior capture, a
  memory-bounded Transformers scorer using `logits_to_keep`, and restartable generation/scoring stages.
- Twenty-six CPU tests pass. No scientific GPU call has run yet.

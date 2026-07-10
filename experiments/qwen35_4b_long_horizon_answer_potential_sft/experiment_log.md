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

## 2026-07-10 — Integrated Smoke and Pre-Pilot Correction

- The integrated four-trace smoke passed HF/vLLM canonical-answer likelihood parity at `0.000735`
  nats per answer token and preserved finite sampled-trace priors.
- Two of four traces closed naturally between 3,171 and 3,665 tokens. The other two reached the
  smoke-only 4,096-token allowance. This directly confirms useful behavior far beyond 512 tokens.
- The first smoke also exposed an implementation mismatch before the termination pilot: a legacy global
  trigram-frequency heuristic marked all long coherent prose as loops. The preregistered exclusion is for
  *exact periodic loops*, so the detector now requires an exact periodic suffix of at least four repeats
  and 64 tokens. Global trigram frequency remains descriptive only. This correction changes neither a
  scientific outcome nor any selection threshold, and is frozen before the registered pilot.
- Transformers' causal-LM auto mapping passed Qwen3.5's multimodal wrapper config to its text-only class
  and failed on the missing outer `vocab_size`. The scorer now uses the checkpoint-native conditional
  generation class in text-only mode; a real 4,096-token forward and the integrated parity gate passed.

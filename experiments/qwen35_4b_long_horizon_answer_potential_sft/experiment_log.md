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

## 2026-07-11 — Termination Pilot and Training Envelope

- Registered pilot complete: 108 traces, 108/108 longer than 512 tokens, median 4,636, p95/max 14,336,
  96/108 natural closes, zero exact periodic loops, and 13 initial allowance contacts. The one
  non-loomfix contact closed after continuation; all 12 residual open traces were loomfix and remain
  mechanically ineligible rather than force-closed. Correctness was not inspected.
- Built an isolated pinned training environment. Both Qwen hybrid fast-path checks pass. Exact text-only
  loading maps all 426 language weights and exposes 42,467,328 rank-32 LoRA parameters.
- The first checkpointed 3--4k loss implementation was unnecessarily slow (96.5 s/two rows). Bounded
  bf16 full logits preserved the same loss and reduced it to 4.2 s at 17.7 GiB peak.
- The required 14,687-token stress row exposed a quadratic SDPA backward allocation (12.86 GiB) and
  OOMed. Training now uses xFormers memory-efficient causal attention, explicit per-layer checkpoints
  plus 256-token recomputed vocabulary chunks only above 8,192 tokens. The exact untruncated 14,687-token
  row passed in 29.1 s at 15.0 GiB peak; the ordinary path remained 4.7 s/two rows. These are operational
  kernel/memory repairs made before any SFT dataset exists, not result-conditioned design changes.

## 2026-07-11 — Calibration Harvest and Scorer Parity

- The preregistered 32-row HF/vLLM canonical-answer likelihood parity gate passed. Maximum absolute
  difference was 0.000448 nats per answer token against the frozen 0.15 threshold.
- Completed all 135 calibration tasks at N=64: 8,640 traces and 45,728,102 sampled thought tokens.
  There were 7,814 natural closes, 27 exact periodic loops, and finite sampled-trace priors on all 8,640
  rows. Loops and unresolved allowance contacts remain ineligible for scoring and selection.
- Loomfix was the clear outlier: 204/960 natural closes (21.3%) and 12,676,528 sampled tokens. Its harder
  tiers frequently remained open after the exact 12,288+2,048 protocol. This is recorded as a support
  boundary, not used to abort the experiment, and no incomplete trace is force-closed into training.
- Generation was restart-safe at per-task atomic/checksummed shard boundaries. Calibration answer
  rollouts and full-prefix scoring began only after the full harvest completed; no score/outcome was
  inspected during generation.

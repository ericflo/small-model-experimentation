# Qwen3.5-4B Native-Thought Seam Budget Ladder Experiment Log

## 2026-07-12 — Intake and design

- Created as the required separate successor to the parent's terminal
  `NO_NATURAL_SEAM` at 160 thought tokens.
- Ran related-work search and named the closest duplicate plus three anchors.
- Froze a paired 256/512/1024 selection ladder and untouched single-cap
  confirmation; no forced close or failed-confirmation fallback is allowed.
- Completed the adversarial review before any model call.
- CPU smoke generated 40/40 unique fresh task fingerprints with zero overlap
  against both scientific parents and proved all terminal gates reachable.
- No scientific outcome has been opened.

## 2026-07-12 — Outcome-blind model smoke

- Ran only after the reviewed design and executable hash anchor were pushed.
- Loaded pinned Qwen3.5-4B (32 layers, width 2560) under Transformers 5.13.0,
  torch 2.11.0+cu129, bf16 SDPA on an RTX 6000 Ada.
- Verified exact think/open/EOS IDs and 12 unique one-token aliases.
- Audited forward input lengths `[472, 1, 1, 1, 1, 1, 1, 1]`; cached decoding
  is active rather than a silently repeated full-prefix path.
- Eight diagnostic tokens were sampled; no answer correctness was computed or
  recorded. Scientific selection remains unopened.

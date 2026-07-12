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

## 2026-07-12 — Budget selection terminal

- Opened the 48-trace selection once under the frozen 1,024 maximum rung.
- Completed all 48 rows under one model instance; all passed the audited cache
  contract and all stopped as `think_cap_without_close` at exactly 1,024 tokens.
- Nested metrics were zero natural close, parse, and usable traces at 256, 512,
  and 1,024. Decision: `NO_BUDGET_SELECTED`.
- Sampled 49,152 tokens/forwards in 1,618.080 seconds. Rows hash:
  `17e3b107154079ecd857af45544c92c2e11b13cd495edfeb6eb24dcf97f5d39c`.
- Confirmation is ineligible and was not opened; no cap or threshold changed.
- Post-decision diagnostics found no exact 1--32-token periodicity across any
  256-token tail. This does not turn cap-bound reasoning into a natural seam.
- Next branch: a separate forced-commit protocol whose injected close is an
  explicit deployed action, not evidence of autonomous termination.

# Mid-Density Token-Matched Universal Curriculum Experiment Log

## 2026-07-13 — intake, feasibility revision, and design freeze

- Created as the result-separated successor to the exact-token 0/40/80 local
  negative; the closest near-duplicate is named in `idea_intake.md`.
- Copied the authenticated 800-row designed and 2,240-row replay sources. No
  benchmark content was read or copied.
- Rejected the initially proposed 320-row arm before freeze: proportional designed
  rows were collectively shorter than the shortest row-matched replay selection.
- Froze a representative 0/160/240 ladder with a 1,280-row common replay core and
  three all-skill 80-row blocks. Each designed/replay block matches forward tokens
  exactly at 33,613, 34,091, and 33,015.
- Materialized three position-aligned 1,520-row arms. Each has exactly 1,405,510
  forward tokens, zero skipped rows, and 190 optimizer steps.
- Froze training seed 43, local seed 88,005, and conditional aggregate-only
  quick@1,024 seed 78,135 before training.
- Adversarial design review passed after the 320-row feasibility revision. No model
  training or new evaluation event has run.

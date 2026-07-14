# Search-Scaffold Universal Curriculum Experiment Log

## 2026-07-14 — Intake and scaffold

- Ran repository related-work discovery before creation.
- Named the closest near-duplicates: the completed universal curriculum, mid-density
  exact-token ladder, and close-weight successor.
- Chose a different mechanism: independently scored decomposition-search substates
  followed by a bounded compact ledger.
- Reserved construction seed 77,111, training seed 45, fresh local seed 88,007, and
  conditional aggregate seed 78,137.
- No data materialization, model access, training, evaluation, merge, or benchmark
  event has run.

Next: publish the intake checkpoint, then implement and truth-audit the smallest
runnable staged-search stream before adversarial review.

## 2026-07-14 — Feasibility and design freeze

- Implemented an experiment-local executable operation universe and 80 deterministic
  lessons: 16 each of apply, fit, reject, execute, and search across colors, digits,
  letters, nonce strings, Romans, and syllables.
- Independently recomputed every operation, fitting second, unique fitting pair,
  dead branch, intermediate state, and final answer in tests. The reject stage is
  balanced 8 `FIT` / 8 `NO_FIT`.
- Reused the predecessor's authenticated 200-row replay core and 120-row control
  partition. Selected a disjoint 40-row candidate filler with an exact token sum.
- Froze replay SHA-256 `c157fb13...355d` and candidate SHA-256
  `79a8d7c9...0b90`: 320 rows and 286,814 forward tokens each, zero skips, max
  sequence 2,991, 40 updates, and exactly 200 byte-identical shuffled positions.
- Preserved the non-equivalent target allocation: replay has 116,036 prompt,
  167,411 thought, 640 close, and 2,727 answer tokens; candidate has 124,245 prompt,
  158,311 thought, 640 close, and 3,618 answer tokens.
- Removed the predecessor's target-specific close-weight interface. Both arms use
  ordinary thought/close weight 0.2; unit tests lock span behavior.
- Froze wrappers for authenticated same-parent training, local seed 88,007, a sole
  candidate promotion, explicit merges, and one aggregate-only quick@1,024 paired
  event at seed 78,137.
- Completed adversarial review with a narrowed claim boundary: the full target shows
  one dead and one true branch, not exhaustive search. All 43 experiment tests and
  the staged smoke harness pass.
- No GPU model load, training, local generation, merge, or benchmark event ran.

Next: commit, rebase, run the full repository check, push this design freeze to
`main`, verify both workflows, and only then train the replay control.

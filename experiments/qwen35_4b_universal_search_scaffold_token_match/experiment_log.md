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

## 2026-07-14 — Replay control training

- Began only after design-freeze commit `603b8107` was pushed to `main` and both
  Validate Repository and Publish Research Site completed successfully.
- Authenticated the `close_xi` warm start and exact replay stream/token receipt.
- Trained `replay_after_close` for the frozen 320 rows, one epoch, 40 optimizer
  steps, seed 45, and ordinary thought/close weights 0.2. All 320 rows encoded and
  zero skipped.
- Completed in 281.2 wall seconds with final train loss 0.4215.
- Preserved receipt/log hashes `5b293eb6...5a66` / `7d3bc262...d5f7`.
  External adapter weights/config hashes are `10155232...fc538` /
  `373c1426...ac9b`; weight size is 169,903,320 bytes.
- No candidate training, local generation, merge, or benchmark event ran.

Next: publish and CI-verify this control checkpoint, then train the frozen candidate.

## 2026-07-14 — Scaffold candidate training

- Began only after control commit `b8f11db6` was pushed to `main` and both GitHub
  workflows completed successfully.
- Independently restarted from the authenticated `close_xi` parent; did not continue
  from or inspect capability behavior of the replay control.
- Trained `scaffold_after_close` for the frozen 320 rows, one epoch, 40 optimizer
  steps, seed 45, and ordinary thought/close weights 0.2. All 320 rows encoded and
  zero skipped.
- Completed in 291.4 wall seconds with final train loss 1.492. This loss is not
  compared causally with replay loss because target composition differs.
- Preserved receipt/log hashes `13ba8897...6dd0` / `ccaffa7b...99c1`.
  External adapter weights/config hashes are `e7957d90...84618` /
  `22859c76...2c4ce`; weight size is 169,903,320 bytes.
- No local generation, merge, or benchmark event ran.

Next: publish and CI-verify this candidate checkpoint, then consume the single frozen
local seed 88,007 over parent, replay control, and candidate together.

## 2026-07-14 — Fresh local negative

- Began only after candidate commit `9e34c675` was pushed to `main` and both GitHub
  workflows completed successfully.
- Consumed the single registered experiment-owned seed 88,007 over parent, active
  replay, and scaffold in one greedy Transformers process at a 1,024-token cap.
- Parent scored 18/26 correct, 23/26 parsed, and three caps; replay scored 16/26,
  23/26, and three; scaffold scored 16/26, 23/26, and three.
- Scaffold was 0/2 execute, 0/2 induct, and 0/2 probe. It failed accuracy, parse,
  cap, execute, and induction checks; route abstention alone passed. Promotion is
  empty and the harness stopped nonzero as registered.
- Preserved local/gate/promotion receipts. Full local receipt SHA-256 is
  `156acd37...acdb`; promotion SHA-256 is `7e1fd417...f1c1`.
- Post-decision paired forensics show 2 wins/4 losses versus parent and 3/3 versus
  replay. Candidate mean output grew to 520.5 tokens, and both execute failures
  computed the correct state before running to cap without a parsed answer.
- No merge ran. Aggregate seed 78,137 remains sealed and no benchmark data was read.

Next: publish this completed negative, then create a fresh result-separated successor
for natural-language variable-depth state execution and hypothesis scoring.

# Hygiene-Explore De-stack Experiment Log

## 2026-07-15 — Model-free design freeze

- Opened after the v2 kill-rule closure recorded third-dose interference on
  the axis adapter lineage. This trial de-stacks: only the two lessons with
  replicated installs (hygiene 4/4 kind wins; explore most), at their proven
  per-kind dose, on the CLEAN designed_fresh lineage at dose two — the
  interference law's safe region.
- Corpus: 80 rows from the byte-identical v2 generator at seed 77,119.
- Gate: two-kind holdout where BOTH installs must strictly win, plus the
  standard retention screen; unconditional recovery flags preregistered; the
  escalation rule (no further dose permutations if recovery fails) is frozen.
- Conditional pilot at the MEDIUM tier, sealed seed 78,148.
- No model, GPU, training, local, or benchmark event has run.

## 2026-07-15 — Model-free pipeline run (freeze → measure → materialize → validate → design → gate)

- Adapted the full pipeline from the v2 staged-repair predecessor
  (build/measure/materialize/validate/train/merge/gate/eval/benchmark/harness),
  retargeted to the `designed_fresh` clean parent and the two de-stacked
  kinds; every fail-closed convention kept (hash pins, `--check` byte-identity,
  TODO-PIN fail-closed, encoder binding, merge self-pin in the gate receipt,
  shared `finalize_promotion` writer, full benchmark CLI, weight
  recomputation). `gen_axis_v2.py`, `gen_curriculum.py`, `train_think.py`,
  `src/vllm_runner.py` copied byte-identical from the predecessor.
- Froze the de-stacked corpus at construction seed 77,119:
  `data/sft_hygiene_explore.jsonl` sha256
  `8b3e97919c62cbb0893add281dc1d3ae881aa0138d0d1721043fec26b0c22cf1`
  (80 rows; hygiene 40 / explore 40; balance 27/40 injected, 17 co-located),
  manifest `cbc9ae6d132d09b9bac2eb43010c2f3eb051993493d6ea659950ac07f0a1e903`;
  replay blend copied byte-identically (`25a9595f…abf0c2`) from the parent
  experiment.
- Measured exact spans (`source_token_lengths.json`
  `f67b916688cf8cdcd182963e43883433157445285d3abab7b9c1991b290a50ea`;
  treatment vector forward 19,582 / nonzero 5,793 / mass×5 9,665); the
  three-axis MILP solved optimally in 0.65 s: both 240-row variable blocks at
  forward 139,986 / nonzero 58,961 / mass×5 67,773; arm totals 1,367,212 /
  574,619 / 629,207; 1,280 position-aligned shared rows; zero skips. Streams:
  `replay_clean.jsonl` `2189d160…ce0017`, `hygiene_explore.jsonl`
  `82aa1a78…4112f`; manifest `16b5c1a8…5baac`; independent validation receipt
  `stream_token_receipt.json`
  `f74988c3647f206b1b379ea482bcf9803cb1916d0b45a3a446fa3c80f199da12`.
- Froze the design receipt (`data/design_receipt.json`
  `5319f208cd63c3482d3b81b2e291619418a3bde6dfbc311cdce9c5a084113982`) binding
  the parent identity (tracked merge receipt `ab3f20cc…6acc2`, tree
  `93433aa2…255`, weights `0a3b89cd…979`, adapter `36f41095…442` /
  `5966461b…055`) and the lifecycle substring contracts.
- Froze the 124-task gate at seed 88,018: `local_tasks_seed88018.jsonl`
  `597f10a44674cc12e5f499be8de6804bb040985019b18aacc5527339a26857eb`,
  runner input `2d58da21…12565`, design receipt `c4952ca9…9b442`; zero
  canonical-message overlap proven against both frozen corpora, both
  materialized streams, regenerated construction rows, prior local seeds
  88,000–88,017, and all five predecessor frozen gates (88,013–88,017).
- Filled the stream pins (materialize/validate/train_trial exposure constants);
  left fail-closed as TODO-PIN: `PUBLISHED_ARM_HASHES` (both arms, filled
  after each training stage publishes) and `EXPECTED_TREE_SHA256` (both arms,
  filled after merge-arms publishes); verified each aborts with the pin
  message.
- `run.py --smoke` green end to end (every `--check` byte-identical, 50 unit
  tests, py_compile); training remains sealed behind the pushed-checkpoint
  gates and the PASS_CONTROL_TRAINING / PASS_CONTROL_MERGE verdicts.
- No model, GPU, training, local, or benchmark event has run.

## 2026-07-15 — Authenticated control training

- A first launch attempt failed pre-GPU on a shell working-directory slip (no
  model event, no artifact); the stage relaunched cleanly from the same green
  checkpoint.
- `train-control` ran from freeze commit `e773ed5e` (clean synced green main):
  `replay_clean` trained 1,520/1,520 rows with 0 skipped over 190 updates;
  receipt/log published and pinned fail-closed.

## 2026-07-15 — Authenticated candidate training

- `train-candidate` ran only after control checkpoint `b063c9c6` matched
  `origin/main` with both workflows green and a clean worktree.
- `hygiene_explore` trained 1,520/1,520 rows with 0 skipped over 190 updates;
  receipt/log published and pinned fail-closed. Merges are next.

## 2026-07-15 — Authenticated explicit composites

- `merge-arms` ran only after candidate checkpoint `a002d5ae` matched
  `origin/main` with both workflows green; both arms merged (128/128 modules,
  fingerprint-verified); tree pins filled fail-closed. The one frozen 124-task
  gate event at seed 88,018 is the only next stage.

## 2026-07-15 — Recovery gate: installs recovered; retention bands failed; closed

- The gate ran from merge checkpoint `866f23ce`: three authenticated engine
  runs over the 124-row input at seed 88,018 with normalized grading.
- Axis holdout of 20: candidate 15, replay 11, parent 8; explore 7/4/6 (win),
  hygiene 8/4/5 (win); RECOVERY both true — the de-stacking reading is
  positive (interference confirmed; content decay refuted; the escalation rule
  does not fire).
- Retention: 58/93/11 vs parent 68/98/7 and replay 66/86/19 — the correct
  band failed against both controls, the cap and parse bands against the
  parent. No promotion; seed 78,148 permanently sealed.
- Cross-receipt isolation: the retention-safe dose-two precedent had a full
  replay round between doses; this direct dose did not — replay interleaving
  protects retention.

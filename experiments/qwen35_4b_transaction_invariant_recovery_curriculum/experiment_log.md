# Experiment log

## 2026-07-12 — design and CPU validation

- Created the successor after the public branch tournament found 20/20 shared
  deterministic failures on atomic reservations.
- Added six transaction training families and three new transfer families;
  retained atomic reservations as the predecessor sentinel.
- Fixed five hidden tests whose partial implementation initially passed the
  hidden half even though it failed the visible suite. Revalidated 30 generated
  tasks across all ten families: initial and partial fail both checks; full
  repairs pass both.
- Built the matched bank path: 24 transaction + 24 recovery blocks versus 48
  recovery blocks, seven transitions per task, equal action mass, zero plan
  loss, exact replay-source hash.
- Added exact parent-weight validation, explicit merge lineage, 512+512
  looping evaluation, locality entropy/varentropy audit, staged feasibility and
  candidate gates, unit tests, and resumable CPU/GPU/full runners.
- CPU smoke: PASS. No candidate training or result-bearing generation yet.
- Full bank/tokenization preflight: PASS. Each arm has 48 tasks/336 rows,
  exactly 152,992 weighted action-token mass per epoch, zero whole-task
  padding, a 1,179-token maximum row, and 72 registered optimizer steps. The
  actual 8.5 GB parent weight hash matched before tokenization.

## 2026-07-13 — GPU smoke, training, and control-manifest correction

- GPU smoke passed two optimizer steps, an explicit 128-module merge, and a
  real six-turn vLLM recovery block.
- Restored the two documented Qwen training fast paths in `.venv`; both import
  checks and an actual forward pass succeeded.
- Trained both 72-step result arms. Primary aggregate loss 0.0612, merge delta
  norm sum 16.735; replay-only loss 0.0419, delta norm sum 5.683. No padding,
  truncation, fallback, OOM, or retry occurred.
- Primary passed apex-relative locality: drift 0.1194, entropy delta +0.0112,
  varentropy delta −0.0002.
- Parent and replay-only calibration controls each scored 25/60, but their
  procedural manifest hashes differed. Forensics localized the byte difference
  to randomized Python set-literal rendering in seat-group visible tests. The
  feasibility analyzer stopped before candidate exposure. Quarantined both
  invalid payloads, froze `PYTHONHASHSEED=0` for every official child process,
  and reran controls before continuing. No threshold or model changed.

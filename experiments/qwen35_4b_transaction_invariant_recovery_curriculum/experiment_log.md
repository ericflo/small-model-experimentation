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

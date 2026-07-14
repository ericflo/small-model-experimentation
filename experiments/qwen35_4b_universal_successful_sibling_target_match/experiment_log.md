# Policy-Supported Successful-Sibling Universal Curriculum Experiment Log

## 2026-07-14 — Intake and model-free freeze

- Claimed the queued successor from the terminal clean-restart result in a new experiment directory.
- Identified `qwen35_4b_universal_failure_selected_restart_target_match` as the closest near-duplicate and active replay as the mechanism-falsifying control.
- Reserved fresh construction/greedy/sibling/selection/training/local/aggregate seeds `77115/66115/66116/55115/49/88011/78141`.
- Materialized 624 fresh truth-audited tasks, 48 per skill, plus an oracle-free greedy input.
- Froze two separate parent events: greedy failure identification first; `n=16` sibling sampling only after those failures are committed.
- Froze a four-per-skill successful-sibling gate, 768-thinking-token ceiling, shortest-qualified selection, and an absolute prohibition on oracle-trace fallback.
- Added adversarial review verdict `PASS_GREEDY_COLLECTION`; no later event is authorized.
- Experiment test suite: 27/27 passes under `.venv` with bytecode disabled.
- No model, GPU, training, local, or benchmark event ran during this stage.

## 2026-07-14 — Authenticated greedy collection

- Launched only after design commit `0038fba1` passed Validate Repository `29371704674` and Publish Research Site `29371704828`.
- The frozen parent completed 624/624 rows and 296,259 sampled tokens at 859.6 tok/s in 392.0 wrapper seconds.
- Raw/metadata/log/receipt hashes are `e91313c0...f556` / `0e82ae73...15ce` / `f1657151...ca4` / `cee1f19d...4962`.
- Recovery was unused and generation was not rerun. Benchmark data was not read; aggregate remains sealed.
- Failure grading and the oracle-free sibling input remain unopened until this collection checkpoint is published green.

## 2026-07-14 — Terminal greedy-failure availability stop

- Opened model-free grading only after greedy commit `5b784ac5` passed Validate Repository `29372380559` and Publish Research Site `29372380538`.
- Found 227 hard failures overall, but the mandatory per-skill quota failed: `count=0`, `route=0`, and `select=2` versus four required.
- Preserved outcome `STOP_INSUFFICIENT_GREEDY_FAILURES`; inventory/receipt hashes are `8e21caf8...d783` / `3397b773...2a6e`.
- Emitted no sibling input and consumed no sibling, training, local, or aggregate seed.
- Closed the experiment immediately. The validation-only harness now treats an authenticated terminal gate's expected exit code 2 as a passing smoke condition; selection logic and the historical design receipt are unchanged.

# Experiment Log

## 2026-07-13 — Intake and design freeze

- Created a result-separated successor to the exact-token mid-density negative.
- Audited the parent local receipt: all three unparsed designed160 cases were
  cap-bound and belonged to `u_execute`/`u_induct`.
- Rejected answer-only teacher forcing after an injected close because C51 already
  showed that counterfactual state is not reliably reached.
- Split the natural `</think>` span from thought and answer loss in the trainer.
- Selected 40 fresh execute and 40 fresh induct rows with zero parent overlap.
- Constructed a 200-row shared replay core, 40-row replay filler, and 120-row replay
  control. Target+filler and replay control each total exactly 87,454 forward tokens.
- Validated 320 rows, 286,814 forward tokens, 40 updates, and zero skips per arm.
- Added stream-freshness, slot-identity, boundary, treatment-locality, negative-row,
  runner, and local-gate tests; 27 tests pass.
- Completed adversarial design review and preregistration. No scientific GPU work,
  model evaluation, merge, or benchmark event has run.

Next: commit/rebase/push the frozen design to `main`, verify CI, then run and publish
each training arm as its own incremental checkpoint.

## 2026-07-14 — Replay control training

- Published design checkpoint `0fe1a931` to `main`; repository validation and site
  publication both passed on GitHub before scientific training began.
- Trained `replay_repeat` for exactly 40/40 updates over the authenticated 320-row,
  286,814-forward-token stream; zero rows skipped.
- Train loss: 0.4477. Wrapper wall time: 303.4403 seconds.
- Adapter weights/config: `ca5601cd...59d78` / `63575f72...a49b`.
- Training receipt/log: `ffac35d1...1067` / `b24444b8...ed28`.
- No treatment, local evaluation, merge, or benchmark event has run.

Next: publish this control receipt checkpoint, then train `standard_xi`.

## 2026-07-14 — Ordinary-weight target training

- Published replay-control checkpoint `6d1761e7` to `main`; repository validation
  and site publication both passed on GitHub before the next arm began.
- Trained `standard_xi` for exactly 40/40 updates on the byte-frozen target stream;
  320 rows, 286,814 forward tokens, and zero skips.
- Train loss: 0.6882. Wrapper wall time: 302.1492 seconds.
- Adapter weights/config: `271569fd...3569c` / `3e035fbb...f91ec`.
- Training receipt/log: `9ed47653...af8b4` / `73c55663...76294`.
- The close-weighted arm, local evaluation, merge, and benchmark remain unrun.

Next: publish this standard receipt checkpoint, then train byte-identical `close_xi`.

## 2026-07-14 — Close-weighted target training

- Published ordinary-weight checkpoint `f8f1b13b` to `main`; repository validation
  and site publication both passed on GitHub before the close arm began.
- Trained `close_xi` for exactly 40/40 updates on the byte-identical target stream;
  320 rows, 286,814 forward tokens, and zero skips.
- The receipt authenticates the sole registered contrast: target close kinds
  `u_execute`/`u_induct` at weight 1.0; all ordinary close spans remain 0.2.
- Train loss: 0.6822. Wrapper wall time: 287.1305 seconds.
- Adapter weights/config: `16e9dc75...3c179` / `de953bd5...c47ff`.
- Training receipt/log: `b18df864...3195a` / `66c00e2b...42308`.
- All three arms are trained. Local seed 88006, merge, and benchmark remain unrun.

Next: publish this final training checkpoint, then run the preregistered paired local
evaluation exactly once.

# Backlog

## Next Experiments

- Stopped experiment: `qwen35_4b_specialist_policy_integration` — incumbent and
  compound-headroom gates passed, but `ferrier = 0.994` made the mandatory
  tools specialist's frozen `+0.10` bar mathematically impossible. Zero
  specialist or MOPD updates ran. Do not lower the bar or extend this directory.
- Candidate follow-up: a new specialist-integration experiment with the same
  controls and corrected MOPD objective, but a disjoint-calibrated harder
  tools/provenance core and a mandatory per-core ceiling/headroom gate before
  best-of-k or training. `gatepost`-style provenance has measured room, but any
  new training/transfer split needs fresh confirmatory seeds and its own
  experiment lifecycle.

- Completed negative: `qwen35_4b_interactive_policy_curriculum` — the run was
  already underway when the specialist experiment became active. Its
  full-sequence state-aware DAgger arm failed the mechanism gate (−25.3pp
  trained, −33.3pp untouched) through semantic-operator capture despite clean
  atom and closure guards. RL, controls, and Menagerie stopped. The active
  specialist experiment retained its copied machinery and registered mixed
  control, but its earlier feasibility stop meant the control never ran. Do not
  independently rerun this broad warm start.

- Experiment: `qwen35_4b_gauntlet_breadth_round1` — build the 12-family gym,
  run round-1 expert iteration, first-ever menagerie-arbitrated install.
- Experiment: round 2 re-harvest with the round-1 adapter (does the frontier
  move, does iteration compound or re-saturate).
- Experiment: breadth-vs-dose ablation — one family at matched total examples
  vs the full mixture (is breadth causal, or is it just dose?).
- Experiment: leave-one-axis-out mixture — train on 9 families, measure the
  left-out axis's gym family + menagerie per-family delta (which axes need
  in-axis data vs transfer in from the mixture).

## Required Controls

- For the interactive-policy line: C53 blend incumbent, DAgger-only,
  compute-overmatched new-state SFT, shuffled trajectory rewards, exact oracle
  ceiling, family holdout, atom/closure retention, and matched-compute sampling.
- Semantic entropy/outcome variance may route state acquisition; it may not
  scale token loss or serve as a correctness reward (C52).
- Any future live-state warm start must gate the scarce `VERIFY`/`COMMIT`
  operator rates and neighboring-policy/logit locality before trajectory RL;
  full-sequence correctness labels alone are insufficient.
- For specialist integration: require all four specialists to beat
  sample-more, DAgger, extra SFT, and shuffled reward before MOPD; require
  correct-teacher continuation and exact-logit locality before integration;
  compare end-to-end matched joint RL, off-policy SFT, parameter merge, and
  KL-matched wrong routing; keep all benchmark seeds sealed until held-out
  compound transfer passes.

- Baseline: base model, same fresh menagerie seed, same tier/decode, every event.
- Mechanism-falsifying control: held-out gym families (never trained) separate
  generic-protocol gains from axis-specific gains; parse/forced-close/horizon
  diagnostics reported alongside scores.
- Shift or robustness check: replication on a second fresh menagerie seed
  before any claim; confirm quick-tier conclusions on medium/slow.

## Stop Conditions

Two consecutive rounds with menagerie quick delta inside the noise floor AND
flat gym-internal held-out-family transfer would establish "locality survives
breadth" — codify the negative claim, then pivot the program to targeted
variants (think-economy-only mixture, abstention-only install) or retire.

- Completed: `qwen35_4b_think_ftpo_round2` — confident-wrong-turn filtering
  plus positive-only uplift. P1/P2/P3 failed; true labels separated from
  shuffled locally but shared-weight collateral erased held-out capability.
- Candidate only after a new intake/design review: locality-first think-pivot
  steering. Compare +0.25 uplift with a context-gated last-layer/activation
  intervention and stop at P1 unless median non-target drift is ≤0.10. Do not
  fund n=32/gap=1.0 harvesting until a mechanism passes that preflight.

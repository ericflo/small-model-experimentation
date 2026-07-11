# Backlog

## Next Experiments

- Active experiment: `qwen35_4b_specialist_policy_integration` — split the
  live-state DAgger/execution-RL curriculum into discovery, control, tools, and
  pairwise-composition specialists, prove their same-prefix advantage, then
  integrate them on-policy and test fully held-out composition. CPU substrate
  smoke is complete; model work is in progress. This experiment owns the next
  capability/integration claim slot until its registered stop hierarchy resolves.

- Precursor: `qwen35_4b_interactive_policy_curriculum` — its state-aware DAgger
  and guarded execution-reward machinery is copied into the active standalone
  experiment. Do not launch the full mixed-policy GPU run independently while
  specialist production is active; that mixed arm is a registered control in
  the new experiment.

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

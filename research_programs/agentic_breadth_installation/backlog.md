# Backlog

## Next Experiments

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

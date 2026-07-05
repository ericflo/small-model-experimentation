# Pre-registration: is the decomposition failure a LOOKAHEAD wall, and does banking fix it?

Logged 2026-07-05, before full-scale data. PIVOT from the original "is depth-3 latent" framing after an
adversarial workflow review showed (a) that framing is unfair/flawed and (b) the sibling C12 already ran the
base-model decomposition search (model guidance buys EFFICIENCY not COVERAGE; brute-force matches it). This
experiment instead DISSECTS *where* the model's decomposition breaks and tests whether BANKING (C24) fixes it.

## Question
"Be your own tool-search": the fixed model ranks the next DSL op (likelihood over 32 ops given current lists ->
goal lists); the interpreter applies + verifies; beam search. A feasibility diagnostic (n=20) already showed
the model's next-op ranking is a LOOKAHEAD wall: step-3 (goal 1 op away) top-1 0.40 >> chance 0.031, but
step-1/2 (goal 2-3 ops away) ~chance. So the model has depth-1 recognition but no multi-step lookahead.

## Method
- Substrate: list 16-op DSL (32 op/param combos). Held-out: 80 min-depth-VERIFIED true-depth-3 tasks
  (families.py collapse-rejects; all 80 confirmed true-depth-3). Search termination uses VISIBLE examples ONLY;
  generalization graded on HIDDEN (peeking bug fixed).
- **Phase RANK** (headline): per-step ground-truth next-op ranking accuracy (top-1, top-6, mean rank) at steps
  1/2/3, for base vs banked_640 vs banked_1280 (C24 adapters). Does banking lift LOOKAHEAD (step-1/2) or only
  terminal recognition (step-3)?
- **Phase SEARCH**: end-to-end hidden-generalizing coverage vs interpreter-call budget, for base-guided /
  banked-guided / BRUTE-FORCE (all 32, the honesty bar) / RANDOM (floor), sweeping beam width.
- **Phase ABLATION** (must-fix): 2x2 {model, brute, random} x {distance-pruned, unpruned} at fixed budget --
  does the interpreter's distance-to-target pruning do the model's work?

## Predictions (locked)
- **P1 (lookahead wall):** base ranking accuracy is monotone in proximity to goal -- step-3 >> step-1/2, with
  step-1/2 near chance (top-1 < 0.10). Confirms depth-1 recognition without lookahead.
- **P2 (does banking install planning?):** THE decision. If banked_640/1280 lift step-1/2 top-1 materially
  above base -> banking installs transferable LOOKAHEAD. If step-1/2 stay ~chance while step-3 rises -> banking
  is MONOLITHIC COMPILATION (an input->output map), NOT planning, and cannot rescue the search.
- **P3 (coverage bar):** base-guided search does NOT beat brute-force on hidden coverage (replicating C12 on
  this substrate). Banked-guided beats brute on coverage ONLY IF P2 shows banking lifts lookahead.
- **P4 (pruning):** random-proposals + distance-pruning does NOT crack depth-3 (pruning alone isn't the
  solver); model must beat random for its proposals to matter (they already tie at 0 in the smoke).

## Honest limits
Likelihood-ranking a closed 32-op set is easier than free generation (a secondary free-gen check is noted, not
core). Single frozen held-out (n=80), one seed. The claim is about step-wise proposal/planning, scoped to this
substrate; it does not re-litigate C24's monolithic banking result (which stands).

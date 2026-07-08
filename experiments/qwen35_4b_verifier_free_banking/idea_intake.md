# Idea Intake

## Program Fit

- Program: posttraining_and_adaptation (banking arc) × evidence_conditioned_selection (confidence filter)
- Existing or new program: existing
- Closest program scorecard reviewed: knowledge/program_scorecards.md
- Related future queue item: committee_verifier_critic_loop (different: that trains/uses a critic committee at inference; this uses the base model's raw P(True) logit as the TRAINING-DATA filter)

## Prior Evidence

- Anchor 1: qwen35_4b_coverage_banking (C18 — banking execution-verified self-solutions shifts the proposal distribution; the harness this experiment replicates)
- Anchor 2: qwen35_4b_code_confidence + qwen35_4b_humaneval_code_confidence (C46 — P(True) is a calibrated verification-free judge on real code: selection 0.76–0.84, abstention AUROC 0.84–0.86)
- Anchor 3: qwen35_4b_neurosymbolic_repl_substrate (C11 — self-training on verified self-solutions banks capability; needs contamination-free substrate)
- Anchor 4: qwen35_4b_coverage_dpo_gap (C29 — the model is a usable read-only verifier, 2AFC 0.81)
- Closest duplicate or near-duplicate: qwen35_4b_oracle_distilled_semantic_verifier (trains the model AS a verifier using oracle hidden-test labels; here NOTHING is trained on oracle labels — the untrained model's own logit filters its own training data, and execution appears only in scoring and the ceiling arm)

## Novelty Claim

The banking arc (C11–C24) and the confidence arc (C40–C46) have never touched: no experiment has tested whether the model's own P(True) readout can REPLACE the execution verifier as the training-data filter in the self-training flywheel — nor measured what banking does to calibration itself.

## Related Claims

- C11: Self-training on verified self-solutions banks capability (verification = execution; this asks if the model's own logit suffices)
- C18: Banking shifts the proposal distribution (concentration + expansion) — the effect this experiment tries to reproduce verifier-free
- C41/C46: confidence-select beats self-consistency verification-free; P(True) is the program-level signal
- C29: the model is a read-only verifier (2AFC 0.81) — supports filter feasibility

## Mechanism

Why it should work: C46 shows per-candidate P(True) is calibrated enough that top-P(True) selection approaches visible-test execution; a top-fraction filter over a large harvest should yield a training set of ~0.8+ purity, and C18-style banking may tolerate the residual noise (banking's gain is diversity-driven, C24, and SFT averages over examples).

What would make it false: (a) banking could be far less noise-tolerant than selection — 15–25% confident-but-WRONG examples may teach exactly the confidently-wrong modes C41 warned about (the mode is confidently wrong), canceling the gain; (b) the filter's survivors may be biased toward EASY tasks (confidence correlates with solvability), collapsing training-set diversity — the actual driver of banking gains (C24); (c) training on own high-P(True) outputs may inflate P(True) globally and destroy the calibration that makes the flywheel possible (feedback collapse — round 2 would then be poisoned).

## Control Plan

- Baseline: base model (no training) + bank-random (same-size uniform sample of the harvest = the no-filter floor).
- Ceiling: bank-exec (execution-verified pairs, same size — the C18 arm).
- Mechanism-falsifying control: bank-conf (top-P(True), same size, NO execution anywhere in its pipeline). If bank-conf ≈ bank-random, the filter adds nothing; if bank-conf ≈ bank-exec, the verifier is replaceable.
- Shift or robustness check: threshold dose-response (filter purity vs quantity); training-set purity and task-diversity measured post-hoc with the oracle (reporting only).
- Calibration survival: within-cell AUROC of P(True) on held-out tasks, before vs after banking, per arm; verbalized-style inflation check (mean P(True) drift on identical candidates).
- Hidden-label boundary: hidden-test execution is used ONLY for (a) final scoring of all arms and (b) constructing the bank-exec ceiling arm. The bank-conf pipeline never touches execution results, visible or hidden.

## Evidence Output

- Program evidence update: posttraining_and_adaptation evidence.md
- Claim ledger or synthesis update: new claim (~C47): can confidence replace the verifier in the flywheel? (positive, negative, or scoped — all informative)
- Reusable artifact: confidence-filter harvest module (P(True) scoring of harvested candidates); calibration-before/after eval
- Stop or branch condition: if bank-conf ≥ ~80% of bank-exec's gain over base AND calibration AUROC does not degrade → branch to round-2 flywheel (harvest from banked model, filter by ITS P(True), bank again). If bank-conf ≤ bank-random → stop; document the noise/diversity autopsy as the scoping law.

## Decision

- Run experiment: YES
- Create program: no
- Write synthesis only: no
- Defer: no

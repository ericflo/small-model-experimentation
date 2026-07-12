# Idea Intake

## Program Fit

- Program: agentic_breadth_installation
- Existing or new program: existing
- Closest program scorecard reviewed: knowledge/program_scorecards.md
- Related future queue item: acquisition_policy_comparison_pool

## Prior Evidence

- Anchor 1: qwen35_4b_recovery_reason_locality_interpolation
- Anchor 2: qwen35_4b_verifier_conditioned_recovery_bank
- Anchor 3: qwen35_4b_repo_search_compress_bank
- Closest duplicate or near-duplicate: qwen35_4b_recovery_reason_locality_interpolation

## Novelty Claim

Freeze the safe λ=.18 recovery checkpoint and test it in a payload-capable
looping harness where all arms receive the same larger tool-answer budget and
conditional recovery allows one valid inspection before a changed patch.

## Related Claims

- C50: Breadth-first expert iteration on a firewall-clean gym INSTALLS SUBSTRATE-GENERAL agentic competence: +0.22/+0.29 on blackbox menagerie quick (paired, deterministic) and +0.52 gym-wide including never-trained families -- the locality laws (C43/C45/C48) do not extend to this regime, and the causal lever was gradient placement at the answer-emission seam, not dose (Promising)
- C53: THE SECOND WALL: the emission-policy install is a large ONE-TIME step to a robust menagerie ceiling (quick later broken to ~0.50 by convex mix composition; medium arm-means top out ~+0.31) — no variant of train-on-own-verified-outputs (dose, iteration, breadth, difficulty escalation, recovery supervision, deploy-budget matching) moves the blackbox band further, even as in-gym frontier competence installs (Promising)
- C41: Beat sample-more with the model's own uncertainty: confidence-select (argmax P(answer), verification-free) beats self-consistency, which is flat; max P(answer) predicts solvability (AUROC 0.83) for abstention (Promising)

## Mechanism

The predecessor's best checkpoint solved 58/60 recovery cases at safe locality,
but every invalid response ended inside JSON at the exact 256-answer-token cap.
All rejected cases changed the patch within two turns and solved, mostly via
INSPECT→PATCH. A 512-token answer slot should expose the already-present tool
action instead of truncating it, while a two-turn transition metric measures
recovery rather than impatience. The mechanism is false if cap hits/invalids do
not fall, the candidate loses its advantage when every control gets the same
budget, or the effect fails untouched families.

## Control Plan

- Baseline: frozen C54 apex under the identical 512-think/512-answer harness.
- Mechanism-falsifying control: happy-action and full recovery-action endpoints
  receive the same payload budget; two shorter apex trajectories get the same
  reserved tokens; an explicit runtime recovery scaffold tests promptability.
- Shift or robustness check: one fixed λ=.18 model gets a wholly new locality
  block, then must pass two untouched four-family transfer blocks with normal
  solve/verify/commit retention.
- Hidden-label boundary: procedural hidden executables stay host-only. No
  benchmark implementation, task, transcript, or result is read; Menagerie is
  CLI-only after both transfer gates.

## Evidence Output

- Program evidence update: whether answer-payload capacity converts the safe
  recovery weights into a deployable, family-transferring agent policy.
- Claim ledger or synthesis update: only if both transfer blocks and Menagerie
  pass; calibration alone remains mechanism evidence.
- Reusable artifact: payload telemetry, valid changed-patch-within-two metric,
  feasibility receipts, and the copied looping harness.
- Stop or branch condition: fresh-locality failure stops before behavior;
  infeasible or failed calibration stops before transfer; either transfer
  failure seals Menagerie.

## Decision

- Run experiment: yes; it directly tests the predecessor's observed bottleneck
  without changing weights or retroactively relaxing that result.
- Create program:
- Write synthesis only:
- Defer:

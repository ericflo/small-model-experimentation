# Idea Intake

## Program Fit

- Program: agentic_breadth_installation
- Existing or new program: existing
- Closest program scorecard reviewed: knowledge/program_scorecards.md
- Related future queue item: synthetic_curriculum_transfer_bakeoff

## Prior Evidence

- Anchor 1: qwen35_4b_universal_curriculum
- Anchor 2: qwen35_4b_transaction_invariant_recovery_curriculum
- Anchor 3: qwen35_4b_verifier_conditioned_recovery_bank
- Closest duplicate or near-duplicate: qwen35_4b_universal_curriculum

## Novelty Claim

Test whether a truth-audited, behaviorally non-collapsed, format-diverse curriculum of
generic search, execution, verification, recovery, uncertainty, and optimization
procedures adds held-out capability beyond the already-proven broad emission-policy
curriculum, rather than merely fitting another synthetic surface.

## Related Claims

- C56: AXIS-STRUCTURED INSTALL COMPRESSION: at the maxed 8192 menagerie budget the two weakest axes DISSOCIATE — EXPLORATION is installable and transfers (gym burrowmaze mean +0.167 at 8192, L6 0.33->0.67; menagerie medium retain-delta +0.190 > the efficiency install's +0.146) while composed-rule INDUCTION is NOT (gym glyphgate L4-L6 stay ~0.0 before and after; trace-SFT even DEGRADES the easy induction the base could already do, L2 0.93->0.53). No single-4B install flavor clears the +0.32 conjunction at fair budget; decomposed by axis, the residual IS the executor-vs-inducer wall (C39/C44/C48), a serial-compute property of the fixed model, not a data or method gap. Answers C55's open next-test. (Promising)
- C40: The model knows when it will fail IMPLICITLY (answer-token probability, within-cell AUROC 0.95) but NOT EXPLICITLY (self-verification P(True) at chance, verbalized confidence a constant 100) (Promising)
- C53: THE SECOND WALL: the emission-policy install is a large ONE-TIME step to a robust menagerie ceiling (quick later broken to ~0.50 by convex mix composition; medium arm-means top out ~+0.31) — no variant of train-on-own-verified-outputs (dose, iteration, breadth, difficulty escalation, recovery supervision, deploy-budget matching) moves the blackbox band further, even as in-gym frontier competence installs (Promising)

## Mechanism

The C50/C53 broad curriculum already installs terse commitment and generic execution,
but leaves several benchmark families flat or negative. C14/C28/C56 say that narrow
answer narration is not enough: transferable gains require correct executable plans,
an intact answer interface, and representation across formats. The proposed curriculum
therefore teaches the same small set of control circuits across fresh pseudo-vocabularies
and heterogeneous task renderings, with explicit dead ends, counterexamples, repair,
abstention, and state carry. Mixing it with the frozen C53 broad replay should preserve
the emission policy while adding missing procedures. The mechanism is false if the
designed-only arm cannot learn its own clean held-out substrate, if the combination does
not beat broad replay on fresh aggregate-only benchmark events, or if gains are confined
to a subset of families.

## Control Plan

- Baseline: pinned Qwen/Qwen3.5-4B and the existing C53 `blend` adapter, evaluated on
  the same fresh seed, tier, canonical think budget, and vLLM merged-checkpoint backend.
- Mechanism-falsifying control: designed curriculum without broad replay versus broad
  replay without designed rows versus their combination at matched update settings;
  require zero skipped targets and audit exact token exposure.
- Shift or robustness check: fresh pseudo-vocabulary and held-out generator seeds first;
  then independent Menagerie quick seeds and a medium-tier confirmation. Same-surface
  accuracy is only an installability gate, never the transfer claim.
- Hidden-label boundary: benchmark subprocesses run only through
  `scripts/run_benchmark_aggregate.py`; the experiment receives aggregate and public
  per-family scores only. No benchmark files, items, transcripts, or raw child streams
  are read, imported, or retained.

## Evidence Output

- Program evidence update: add every positive and negative trial to
  `agentic_breadth_installation/evidence.md` and revise its backlog if a curriculum
  component changes the best next experiment.
- Claim ledger or synthesis update: only after independent quick and medium
  confirmation; a pilot or mixed-family result remains experiment-local.
- Reusable artifact: deterministic generator, truth/min-depth validator, corpus and
  training receipts, aggregate-only trial ledger, and external adapter/merge manifest.
- Stop or branch condition: reject a corpus on any truth contradiction, behavioral
  depth collapse, duplicate, target truncation, or firewall failure. Reject a model
  before confirmation unless aggregate delta is positive and every public family delta
  is non-negative; iterate in a new experiment after any result-bearing adaptive change.

## Decision

- Run experiment: yes, after the clean-corpus and aggregate-gateway smoke gates pass.
- Create program: no; `agentic_breadth_installation` is the closest existing program.
- Write synthesis only: no; the current evidence does not answer the clean designed-data
  plus broad-retention question.
- Defer: benchmark confirmation remains sealed until local corpus/training gates and a
  positive quick pilot pass.

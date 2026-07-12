# Idea Intake

## Program Fit

- Program: `agentic_breadth_installation` (primary), with
  `posttraining_and_adaptation` and `benchmark_generalization` supporting.
- Existing or new program: existing. C53 explicitly names execution-reward RL
  as the next capability-production mechanism after the verified-output wall.
- Closest program scorecards reviewed: all three attached program entries in
  `knowledge/program_scorecards.md`.
- Related queue item: `posttraining_method_shared_substrate`; this experiment
  is narrower and tests a new capability-integration mechanism after qualified
  specialist RL.

## Prior Evidence

- Anchor 1: C50/C53 — emission-seam expert iteration created a large one-time
  transfer gain, then every train-on-own-verified-output variant saturated and
  mixture composition exposed cross-profile tradeoffs.
- Anchor 2: C11/C21-C24 — self-banking is coverage-bounded; an external
  explorer can seed a missing rung and training can install it.
- Anchor 3: C29/C52 — dense or token-local updates can collapse or move
  unrelated logits, so shuffled-signal and exact-logit locality controls are
  prerequisites.
- Anchor 4: the OPSD hint and execution-feedback audits found no incremental
  same-prefix advantage over shuffled controls.
- Closest near-duplicate:
  `qwen35_4b_interactive_policy_curriculum`. It creates one mixed live-state
  DAgger/GRPO policy. This experiment creates independent same-origin
  specialists, proves their prefix-level advantage, and integrates them into a
  fresh student before testing unseen composition.

## Novelty Claim

This is the first repository experiment to separate capability production from
capability integration. It is also the first to require an integrated fixed-4B
policy to outperform its own composition specialist on tasks that combine
primitives in a never-trained pairing or depth.

## Mechanism

Execution reward first creates policies with demonstrated outcome headroom.
Because every teacher descends from the same C53 checkpoint and receives the
same observable prefix as the student, on-policy policy-space distillation can
transfer dense decisions without a solution-conditioned privileged branch or
off-policy exposure mismatch. The explanation is false if specialists do not
beat sample-more, if their token pressure does not predict better branches, if
wrong-route distillation matches correct routing, or if gains stop at the
single-domain union and fail held-out compounds.

## Control Plan

- Baseline: regenerated merged C53 incumbent, greedy and execution-filtered
  best-of-8 under the identical vLLM protocol.
- Mechanism-falsifying controls: DAgger-only, additional SFT, shuffled rewards,
  end-to-end compute-matched joint RL, off-policy specialist SFT, parameter
  merge, and initial-KL-matched wrong-teacher routing.
- Shift or robustness: three primitive transfer families receive no new
  exposure; two compound families and order reversals are fully held out;
  three end-to-end seeds are required for the primary arms.
- Hidden-label boundary: programmatic state may label DAgger and score executed
  outcomes, but the model sees only the transcript. Specialist prefills see the
  identical observable prompt and prefix. Benchmark contents stay unread and
  the CLI remains closed until every whitebox gate.

## Evidence Output

- Program evidence update: whether specialist RL creates beyond-C53 headroom,
  whether MOPD consolidates it, and whether consolidation composes.
- Claim ledger or synthesis update: only after a reached gate yields a durable
  positive or scoped negative.
- Reusable artifact: four exact compound environments, necessity ablations,
  same-prefix teacher audit, corrected top-k loss, compute ledger, and
  integration analyzer.
- Stop or branch condition: stop at the first failed capability, teacher-gap,
  locality, or integration gate. Never rescue a failed gate by increasing
  dense-loss weight. Preserve the negative stage and do not spend benchmark
  seeds.

## Decision

- Run experiment: yes; accepted by the active goal on 2026-07-11.
- Create program: no.
- Write synthesis only: no.
- Defer: no.

## Reached Outcome and Follow-up Intake (2026-07-12)

- Outcome: stopped before specialist production. `ferrier` scored 0.9940 on
  the paired baseline, making the frozen `S0 + 0.10` tools target 1.0940 under
  a hard score ceiling of 1.0. The proposed mechanism was not tested.
- Novel durable lesson: a multi-teacher experiment needs a per-teacher
  theoretical-headroom gate before best-of-k or training; endpoint/compound
  headroom alone is insufficient.
- Closest follow-up duplicate: this experiment itself. A replacement tools
  family or split must therefore be a new standalone experiment, not an
  extension of this result-bearing directory.
- Candidate follow-up: preserve discover/control/compose and the full MOPD
  control stack, but use a disjoint-calibrated harder tools/provenance core
  (potentially fresh `gatepost`-style procedural seeds plus a new held-out
  provenance chain) with `S0` far enough below 1.0 to clear every frozen bar.
- Mechanism-falsifying controls remain DAgger, compute-overmatched SFT,
  deranged rewards, correct-vs-wrong routing, and matched-compute joint RL.
- Required pre-run evidence: every core must pass `1 - S0_macro >= required
  gain` on calibration cells; replacement selection cannot use confirmatory
  items.
- Decision: queue as a new experiment candidate; do not silently rescue or
  continue the stopped experiment.

# Idea Intake

## Program Fit

- Program: agentic_breadth_installation
- Existing or new program: existing
- Closest program scorecard reviewed: knowledge/program_scorecards.md
- Related future queue item: gauntlet_round3_expert_iteration

## Prior Evidence

- Anchor 1: qwen35_4b_verifier_conditioned_recovery_bank
- Anchor 2: qwen35_4b_think_ftpo_round2
- Anchor 3: qwen35_4b_repo_search_compress_bank
- Closest duplicate or near-duplicate: qwen35_4b_verifier_conditioned_recovery_bank

## Novelty Claim

Interpolate only the learned contrast between a locality-safe recovery-action
adapter and its byte-identical plan-supervised sibling, then screen the entire
frozen path for locality before observing behavior; this separates useful plan
pressure from weakening the already-useful action policy.

## Related Claims

- C50: Breadth-first expert iteration on a firewall-clean gym INSTALLS SUBSTRATE-GENERAL agentic competence: +0.22/+0.29 on blackbox menagerie quick (paired, deterministic) and +0.52 gym-wide including never-trained families -- the locality laws (C43/C45/C48) do not extend to this regime, and the causal lever was gradient placement at the answer-emission seam, not dose (Promising)
- C28: Banking correct decomposition PLANS installs deployable depth-3, but banking the model's OWN rejection-sampled thoughts does NOT (they are rationalizations) -- it is the plan QUALITY, not reasoning-as-such (Promising)
- C54: TIER-PARETO FRONTIER: novel serial-compute mechanisms (length-penalized compression advantage + skin-shuffle) DECISIVELY clear the +0.32 MEDIUM menagerie bar for the first time (+0.345, all events), but no single Qwen3.5-4B model clears quick AND medium together by ANY method (training, capacity, data-interpolation, or weight-space model-soup) — the two tiers occupy a non-convex Pareto frontier and compete for the fixed model's representational budget (Promising)

## Mechanism

The parent experiment found two unusually clean endpoints from the same base,
rows, seed, and optimizer schedule: action-only was local (0.098 drift) but
invalid-action heavy, while adding nominal 5% plan loss repaired validity and
raised calibration success but produced 0.303 drift. If the plan-induced update
contains a low-dose formatting/recovery component before its broad lexical
component dominates, a short action→reason weight path should contain a point
that preserves full action learning, corrects invalid recovery, and remains
local. The explanation is false if every locality-safe point behaves like the
action endpoint, or every behavior-improving point fails either independent
locality confirmation or held-out-family transfer.

## Control Plan

- Baseline: frozen C54 apex, the matched happy-action checkpoint, and the
  full-dose recovery-action endpoint from the parent experiment.
- Mechanism-falsifying control: the full reason endpoint anchors the known
  locality failure; the full action endpoint tests whether interpolation adds
  anything beyond simply deploying the safe parent arm.
- Shift or robustness check: screen all frozen scales on the original 48
  contexts, select with the already-designated trained-family calibration
  block, then test the single winner on a new disjoint 48-context locality
  confirmation with no fallback. Two untouched procedural-family blocks must
  beat matched sampling and an explicit runtime recovery scaffold.
- Hidden-label boundary: generated repository hidden executables remain
  host-only and are reduced to booleans. No benchmark family source, item,
  transcript, or result is read; Menagerie remains sealed behind all gates.

## Evidence Output

- Program evidence update: whether a locality-compliant action→reason region
  exists and whether its conditional recovery transfers.
- Claim ledger or synthesis update: only after independent locality and both
  held-out-family blocks; a calibration-only optimum is not a claim.
- Reusable artifact: explicit two-adapter delta interpolator, entropy/varentropy
  locality ladder, frozen selector, and gate-reachability receipt.
- Stop or branch condition: stop before behavior if no scaled point passes the
  screen; stop without fallback if the selected point fails confirmation; stop
  before Menagerie on either transfer failure.

## Decision

- Run experiment: yes; this is the immediate registered successor to the
  parent's locality-gated negative and reuses no transfer observation.
- Create program:
- Write synthesis only:
- Defer:

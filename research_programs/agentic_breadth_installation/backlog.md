# Backlog

## Next Experiments

- Completed qualification negative: `qwen35_4b_pareto_policy_integration`
  regenerated C54's `blend` and `apex` policies and exercised the corrected
  paired `delta > 0` rule. `blend` lost quick capability in both blocks
  (`-0.0069`, `-0.0379`; pooled `-0.0224`), while `apex` won deep capability
  (`+0.0456`, lower bound `+0.0340`) but missed six retention cells. The run
  stopped before teacher audit or MOPD; do not describe it as an integration
  failure.
- Candidate follow-up, new experiment only: replace assumed tier labels with a
  disjoint same-prefix continuation-advantage router. Both same-origin teachers
  must be sampled and verifier-scored at each calibration state; a frozen route
  qualifies only with replicated positive advantage over the alternate
  teacher. Compare any integrated checkpoint against both teachers, a visible
  two-checkpoint router, and matched-compute sampling. Do not reuse the observed
  qualification cells to select routes.
- Completed negative: `qwen35_4b_repo_search_compress_bank` — exact-token
  operator balance plus replay-minimized successful repository traces improved
  trained families 40/48→48/48 but regressed wholly held-out families
  49/72→25/72 and failed locality. Do not repeat the one-patch success-only
  bank at another dose. Any successor needs a new intake, fresh families/seeds,
  and either (a) verifier-conditioned recovery transitions balanced at the
  state→action level or (b) external scaffold retrieval/execution that avoids a
  broad shared-weight policy edit. Require a synthetic failed-patch/failed-test
  recovery gate before training.
- Completed locality-gated negative: `qwen35_4b_verifier_conditioned_recovery_bank`.
  Conditional transition balance worked on fresh trained-family recovery
  (base 0.483, happy 0.817, action-only 0.850, reason 0.917), and full-dose
  recovery action-only passed locality at 0.098. The selected 5%-plan arm
  failed locality at 0.303 because highly off-policy plan-start tokens created
  a 42.1 pre-clip gradient and 29.5% larger delta norm. Transfer and Menagerie
  stayed sealed. Do not rerun this dose or reinterpret the exploratory
  action-only arm inside the result-bearing directory.
- Completed policy-gated result:
  `qwen35_4b_recovery_reason_locality_interpolation`. Every frozen mixture
  passed locality; λ=.18 reached 0.967 trained-family recovery at 0.104 drift,
  beating base/happy/action/full-reason. It still failed the frozen invalid-turn
  and immediate-rejected-transition gates, so confirmation, transfer, and
  Menagerie remained sealed. Do not extend the ladder or lower gates in that
  result directory.
- Immediate successor, new experiment only: freeze λ=.18 and repair the actual
  deployment interface. All 24 invalid actions were closed-thinking JSON patch
  payloads truncated exactly at the 256-answer-token cap; give every model arm
  realistic answer payload capacity under matched total compute. Replace the
  over-literal immediate rejected-patch proxy with changed patch within two
  turns while retaining the intervening-operator census (all 30 λ=.18 cases
  already changed within two and solved). Re-run selection-only calibration,
  then independent locality and the untouched four-family blocks against base,
  happy, action, external scaffold, and sample-more before Menagerie.
- Future retraining should calibrate plan dose by realized gradient/surprisal
  and avoid supervising plan starts already rank 1 or wildly off-policy lexical
  templates.
- Stopped experiment: `qwen35_4b_specialist_policy_integration` — incumbent and
  compound-headroom gates passed, but `ferrier = 0.994` made the mandatory
  tools specialist's frozen `+0.10` bar mathematically impossible. Zero
  specialist or MOPD updates ran. Do not lower the bar or extend this directory.
- Deferred predecessor repair: a harder disjoint-calibrated tools/provenance
  core could still make the original four-specialist design feasible, but it
  is no longer the best immediate test. The corrected successor shows that
  feasible teachers also need same-prefix advantage on the distillation
  distribution. Any revival must satisfy both prerequisites before best-of-k
  or training and use a new experiment with fresh confirmatory seeds.

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
- Any compressed trajectory bank must balance conditional transitions, not
  only operator marginals: explicitly measure `failed_patch→changed_patch`,
  `failed_test→revision`, and `passed_test→commit`. Success-only minimization
  is not an agent-policy compression method.
- For specialist integration: require all four specialists to beat
  sample-more, DAgger, extra SFT, and shuffled reward before MOPD; require
  correct-teacher continuation and exact-logit locality before integration;
  compare end-to-end matched joint RL, off-policy SFT, parameter merge, and
  KL-matched wrong routing; keep all benchmark seeds sealed until held-out
  compound transfer passes.
- For any state-routed successor: route on a predeclared observable state key or
  a training-only verifier advantage, never on a hidden benchmark label; freeze
  the routing rule on disjoint calibration prefixes before producing updates.

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

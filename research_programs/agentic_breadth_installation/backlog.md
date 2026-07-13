# Backlog

## Next Experiments

- Completed qualification negative: `qwen35_4b_pareto_policy_integration`
  regenerated C54's `blend` and `apex` policies and exercised the corrected
  paired `delta > 0` rule. `blend` lost quick capability in both blocks
  (`-0.0069`, `-0.0379`; pooled `-0.0224`), while `apex` won deep capability
  (`+0.0456`, lower bound `+0.0340`) but missed six retention cells. The run
  stopped before teacher audit or MOPD; do not describe it as an integration
  failure.
- Completed route-qualification negative:
  `qwen35_4b_same_prefix_advantage_routing` replaced tier labels with disjoint
  same-prefix outcomes. Deep and the combined router passed, but quick's
  soup-relative audit macro reversed from `+0.2009` to `-0.0253`; no MOPD,
  locality, or Menagerie event ran. Post-result diagnostics show conditional
  winner noise, not a missing `+0.10` threshold.
- Active experiment: `qwen35_4b_deep_advantage_mopd` uses fresh states to
  requalify the already validated deep route, then conditionally tests the MOPD
  update kernel from the immutable joint soup. It preserves quick behavior
  through the frozen-soup anchor and requires one checkpoint to beat both
  sources, visible routing, matched controls, and sample-more before Menagerie
  escalation. Fresh qualification has now passed: deep routed on 28/26 states
  and its audit advantage over soup was +0.1650/+0.1220 (pooled lower bound
  +0.1230). The five-update pilot also passed literal exact-logit locality
  (drift 0.02760; entropy drop 3.11%; target loss improved), so four-round
  integration is active. No capability result exists yet.
- A new two-teacher attempt needs cross-fitted direct advantage prediction and
  a third untouched route block. If quick again lacks independent support,
  retire it as a complementary teacher; do not tune an observed-margin cutoff.
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
- Completed confirm-gated negative:
  `qwen35_4b_recovery_payload_budget_harness`. The fixed λ=.18 candidate passed
  locality, calibration, and transfer dev, then tied action-only at 0.6875 on
  independent confirmation; every other gate passed and Menagerie stayed
  sealed. The larger payload and two-turn transition metric were validated,
  but reason mixing is complementary rather than uniformly superior.
- Completed prospective infeasibility:
  `qwen35_4b_recovery_verifier_branch_tournament`. On four new families, both
  sources scored 0.7375 and their union only 0.7500, tying action pass-if-either
  sample-more. Feasibility stopped the selector; confirm, banking, and
  Menagerie remained sealed. Do not tune the tie break or try another public
  source router—the two policies had only one exclusive win each.
- Completed transfer-gated negative:
  `qwen35_4b_transaction_invariant_recovery_curriculum` passed locality and
  installed trained transaction families (0.817 versus parent 0.517 and
  replay-only 0.383), but unseen dev was only 0.719 versus parent/sample-more
  0.703. It failed the +10/+5 and bootstrap bars; confirmation, broad retention,
  and Menagerie stayed sealed. Do not repeat generic transaction dose.
- Completed calibration infeasibility:
  `qwen35_4b_validation_policy_counterexample_curriculum` passed locality at
  0.109 but parent and matched extra-training control both scored 48/48 on the
  exact trained-family recovery block. Explicit contract + near-correct partial
  removed the historical residual; +15/+10 bars were impossible and candidate
  behavior, transfer, and Menagerie stayed sealed. Do not lower bars or expose
  the trained candidate post-stop.
- Completed instrument-gated qualification:
  `qwen35_4b_semantic_policy_headroom_tournament`. The formal verdict is
  `INSTRUMENT_FAIL`: answer-cap contacts were 12.08%/12.67% versus the frozen
  5% ceiling. No post-failure axis qualified independently—negative and
  non-integer were 9/9 in both blocks, while blank had only one in-band shape
  per block and the shape changed. Do not train on these failed-test states.
- Active designed and preregistered successor:
  `qwen35_4b_counterfactual_evidence_acquisition_curriculum`. On the opened
  predecessor trajectories, all 72 failed-test cases reached a correct patch,
  but inferred rejected-patch first-patch correctness was 0/54 and visible-test
  inspection before first patch was 0/72. Build counterfactual pairs with
  issue/source held constant and discriminating public evidence flipped; first
  qualify, then train
  `ambiguous_state→inspect_evidence→evidence_faithful_patch` alongside complete
  recovery/verify/commit replay. Repair answer allowance in a separate frozen
  preflight, and keep entropy/varentropy to mining/stratification rather than
  correctness labels. No model-bearing run or result exists yet.
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

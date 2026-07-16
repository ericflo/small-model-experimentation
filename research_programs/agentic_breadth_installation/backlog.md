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
- Completed capability negative: `qwen35_4b_deep_advantage_mopd` found
  replicated same-prefix deep advantage and passed exact locality, but primary
  seed 42 trailed deep by `−0.006845` pooled joint and soup best-of-eight by
  `−0.169239`; seeds 43/44 also trailed deep. Correct-teacher pressure beat
  wrong-teacher and non-advantage controls modestly, while retention/transfer
  passed. This is directional routing signal without source-frontier crossing,
  not a reason to repeat the NF4 recipe at larger scale. Benchmarking stopped.
- Highest-value update-operator experiment: use direct-bf16 microtraining on a
  fresh self-contained substrate and require the causal update to survive its
  deployed merge and beat deep, ordinary interpolation, and matched-compute
  sampling before any full campaign. Preserve the same-prefix verifier and
  matched controls, but make train/deploy parity a hard admission gate.
- A later two-teacher attempt needs cross-fitted direct advantage prediction,
  uncertainty-aware adaptive allocation, and a third untouched route block.
  Permit zero quick allocation if it lacks independent conditional value; do
  not tune an observed-margin cutoff. This estimator work is downstream of the
  direct-bf16 update-kernel prerequisite.
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
- Completed prerequisite stop:
  `qwen35_4b_counterfactual_evidence_acquisition_curriculum` ended at
  `LINEAGE_LOCALITY_INFEASIBLE` before interface behavior or training. The
  transaction-replay parent measured 0.110735 direct drift from C54 apex versus
  the frozen 0.10 ceiling, while entropy passed; every downstream white-box
  stage and Menagerie stayed sealed. The evidence-acquisition hypothesis
  remains untested. Do not relax the gate or swap the parent/anchor in this
  directory. Any successor must use fresh contexts and seeds, start from apex
  or a prospectively fixed apex-compatible parent, and separately requalify
  complete-loop retention plus acquisition headroom before copying the
  curriculum onto fresh skins.
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

- Completed negative parent factorial: `qwen35_4b_universal_curriculum`. The
  designed-only continuation is a specialization negative (+0.1406 versus base
  but three negative families and -0.1385 versus blend). The from-base replay
  union reached 0.692 local accuracy but failed parse (0.846 < 0.90) and cap
  (4 > 2) gates, so benchmark seed 78132 remained sealed.
- Completed designed-arm negative with a stronger control:
  `qwen35_4b_universal_replay_anchor`. The candidate passed local gates but
  scored 0.4238, below `blend` by 0.0172 and replay refresh by 0.0613, with one
  family below base. Replay refresh scored 0.4851, +0.0441 over `blend`, with
  eight positive and two tied families. It is a new anchor, not a universal
  winner.
- Completed exact-token local negative:
  `qwen35_4b_universal_low_density_token_match` trained nested 0/40/80-row doses
  from authenticated replay refresh with exactly 1,429,053 forward tokens,
  1,520 rows, and 190 steps per arm. On fresh seed 88004 every arm missed the
  0.65 accuracy, 0.90 parse, and at-most-two cap bars; the best 80-row arm was
  0.538/0.615/10 and benchmark seed 78134 remained sealed.
- Completed exact-token mid-density negative:
  `qwen35_4b_universal_mid_density_token_match` trained representative 0/160/240
  doses with 1,520 rows, 190 updates, and exactly 1,405,510 forward tokens per arm.
  On fresh seed 88005, the 160-row arm improved replay from 17/26 to 19/26 accuracy,
  18/26 to 23/26 parse, and 9 to 3 cap contacts, but missed the parse and cap gates
  by one case each. The 240-row arm reversed the accuracy gain and worsened both
  emission metrics. No arm advanced; benchmark seed 78135 remains sealed.
- Completed close-weight mechanism negative:
  `qwen35_4b_universal_close_weight_token_match` compared exact-token replay,
  ordinary fresh execute/induct continuation, and byte-identical continuation with
  0.2→1.0 natural-close loss. Fresh seed 88006 scored parent 16/26 accuracy,
  20/26 parse, 6 caps; replay 14/26, 18/26, 8; standard 15/26, 23/26, 3; and close
  16/26, 23/26, 3. No treatment passed, and seed 78136 remains sealed. Retire
  close-span dose tuning: it did not improve parse/cap over ordinary training and
  both target arms remained 0/4 on execute/induct. Do not lower the gate, reuse
  seeds 88005/88006, or consume sealed seeds 78134/78135/78136.
- Completed bounded-computation successor (terminal local negative, positive
  mechanism reading): `qwen35_4b_universal_fresh_surface_budget_commit_target_match`
  trained the fresh-surface designed160 re-dose and the budget-commit ablation
  against a three-axis exact-exposure replay control and evaluated one frozen
  104-task original-surface gate at seed 88013. designed_fresh 69/97/7
  (correct/parsed/caps) beat parent 63/87/18 and replay 62/91/13 on all four
  preregistered strict comparisons with 31% shorter generation — the designed dose
  is surface-general and think-economical — but induct was 0/8 for every arm
  including the parent, so the induct >= 4/8 floor was structurally unpassable; no
  promotion, aggregate seed 78143 permanently sealed. budget_commit was at-or-below
  replay everywhere; the bounded-scan lesson is retired. The generic-dose line
  CLOSES here: its local gate cannot promote any continuation of this lineage
  (0/8 induct is the C38/C39 wall made exact at n=8). Do not reopen with lowered
  floors in an existing directory.
- Completed goal-gap axis-curriculum successor (first local promotion; pilot
  negative): `qwen35_4b_goal_gap_axis_curriculum_target_match` installed its
  axis atoms (holdout 28/40 vs parent 22 / replay 18, retention byte-equal to
  parent) and consumed aggregate seed 78144: base 0.1085, axis 0.4223, parent
  0.4644, replay_repeat 0.5081. Candidate vs base: +0.3138, 7 positive / 3 tie
  / 0 negative families, warren flipped; replay flipped rites and posted the
  line's best recorded aggregate. Pilot gate failed (lost aggregate to parent
  and replay); closed per contract. Standing facts for successors: menders is 0
  and sirens exactly 0.500 for EVERY arm at EVERY seed at quick/tb1024; replay
  continuation has now compounded aggregate three consecutive times.
- Completed stack trial (local negative on the breadth bar alone):
  `qwen35_4b_axis_replay_stack_medium_target_match` — axis install REPLICATED
  on the new parent (24/40 vs 18/15; hygiene 9/10 twice across experiments;
  best-in-event termination), replay round two DRIFTED locally (parse 86, caps
  18), and the protocol holdout tied at the parent ceiling for the second
  consecutive experiment, converting 3-of-4 into 3-of-3 and letting one control
  kind-fluke (explore 7/10) veto promotion. Seed 78145 sealed; medium pilot
  never ran.
- Completed re-adjudication (corrected-bar negative; mechanism map final):
  `qwen35_4b_axis_stack_readjudication_medium_pilot` — all four kinds
  detectable, candidate 22/40 vs 15/18 (third consecutive axis-total win),
  kind wins explore+hygiene only, protocol tied the parent a third time,
  tracefix trended to chance (4->2->1 of 10). Seed 78146 sealed. The corrected
  instrument worked; the deficit is CONTENT: hygiene/explore/termination
  install, tracefix/protocol do not.
- Completed axis corpus v2 (kill rule fired; axis and dose-stacking closed):
  `qwen35_4b_axis_corpus_v2_staged_repair` — demonstrated-search staged repair
  lessons did not install (bugfind tie 3/0/3, bugmend loss 3/4/2 vs
  parent/replay); the candidate tied its parent on the 50-row axis total (19)
  while the THIRD replay round won it outright (25) and the candidate lost
  retention (66 vs 71). Seed 78147 sealed. Standing laws: (1) the trace-repair
  axis is closed for this model at this dose — no v3 without a new mechanism
  argument; (2) third-dose interference — this adapter lineage is saturated
  for designed doses; fresh adapters from clean parents are required for any
  future dose, and replay continuation remains the strongest broad move.
- Completed de-stack test (recovery confirmed; retention bands refused):
  `qwen35_4b_hygiene_explore_destack_medium` — both recovery flags TRUE
  (explore 7/4/6, hygiene 8/4/5; axis 15/20 vs 11/8, the session's strongest
  axis result), so v2's stall was lineage interference, not content decay. The
  direct dose paid ten retention points (58 vs 68/66) and the gate correctly
  refused it; seed 78148 sealed. New law: replay interleaving between doses
  protects retention (the retention-safe dose-two precedent had a dedicated
  replay round at the dose boundary; this trial did not).
- Completed interleaved-dose direct test (interleaving REFUTED; escalation
  fired): `qwen35_4b_interleaved_replay_dose_medium` — hygiene won its sixth
  consecutive event (7/2/3) but explore lost (4/5/3) and retention broke
  against both controls (59 vs 68/69) despite the interleaved parent,
  reproducing the direct dose's cost. Seed 78149 sealed. The replay-
  interleaving law is refuted by direct test; the dose-recipe search is CLOSED
  by the frozen escalation rule. The retention-safe precedent's true cause is
  unidentified (corpus composition 160/4-kinds vs 80/2, lineage depth, or
  screen fortune).
- Completed mechanism cell (REFUTED_INTRINSIC):
  `qwen35_4b_dose_diversity_mechanism_cell` — retention 70/65/61/60
  (parent/replay/axis160/hyg-exp): the diverse dose broke the band (-9), the
  known -10 reproduced, replay itself measured -5, and the sole
  retention-safe precedent is resolved as screen fortune. Hygiene 10/10 (7/7
  all-time); axis160_direct best axis (26/40) and caps (5). LAW: designed
  doses cost ~5-10 retention points intrinsically at this vehicle; the trade
  is priced, real, and two-sided.
- Completed rank-capacity cell (SCREEN_INSTABILITY):
  `qwen35_4b_rank_capacity_vehicle_cell` — the r32 arm's known -9 re-measured
  at -5 on a fresh screen, tripping the preregistered instability guard; no
  capacity inference (r64 measured -7 retention, 19/40 axis,
  install_preserved false, but none of it is adjudicable at current bands).
  Pooled noise across four gates: same-composite retention deltas scatter
  +/-3-4 points; the screen's seed noise rivals the +/-5 band.
- Completed calibration study (CALIBRATION_READ_COMPLETE):
  `qwen35_4b_retention_screen_calibration` — the five published composites
  across four fresh screens (88022-88025, 20 authenticated eval runs, zero
  training). Governing delta-vs-parent SD 4.27 (adversarial review corrected
  the draft's level-SD estimand pre-freeze); recommended band 9; frozen
  protocol `pooled_k3` — every future retention adjudication pools THREE
  fresh screens and applies the +/-5 band to their mean (2 x 4.27/sqrt(3) =
  4.9). All five historical tax readings (-9, -10, -10, -7, -5) sit inside
  measured noise; pooled deltas are only -3.75 (axis160_direct), -2.25
  (hygiene_explore_direct), -0.75 (axis160_r64), -0.75 (replay_clean). The
  5-10-point per-dose tax law REVISES to 1-4 points pooled; the trade is
  cheaper than priced. Vehicle descriptive: r64 -0.75 vs r32 -3.75 (+3.0,
  within noise) — capacity relief is plausible but unadjudicated.
- Next funded branch (pick one, own intake): (a) vehicle resumption under
  pooled_k3 — adjudicate rank-64 vs rank-32 retention on three fresh pooled
  screens (eval-only if both arms stay published; decisive on the capacity
  question the rank cell paused); (b) the goal-gate path — a
  hygiene+explore-lesson dose from the best parent, gated under pooled_k3
  retention plus the standard axis instrument, then the MEDIUM-tier
  four-model pilot (candidate > base/replay/parent aggregate; every-family
  -vs-base recorded as the goal gate). The revised 1-4-point tax makes (b)
  cheaper than previously priced.
- Completed tier forensics (CONSTANTS_ARE_INSTRUMENT_ARTIFACTS):
  `qwen35_4b_menders_sirens_tier_forensics` — zero-GPU receipt analysis; the
  menders/sirens constants have three committed quick/tb1024 counterexamples
  and the all-ten-families goal gate passed 9/94 historical medium
  arm-events vs 1/84 quick (base never at a family ceiling at medium). The
  goal gate's venue is medium; a same-shape menders/sirens treatment at
  quick is dead. Caveat: medium passers were gym-trained; the universal
  line is unmeasured at medium.
- Completed medium measurement (MEASUREMENT_READ_COMPLETE):
  `qwen35_4b_universal_medium_tier_measurement` — seed 78150, tb1024;
  hygiene_explore 0.3379 leads (quick ordering inverted; tier-Pareto
  non-convexity replicates in the line); all three treated arms 8/10 strict
  wins vs base; hygiene_explore and replay_repeat ZERO losses, ties only at
  menders and rites (both 0.0). The recorded goal gate is two tie-flips
  wide. Sirens resolved at medium as the forensics predicted; base inside
  the historical envelope everywhere.
- Completed two-tie install (NOT_PROMOTED, split):
  `qwen35_4b_feedback_loop_state_chain_install` — u_statechain INSTALLED
  (11/20 fresh-instance holdout, strict over parent 7 and replay 10) but
  u_feedloop failed completely (0/20 in-domain, below untrained controls):
  the THIRD failed pedagogy at the menders-shaped skill. Retention -3.0 vs
  parent (inside the tax law) but -5.67 vs replay (0.67 outside the
  calibrated band; replay itself gained +2.67). pooled_k3's first live use
  measured SD 4.08 vs calibration 4.27. Seed 78151 permanently sealed.
- REPAIR KILL RULE EXTENDED: asserted repair, demonstrated bounded search,
  and episode feedback-loop doses have ALL failed at the menders-shaped
  skill; no further small designed SFT dose of any pedagogy is believable
  there without a different mechanism CLASS (scale, external scaffolding,
  or non-SFT levers).
- Completed intermediate budget probe (second and FINAL BUDGET_GATE_STOP):
  `qwen35_4b_medium_intermediate_budget_probe` — base refused at tb4096
  exactly as at tb8192; per the frozen consequence the thinking-budget
  lever is CLOSED ENTIRELY for paired medium events (total cost: two
  seeds, zero exposed scores). Believable menders mechanisms remaining:
  dose SCALE (C43 precedent — the three failed attempts were all 80-160
  rows; an 800+-row episode-feedback corpus tests scale as the mechanism
  class) and on-policy episode training. Each needs its own intake.
- Completed budget probe (BUDGET_GATE_STOP):
  `qwen35_4b_medium_budget_probe_measurement` — the gateway's hard wall
  budget refused base at medium/tb8192 before any treated arm ran
  (base-first order; no score; seed 78152 spent per the preregistered
  stop). The 8x lever is closed; tb2048-4096 remains the lever's only
  believable range, with hygiene_explore (230 s at tb1024, the slowest
  arm) as likely to bind as base. ONE further intermediate probe max; a
  second stop closes the lever entirely.
- Completed statechain-only dose (PROMOTED + RITES_CONVERTED +
  PARENT_GOAL_GATE_PASS_RECORDED): `qwen35_4b_statechain_only_dose` —
  first pooled_k3 promotion (axis 21/40 strict; retention in-band); at
  sealed 78154 the candidate beat base+replay but not its parent
  (-0.017), its rites 0.300 vs 0.100/0.100 is the first
  local-install→family conversion, and hygiene_explore_parent recorded
  the FIRST 10/10 all-families goal-gate pass (menders 0.017, warren
  0.150 vs 0.100 — single-item margins). One seed ≠ a claim.
- QUEUED (zero-root rebuild, own intake — opened by the standalone
  directive's lineage walk): the hygiene_explore adapter chain warm-starts
  from the frozen C53-era `blend` adapter (no committed creation receipt;
  gym-era training; vendored+hash-pinned in the confirmation cell). A
  zero-root rebuild — fresh adapter on the official base, the same six
  receipted stages replayed at their fixed seeds — answers how much of
  the recorded goal-gate pass the undocumented prefix is buying. It
  produces a DIFFERENT model (blend alone scored ~0.44 quick aggregate vs
  base 0.17), so it is a new experiment, not a reproduction; ~2.5-3h GPU.
- Completed confirmation (AGGREGATE_ONLY):
  `qwen35_4b_goal_gate_confirmation` — aggregate strict wins 3/3 fresh
  seeds (4/4 all-time, never close); goal gate 1/3 vs the 2/3 bar (78157
  swept 10/10 — two full sweeps across four independent seeds; 78155 9/10
  and 78156 8/10 with ZERO losses, blocked by menders 0-margin ties plus
  one warren tie). Demonstrated, not confirmed; the gate is localized to
  menders alone. First standalone-compliant cell (full lineage package).
- Completed dose-scale null (NOT_PROMOTED + DOSE_SCALE_NULL):
  `qwen35_4b_menders_dose_scale` — 10x dose, 8 formalisms: candidate 1/40
  on fresh instances = both untrained controls exactly (guess floor);
  retention failed the parent band (-5.33); dose curve flat at floor
  (0/20 at 80 rows, control-level at 800); seed 78158 sealed. MENDERS IS
  CLOSED TO SFT at every tested point (3 pedagogies x 80-800 rows x 4-8
  formalisms + budget levers). Remaining classes: on-policy episode
  training (own charter, heavy) or stand on demonstrated-not-confirmed.
- (superseded) FUNDED NEXT (menders dose-scale intake): the one permitted mechanism
  class — 800+ episode-feedback rows (10x the failed doses; C43: partial
  installs were data-limited) from the hygiene_explore parent, calibrated
  gates, success criterion known exactly: any reliable nonzero menders
  yield completes the all-families gate whose every other family now
  holds on every seed.
- (superseded) FUNDED NEXT — THE CONFIRMATION CELL (nothing outranks it): per the
  confirmation law and the goal's own text, hygiene_explore vs base (and
  a same-backend matched-compute sample-more baseline arm) on MULTIPLE
  independent fresh sealed medium seeds at tb1024; the goal gate recorded
  per seed; pre-registered replication bar (e.g. aggregate + strict
  all-ten on a majority of seeds, with the menders/warren single-item
  fragility priced explicitly). Eval-only, no training. A confirmed pass
  completes the program goal's primary condition; a non-replication
  prices the 78154 pass honestly as a draw and the program continues
  from the 9/10 position with the conversion mechanism in hand.
- Queued (calibration notes attached): (a) replay-compounding line — measure
  whether iterated replay rounds keep gaining aggregate and where they saturate;
  the published `replay_repeat` composite (0.5081) is the presumptive parent;
  cheap, believable, but cannot flip menders/sirens on current evidence.
  (b) menders/sirens forensics — explain the two frozen constants from public
  metadata and score behavior only (e.g., whether sirens' 0.500 is a fixed
  solvable/unsolvable item split at tb1024 and whether any config has ever
  scored menders > 0 at quick), BEFORE designing another treatment; two
  same-shape transfer attempts at menders (loomfix, tracefix) have failed, so a
  third is not believable without new mechanism evidence.
- Completed staged-search mechanism negative: `qwen35_4b_universal_search_scaffold_token_match` decomposed
  two-step search into five independently scored executable lesson stages before a
  bounded full-search ledger. It starts from the authenticated
  close-weight near-miss, trains a new same-parent replay continuation, and reserves
  fresh local seed 88007 and conditional aggregate seed 78137. CPU feasibility and
  adversarial review passed: both arms have 320 rows, exactly 286814 forward tokens,
  zero skips, 40 updates, and 200 identical replay positions. Train replay first,
  publish/verify it, then train the sole scaffold candidate; local failure seals the
  benchmark. On fresh seed 88007, parent/replay/scaffold scored 18/16/16 correct,
  all parsed 23/26, and all had three caps. Scaffold was 0/2 execute, 0/2 induct,
  and 0/2 probe, failed five gates, and seed 78137 remains sealed. Do not repeat
  canonical two-op/two-branch lessons. A successor must use a new directory and fresh
  seeds to test variable-depth natural-language state tables plus independent
  hypothesis simulation/scoring and verified answer commitment.

- Completed state-table mechanism negative:
  `qwen35_4b_universal_state_table_compiler_token_match` trained variable-depth
  natural-language state tables, independent hypothesis scoring, repair, and commit
  against same-parent exact-token replay. Fresh seed 88008 scored parent/replay/
  candidate 19/16/16 correct, 23/21/22 parsed, and 3/5/5 caps; target subtotals were
  4/2/1 of six. Candidate failed five absolute gates and every relative gate, so seed
  78138 remains sealed. Retire another idealized trace surface.
- Completed on-policy mechanism negative:
  `qwen35_4b_universal_on_policy_prefix_repair_token_match` froze 288
  fresh truth-audited tasks (48 each across six failure classes), an explicitly
  merged `close_xi` vLLM deployment, exact generated-token prefix masking, and ten
  reachable failures per class. Reserved construction/rollout/training/local/
  aggregate seeds are 77113/66113/47/88009/78139. Design receipt
  `98c6a168...5638` authorized the parent merge. That merge applied 128/128 nonzero
  LoRA modules and produced composite weight hash `4933f2dd...eb373`. The separately
  checkpointed parent event then completed 288/288 rollouts and 170252 sampled tokens
  at 849.9 tokens/s; rollout/receipt hashes are `8010632f...3b17f` /
  `c6b98b79...74fa`, with no generation rerun during wrapper recovery. Model-free
  mining found 230 reachable failures and cleared every fixed class quota with
  availability 46/48/35/24/36/41, selecting 10 each. Inventory/source hashes are
  `7230af52...dfe7` / `30141538...d84b8`; the selected prefixes contain 47123 masked
  tokens and are dominated by cap boundaries. The second review now freezes two
  320-row arms at exactly 304313 forward tokens, zero skips, 40 updates, and 200
  aligned replay positions; receipt hash is `eb08026f...e0cfc`. Candidate has 33421
  fewer target tokens and lower absolute loss mass than replay, a required causal
  caveat. From pushed-green commit `a8529c04`, replay trained 320/320 rows with zero
  skips and 40/40 updates; receipt/adapter hashes are `f78f2069...d6de` /
  `bb59d3bd...5154d`. After that checkpoint passed both workflows, the candidate
  independently trained 320/320 rows with zero skips and 40/40 updates; its
  receipt/adapter hashes are `846d8107...7098` / `85811191...0f14`. Publish and
  verify paired training before freezing the fresh local design. Seed 88009 now
  freezes 26 truth-audited rows with source/input/receipt hashes
  `9682744e...acdee` / `ff407551...ce988` / `3982d5b8...6e85a`, zero overlap against
  training and prior reserved local messages, and an identical merged-composite vLLM
  protocol for all arms. Verdict `PASS_CONTROL_MERGE` authorizes control merge only;
  from pushed-green commit `6dc0e677`, that merge applied 128/128 nonzero modules.
  Its tracked receipt, external receipt, and full-weight hashes are
  `bc78f332...d550` / `aa763255...45a3` / `7ab4c419...6e2e`. From the resulting
  pushed-green commit `619f1e53`, the candidate merge also applied 128/128 nonzero
  modules; its tracked receipt, external receipt, and full-weight hashes are
  `3deff026...438d` / `baa2027e...6d5a` / `376e2082...b528`. Fresh seed 88009 then
  scored parent/replay/candidate 16/18/15 correct, 24/23/23 parsed, 2/3/3 caps, and
  2/1/0 of six on execute+induct+probe. Candidate failed six absolute and all four
  relative checks; it won one task and lost four versus replay, with no per-skill
  count improvement. Local/promotion hashes are `b4b333ca...b8c8` /
  `1e048e75...f5c`; aggregate seed 78139 remains sealed. Retire long masked
  failure-prefix continuation. A successor must move to short pre-failure decision
  interventions and match supervised target exposure.

- Completed counterfactual-restart mechanism negative:
  `qwen35_4b_universal_failure_selected_restart_target_match` uses the published
  stronger replay composite on 624 fresh tasks, four prospective failures per each
  of 13 universal skills, then removes the parent's failed trajectory and supervises
  a clean executable restart from the original prompt. Construction/rollout/
  selection/training/local/aggregate seeds are 77114/66114/55114/48/88010/78140.
  Source/input hashes are `81edc9ea...de304` / `25382689...0f5b`; zero prompt
  overlap was found against predecessor sources and prior local seeds. The design
  requires exact equality of forward tokens, loss-bearing targets, and absolute loss
  mass against same-parent replay. `PASS_PARENT_ROLLOUT` authorizes one parent event
  only. From pushed-green commit `1744e753`, that event completed 624/624 rows and
  304013 sampled tokens at 879.9 tok/s with no recovery or rerun. Rollout/receipt
  hashes are `4bf15134...1099f` / `1d35c63a...2b381`; no benchmark data was read.
  After that collection was published green, model-free selection found 602 eligible
  and 228 hard-failure rows, cleared all 13 quotas, and froze 52 restarts: 40 hard
  failures plus 12 budget-only cases. Inventory/restart/selection/summary hashes are
  `c19d3de7...66240` / `022b1ea4...d951f` / `567d6b02...b662` /
  `2e8a2192...e28ddf`. The self-contained replay and inherited partition hashes are
  `25a9595f...f0c2` / `abf8b505...0966f`. Exact integral
  matching then passed: both 320-row arms have 297731 forward tokens, 126796
  loss-bearing targets, absolute loss mass 27632.8, zero skips, and 200 aligned shared
  rows. Manifest/control/candidate/final-receipt hashes are `7ba55045...91de1` /
  `7a8d4566...b5078` / `28deb20e...3190` / `52a761ef...170`. Second-review verdict
  `PASS_CONTROL_TRAINING` authorizes replay control only after this freeze is
  published green. From pushed-green commit `821d50d4`, replay control then trained
  320/320 rows with zero skips and 40/40 updates; receipt/log/adapter hashes are
  `3a9cc1ea...6d49` / `3bedc341...f25` / `5840757d...b1c`. Publish this control
  checkpoint before the independently warm-started candidate. From pushed-green
  commit `2c78e655`, that candidate trained 320/320 rows with zero skips and 40/40
  updates; receipt/log/adapter hashes are `6aa5c3f1...9871` /
  `c8572c88...202a` / `2072c5c8...39bc`. Publish paired training, then freeze
  explicit merges and fresh-local evaluation. Both exact seven-file composite trees
  authenticated. Fresh seed 88010 then scored parent/replay/candidate 17/16/15
  correct, 21/22/25 parsed, 5/4/1 caps, and 2/2/0 of six on
  execute+induct+probe. Candidate failed its accuracy and all three target floors
  plus all four relative checks; local/promotion hashes are `39fe68b9...de9e` /
  `4c381fbd...6759`, and aggregate seed 78140 remains sealed. Retire another
  hand-authored oracle restart surface: it improved termination while losing semantic
  target competence.
- Completed successful-sibling prerequisite stop:
  `qwen35_4b_universal_successful_sibling_target_match` froze 624 fresh tasks and one
  authenticated greedy event before any sibling sampling. From green commit
  `0038fba1`, the parent completed 624/624 rows and 296259 sampled tokens at 859.6
  tok/s with no recovery or rerun; raw/receipt hashes are `e91313c0...f556` /
  `cee1f19d...4962`. Model-free grading then found 227 hard failures, but the
  prospective four-per-skill prerequisite failed: count and route supplied zero and
  select supplied two. Inventory/receipt hashes are `8e21caf8...d783` /
  `3397b773...2a6e`; no sibling input was emitted and every downstream seed remains
  sealed. This does not test sibling distillation. The next result-separated design
  may reuse the immutable collection, define the ten skills with at least four
  failures as the residual treatment set, sample only those failures, and protect
  saturated skills through exact-exposure replay plus the unchanged all-skill local
  gate. Do not lower this experiment's quota or rescue it in place.
- Completed residual successful-sibling availability stop:
  `qwen35_4b_universal_residual_successful_sibling_target_match` copied the immutable
  624-task source and 227-failure inventory, treated only the ten skills with at
  least four hard failures (225 oracle-free rows), and completed the one same-parent
  `n=16` event at seed 66117 from published-green commit `fc5a333b`: 225 prompts,
  3,600 outputs, 2,337,087 sampled tokens, 739.2 tok/s, no recovery or rerun
  (raw/receipt `688c4f7e...c332` / `c3a3a297...f614`). The frozen model-free
  selection from green checkpoint `915a7c62` qualified 855/3,600 siblings but only
  two of 46 induct failure tasks supplied any qualified sibling, below the four-task
  quota (all other skills ≥6). Outcome `STOP_INSUFFICIENT_SUCCESSFUL_SIBLINGS`;
  inventory/receipt hashes `60c95b7a...083e` / `d3926daf...ad01`; training/local/
  aggregate seeds 50/88012/78142 were never consumed. Read together with the
  balanced predecessor: same-parent successful-sibling mining is closed as a
  curriculum source — the residual skill that most needs repair (induct) is the one
  whose successes the parent cannot sample (736 samples, 2 supported tasks). Do not
  reopen with larger `n`, relaxed token ceilings, or a nine-skill treatment; the
  wall skill would be untreated by construction.

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

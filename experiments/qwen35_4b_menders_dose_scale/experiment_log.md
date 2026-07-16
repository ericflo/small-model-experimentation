# Qwen35 4b Menders Dose Scale Experiment Log

## Scaffold

Created as a new experiment scaffold.

## 2026-07-16 — Model-free design freeze (lifecycle 20)

- Opened as the dose-SCALE cell at the last blocking family: nine families
  hold vs base on every sealed seed and menders alone gates the goal
  (0-margin ties). Three small-dose pedagogies failed there; dose scale
  (C43: partial installs were data-limited) is the one permitted mechanism
  class. Dose: 800 u_feedloop episode rows — 10x the reference cell's
  failed 80-row dose (0/20 on fresh instances).
- Frozen: 800-row corpus (080c3603…, construction seed 77,150) on EIGHT
  legality-bounded formalisms — troughline/trinketcord/crankwheel/
  sigilslate reused as fresh instances (zero row-overlap receipts vs 36
  pinned predecessor sources) + four new ones (barrowyoke, balesled,
  millround, skeinreel; fresh-surface grep audit, 47 claimed tokens, zero
  hits). All reviewed episode invariants kept (>=2 round-1 candidates with
  the wrong attempt among them, unique after rounds 1+2, extended-grammar
  exclusion audit; banned vocabulary extended with the statechain cells'
  surface pools, retaining only the reused feedloop formalisms' nouns).
- Exact zero-delta 3-axis MILP at namespace seed 55,140: 1,280 shared core
  + 1,000-row variable blocks; 2,280 rows/arm, 285 updates (1,878,709 /
  771,405 / 867,281 per arm; zero skips; 1,280 position-aligned rows).
  POOL BIND recorded honestly: the predecessor formulation (pairwise
  disjoint blocks over the 960 non-core rows) is arithmetically infeasible
  at this dose — the pool is 40 rows short AND the treatment's answer-token
  mass (mass-minus-nonzero = 4x answer, an encoder identity) is unreachable
  from the non-core rows alone. The frozen geometry is kept by letting the
  control block draw from the full pool under an ARM-LEVEL multiplicity cap
  of 2; the MILP minimizes repeats (575, solver-proven optimal at gap 0)
  and the candidate arm stays duplicate-free. Documented in the stream
  manifest, independently re-validated, and unit-tested.
- Local gate frozen: axis holdout 88,037 (40 u_feedloop, 5 per formalism)
  + three 104-row pooled_k3 retention screens 88,038/88,039/88,040
  (canonical generator 70fc722a…); overlap receipts vs all prior gates
  88,013-88,036, all corpora and streams (this cell's and predecessors'),
  and each other; gen_local_gate --check green twice. Promotion: axis
  total strictly > parent AND > replay_ctl3, pooled bands (-15/+9/-9 on
  sums) vs BOTH controls. Preregistered NON-GATING dose-response reading
  vs the frozen 0/20 baseline (reference promotion receipt sha d232a1be…),
  rendered per formalism, both consequence statements recorded either way.
- Conditional benchmark frozen: medium/tb1024, ONE sealed fresh seed
  78,158, four arms (base b654e033…, parent 9eb653d7…, replay_ctl3,
  feedloop_scale), receipt-pinned closed-ledger runner (the closed record
  sha-pins the summary AND all four gateway receipts; unopened events
  demand a clean slate; crashed summaries reconcile byte-identically —
  the confirmation cell's fix class, now standard). Power statement:
  menders > 0 for the candidate is the reading of consequence; any 10/10
  feeds a fresh confirmation cell before any claim.
- Standalone lineage package: the confirmation cell's six-stage package
  copied byte-identically (6 datasets + 3 trainers + merger + vendored
  root ad2ef4fa…/cd764ae8… under this cell's own large_artifacts) and
  EXTENDED with stage 7 (the candidate's training; dataset =
  feedloop_scale.jsonl 3aee5f5e…; produced pins are post-training
  TODO-PINs with explicit pending markers; the GPU rebuild refuses while
  pending). rebuild_lineage.py --verify-inputs green (7 datasets, 4
  trainers, merger, 6 root files) and wired into smoke.
- Seeds: 77150/55140/88037/88038/88039/88040/78158 verified grep-fresh in
  seed contexts (zero hits). Training seed 71 verified fresh in the
  qwen35_4b lineage's training-seed contexts; its only grep hits are run
  artifacts of the retired sparse_support_memory_executor track and one
  unrelated scorer test's run_seed — the same artifact class the seed-67
  audit accepted — recorded in the design receipt rather than substituted.
  No substitution required anywhere.
- 152 tests green; run.py --smoke green; boundary drills refuse without
  verdicts (train/merge/local/benchmark all fail closed on missing
  reviews, unpinned TODO-PINs, and dirty git). No GPU stage has run; no
  model has been loaded.

## 2026-07-16 — Four-lens review: multiplicity deviation ACCEPTABLE; four minors fixed

- The review adjudicated the control-arm multiplicity deviation ACCEPTABLE
  (all four lenses independently; solver-proven minimal; parent-anchored
  gates prevent false promotion). Zero majors. Four minors fixed:
  1. BIAS DIRECTION stated explicitly where the deviation is documented
     (stream manifest ``row_duplication.residual_bias_direction``,
     preregistration, README): repetition plausibly DEFLATES the replay
     control slightly, making candidate-vs-replay comparisons marginally
     easier; the parent-anchored bars bind independently and are
     unaffected; the retention band vs replay is conservative in the
     direction that costs the candidate nothing.
  2. DOSE x DIVERSITY CONFOUND written into both frozen consequence
     statements (check_local.DOSE_RESPONSE_CONSEQUENCES, preregistration,
     README, site brief): a nonzero reading is evidence that
     SCALE-PLUS-DIVERSITY reopens the family (10x dose the dominant
     delta; formalisms doubled 4->8 with it — not a pure dose-response
     isolate); a zero at 10x closes the scale class AND the
     added-diversity variant together. Reading stays non-gating.
  3. EXTENDED-GRAMMAR AUDIT: chose the preferred option — the audit now
     PROBES the container dimension for troughline and barrowyoke over
     the full module pools via a tolerant probe apply (phantom containers
     start empty; an op touching one can never reproduce the wanted
     state; probe apply verified equal to the bounded apply on every
     bounded op, per row). The probe scope is now stated exactly per
     formalism (EXTENDED_PROBE_SCOPE, recorded row-by-row in the audit
     and in the corpus manifest; sigilslate's slot indices are declared
     structural and NOT probed past 4 — the old "items over the full
     pools" wording is gone). Corpus bytes UNCHANGED (080c3603… —
     regeneration verified byte-identical; the probe adds enumeration,
     not rng draws); gate runner inputs and retention sources unchanged;
     only the axis SOURCE file changed (it embeds the per-row audit).
  4. DEAD CODE: run.py local_stage's unreachable check_local --out
     recovery branch REMOVED; the stage now requires both receipts after
     the eval and re-adjudicates verify-only; the real post-crash
     recovery path (manual ``check_local.py <local_receipt> --out
     <promotion>``) is documented in the stage docstring.
- Receipt regeneration cascade (--check green twice where applicable):
  corpus manifest 5617c2c5… (was f354f2f9…; corpus 080c3603… unchanged),
  stream manifest 5738f0b8… (was 37080454…; both streams byte-identical:
  02275b95…/3aee5f5e…), stream token receipt 46ae4cf6… (was 6d1377c3…;
  exposure numbers unchanged), axis source local_tasks_seed88037 5ec590bf…
  (was 2547dec1…; all runner inputs + retention sources byte-identical),
  local design receipt 7ac40653… (was a1a8d969…), design receipt
  cce84ff4… (was a1f90e93…). train_trial/run.py/check_design pins
  updated accordingly.
- 154 tests green (two new container-probe/scope suites); run.py --smoke
  green; boundary drills re-verified. Still no GPU stage, no commit.

## 2026-07-16 — Adversarial review: deviation adjudicated, minors fixed, freeze

- Four lenses, zero blockers/majors; all four independently accepted the
  pool-bind deviation (control multiplicity 2, solver-proven minimal,
  parent-anchored gates unaffected). Six minors fixed across two rounds:
  bias direction stated, dose×diversity confound acknowledged in both
  frozen consequences, container-dimension probe extended with per-row
  scope records (corpus bytes unchanged), dead recovery branch removed.
- 154 tests green; smoke green; receipts regenerated (--check twice);
  PASS_EXPENSIVE_RUN and PASS_CONTROL_TRAINING granted.

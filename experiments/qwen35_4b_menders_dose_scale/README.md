# Qwen35 4b Menders Dose Scale

Lifecycle 20 — the dose-SCALE cell aimed at the last blocking family. Nine benchmark families now hold vs base on every sealed seed; menders alone gates the all-families goal (0-margin ties). Three small-dose pedagogies failed at it; the scale hypothesis (C43: partial installs were data-limited) is the one permitted mechanism class. The dose: 800 episode-feedback rows — 10x the reference cell's failed 80-row dose (u_feedloop scored 0/20 on fresh instances there).

**Status:** in-progress · since 2026-07-16 · design frozen and reviewed (zero majors; the pool-bind deviation adjudicated acceptable by all four lenses); training, the calibrated gates, and the conditional sealed-seed event remain

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the reference cell `qwen35_4b_feedback_loop_state_chain_install` (lifecycle 15: u_feedloop dead at 0/20 inside an 80-row mixed dose); `qwen35_4b_statechain_only_dose` (lifecycle 18: the proven sibling skill promoted locally and converted rites; its event recorded the parent's first 10/10); `qwen35_4b_goal_gate_confirmation` (the parent's 10/10 confirmed on three sealed seeds — menders margins single-item thin); menders closed for three small-dose pedagogies + the budget lever.

## Question

Does the feedback-loop episode lesson install at 800 rows (10x the failed dose) — and even short of promotion, does ANY fresh-instance u_feedloop transfer appear at 10x? That dose-response reading is preregistered and non-gating, and it carries a stated DOSE x DIVERSITY confound (formalism diversity doubled 4→8 alongside the 10x dose): nonzero at 10x is evidence that scale-plus-diversity reopens the family (the 10x dose is the dominant delta, not a pure dose-response isolate); a 0 at 10x closes the dose-scale mechanism class AND the added-diversity variant together for this skill.

## Hypothesis

C43 (meta-induction bankability): partial installs were data-limited — the shift-induction skill moved 0.087→0.40 with dose and plateaued only below the execute ceiling. The feedback-loop lesson at 80 rows inside a mixed dose taught nothing measurable; if the C43 mechanism transfers, 10x the rows on eight formalisms (double the surface diversity) should move fresh-instance transfer off zero, and an installed act→observe→revise loop is the program's best-calibrated bet at the menders family.

## Setup

- Parent and adapter base: the `hygiene_explore` composite (tree 9eb653d7…), fresh rank-32/alpha-64 adapters, no warm start, training seed 71.
- Corpus: `data/sft_feedloop_scale.jsonl` (080c3603…), 800 rows, construction seed 77,150, eight formalisms x 100 — troughline/trinketcord/crankwheel/sigilslate reused from the reference cell as FRESH instances (zero row-overlap receipts) plus four NEW legality-bounded formalisms (barrowyoke, balesled, millround, skeinreel). Every reviewed invariant kept: >=2 legal fix candidates after round-1 evidence (the wrong attempt among them), exactly 1 after rounds 1+2, extended-grammar exclusion audit with a per-formalism probe scope recorded row-by-row (numeric parameters to 12; item parameters over the full pools; the two named-container machines — troughline and barrowyoke — additionally probe every op's CONTAINER dimension over the full pool via a tolerant probe apply where phantom containers start empty; out-of-bound alternatives excluded only by the rendered legality clause), think targets quantifying over legal steps, easy repairs (the lesson is the loop). Banned vocabulary extended with the statechain cells' surface pools; fresh-surface grep audit + zero row-overlap receipts vs 36 pinned predecessor sources.
- Arms: `replay_ctl3` (control, trains FIRST) and `feedloop_scale` (candidate).
- Exposure: exact zero-delta 3-axis MILP at namespace seed 55,140 — 1,280 shared replay core + a 1,000-row variable block per arm (control: 1,000 replay slots; candidate: 800 treatment + 200 replay fillers); 2,280 rows/arm, 285 optimizer updates at accum 8 (1,878,709 forward tokens, 771,405 targets, 867,281 mass x5 per arm; zero encoder skips). POOL BIND (documented in the stream manifest): the 2,240-row replay pool cannot fill 1,280 core + 1,000 distinct control rows, and the treatment's long answer spans are unreachable from the 960 non-core rows alone, so the control block draws from the full pool under an ARM-LEVEL multiplicity cap of 2 (575 solver-minimized repeats; no replay row is seen more than twice in the control arm's epoch; the candidate arm is duplicate-free). RESIDUAL BIAS DIRECTION (stated in the manifest and preregistration): repetition plausibly deflates the replay control slightly, making candidate-vs-replay comparisons marginally easier; the parent-anchored bars bind independently and are unaffected, and the retention band vs replay is conservative in the direction that costs the candidate nothing.
- Local gate: axis holdout 88,037 (40 u_feedloop, 5 per formalism; strict TOTAL over both controls, no per-kind split — single-kind dose) + retention pooled over screens 88,038/88,039/88,040 under pooled_k3 bands on pooled sums (correct >= -15, caps <= +9, parsed >= -9 vs BOTH controls; i.e. +-5/3/3 on means). PLUS the preregistered NON-GATING dose-response reading vs the reference cell's frozen 0/20 baseline, rendered per formalism, both consequence statements recorded either way.
- Conditional benchmark (only on promotion): medium, tb1024, ONE sealed fresh seed 78,158, four models (base, parent, replay_ctl3, feedloop_scale), hardened seed-boundary runner with the receipt-pinned closed-ledger pattern (the closed record sha-pins the summary AND all four gateway receipts). Pilot gate = candidate aggregate strictly > base AND > replay_ctl3 AND > parent; goal gate recorded either way; frozen power statement: menders > 0 for the candidate on this seed is the reading of consequence; any 10/10 feeds a fresh confirmation cell before any claim.
- Standalone lineage: the confirmation cell's six-stage hygiene_explore package copied byte-identically (datasets, manifest, three trainers, merger, vendored root adapter ad2ef4fa…/cd764ae8…) and EXTENDED with this cell's stage 7 (the candidate's own training; produced pins are post-training TODO-PINs); `scripts/rebuild_lineage.py --verify-inputs` is wired into smoke.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_menders_dose_scale/scripts/run.py --smoke
```

Full (one stage per pushed checkpoint, each behind its review verdict):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_menders_dose_scale/scripts/run.py --stage train-control
# then: train-candidate, merge-arms, local, benchmark
```

## Results

The staged runs have not started: control training, candidate training, merges, the eight-formalism holdout plus pooled_k3 retention gate, the preregistered non-gating dose-response reading against the 80-row cell's 0/20, and the conditional medium event at sealed seed 78,158 all remain. Separate deployable evidence from oracle/hidden evaluation. The dose-response reading (candidate u_feedloop axis total vs the frozen 0/20 baseline, per formalism) is recorded either way and never feeds the promotion verdict.

## Interpretation

Pending. The frozen consequences: any nonzero fresh-instance transfer at 10x reopens the last family with a scale-plus-diversity reading; a zero closes the dose-scale mechanism class and the added-diversity variant together, leaving the zero-root rebuild and on-policy episode training as the remaining intakes.

## Knowledgebase Update

- Program evidence updated:
- Program backlog updated:
- Claim ledger updated:

## Artifacts

- `scripts/` — generators, builders, exact-exposure solver/validator, gate, trainers' wrappers, merger copy, benchmark runner, lineage rebuilder
- `data/` — frozen corpus + manifest, exact-exposure streams + receipts, four gate input files + local design receipt, design receipt, lineage package
- `runs/` — staged GPU outputs (training/merges/local/benchmark; none yet)
- `reports/artifact_manifest.yaml`

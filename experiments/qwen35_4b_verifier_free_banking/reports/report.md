# Qwen3.5-4B: Can Confidence Replace the Verifier in the Banking Flywheel? Report

## Summary

Split verdict, and the split is the finding. At the **capability seat: no** — in the one eval cell
where execution-verified banking produced a real gain (depth-2 think greedy 0.08 → 0.24, CI
[+0.04, +0.32]; a post-hoc cell — no pre-registered cell cleared the bar), the pre-registered
trichotomy RULE classifies confidence-filtered banking exactly on the no-filter floor (both
+0.04): **conf~rand**, despite the confidence arm being ~15× purer than random (0.43 vs 0.03). At the **judge seat: yes, with a caveat** — think-judge discrimination fully
survives being trained on its own approvals (fixed-set within-task AUROC 0.872 → 0.883), but
P(True) *scores* inflate on the model's own post-banking distribution (P(True) on own incorrect
candidates 0.091 → 0.204, doubled). A flywheel that filters by **rank** keeps its filter at round
2; one that filters by a **fixed threshold** silently degrades. Two upstream findings condition
everything: the no-think P(True) judge is at *chance* within-task on this substrate (C46's law is
substrate-scoped), and CoT judging rescues it (0.47 → 0.85 within-task) — verification of
computational correctness is itself a serial computation (C44's law reaches the judge seat).

## Research Program Fit

`posttraining_and_adaptation` × `evidence_conditioned_selection`: first experiment to connect the
banking arc (C11–C24: self-training needs an executable verifier at every rung) to the confidence
arc (C40–C46: a calibrated verification-free judge exists). The connection fails at the training
seat and succeeds at the judging seat, which cleanly relocates where the verifier is load-bearing.

## Method

Shared candidate pool: 90 procedural list-DSL identification tasks (depths 1/2/3 = 20/35/35), K=40
think-mode samples each → 2,130 unique candidates, oracle-graded (unique-candidate pass rate 0.033)
and judged twice: no-think P(True) at harvest, think-P(True) (`judge_pool_think.py`, budget 512)
after the gate-forced pivot. Four matched-size arms (n=70 pairs, hard trim; identical C18 QLoRA
recipe r32/α64, 3 epochs, lr 2e-4, no-think prompt→code; the ONLY variable is the keep-test):
`exec` (execution-verified, C18-identical keep rule — the ceiling, purity 1.0, depth mix 43/20/7),
`conf_strat` (PRIMARY verifier-free arm: depth-stratified top think-P(True), per-depth quotas ∝
judge-score MASS, purity 0.429, mix 15/30/25), `conf_global` (ablation: naive global top-score,
purity 0.614, mix 36/29/5), `rand` (draw-frequency-weighted floor, purity 0.029). Frozen paired
eval (75 held-out tasks, behavioral func-sig + op-composition dedup, 0 leakage), 25/depth, K=16,
greedy@1 + coverage. Pre-registered run/stop gate and trichotomy decision rule; paired-bootstrap
CIs on every delta; calibration survival on a fixed judge set (777 identical base-generated
candidates judged by every arm) plus a self-distribution pass (each model think-judges its OWN
K=8 eval-task candidates — the round-2 flywheel condition).

## Results

**1. The no-think judge reads difficulty, not correctness, on this substrate (gate stop).** Pooled
no-think P(True) AUROC 0.749 but **within-task 0.471 = chance** (mean P(True) by depth
0.38/0.25/0.20 merely tracks solvability). C46's within-problem discrimination (0.74 MBPP, 0.78
HumanEval) does not transfer to a substrate where correctness is only *computable* (mentally
execute the candidate on 8 examples), not *semantically readable* from a docstring.

**2. Serial compute rescues the judge.** `judge_think` (CoT before the same A/B verdict token)
lifts within-task AUROC **0.471 → 0.845** on the full pool (pooled 0.961; diagnostic subset 0.49 →
0.802), P(True) correct/incorrect 0.61/0.08 full-pool (0.58/0.10 on the diagnostic subset) — with
thinking budget-truncated at 512 tokens in 99% of judgments. C44's serial-compute law governs the judge seat: verifying a computational candidate
IS a serial computation. This is what made the experiment runnable verifier-free: the conf arms
rank on think-P(True); the gate passed (pool AUROC 0.961, purity gap 0.400).

**3. MAIN — trichotomy verdict: conf~rand; purity is not the binding constraint.** No
pre-registered cell cleared the exec-gain ≥ 0.10 bar (see Result 4). The only significant
pre-registered exec effect is depth-2 think cov_frac +0.035 (CI [+0.003, +0.072]) — concentration
of sample mass on already-covered tasks, not expansion. In the post-hoc cell (labeled `.POSTHOC`
in `runs/verdict.json`; added after seeing raw evals) exec moved hard — depth-2 think greedy 0.08
→ 0.24 (CI [+0.04, +0.32], p=0.011):

| arm | Δ vs base (d2 think greedy) | purity | role |
|---|---|---|---|
| exec | **+0.160** (CI [+0.04, +0.32]) | 1.000 | ceiling |
| conf_strat | +0.040 (n.s.) | 0.429 | **conf~rand** |
| rand | +0.040 (n.s.) | 0.029 | floor |

Recovery ratio (conf−base)/(exec−base) = 0.25, joint-bootstrap CI [−1.0, +1.0] (uninformative at
n=25). The ~15× purity advantage bought zero deployable gain — only the 100%-pure arm moved.
Banking is far less noise-tolerant than selection (C41/C46's seat): 40% confidently-wrong pairs —
plausible near-misses, by construction of the filter — erase what 100%-pure data teaches.
Suggestive (multiple-comparison caveat): conf_strat is the only arm that did NOT collapse depth-1
no-think coverage (Result 6) and shows small d1 think cov_frac wins over both rand (+0.110, CI
[+0.018, +0.207]) and exec (+0.090, CI [+0.005, +0.185]).

**4. The exec ceiling itself is dose-limited: C18's headline was a low-dose overestimate.** The
exec arm is the first **matched-dose** re-run of the C18 recipe (20 d2 pairs / 8 distinct d2
functions vs C18's 24 / 9) under the strict C24-standard eval — and C18's depth-2 coverage
expansion (0.15 → 0.45) is **flat here (0.28 → 0.28)**. Lineage audit: (a) C18's headline was 3/20
vs 9/20 (Fisher p=0.082) on a leak-permissive eval — exact-ops dedup only; 3/20 of its d2 eval
tasks were train-identical under the canonical probe-set func-sig (the same criterion the frozen
eval dedups on), leak-excluded 2/17 vs 7/17 (p=0.118); (b) the expansion
effect is real but was re-established at 3.5× the d2 dose — C22 0.225 → 0.425 (n=40), C23 0.175 →
0.50 (n=40, p=0.004, 83 pairs / 26 distinct functions); (c) under C24's diversity dose law, 8–9
distinct d2 functions sit at the bottom of the curve where a null at n=25 is expected. Consequence:
at matched dose the exec ceiling is intrinsically small, so the trichotomy had little headroom —
conf~rand holds in the one cell that cleared the bar, with n=25 power caveats.

**5. Calibration survives banking as a RANKER; scores inflate on the self-distribution.** Fixed
judge set (777 identical candidates, 47 correct; think judge = the flywheel's filter):

| model | within-AUROC | pooled | P(True)\|correct | P(True)\|incorrect |
|---|---|---|---|---|
| base | 0.872 | 0.961 | 0.613 | 0.098 |
| exec-banked | 0.873 | 0.962 | 0.708 | 0.149 |
| conf_strat-banked | 0.883 | 0.957 | 0.682 | 0.141 |

Self-distribution (own think-mode K=8 eval-task candidates — the round-2 harvest):

| model | within-AUROC | pooled | P(True)\|incorrect | own pass-rate |
|---|---|---|---|---|
| base | 0.842 | 0.930 | 0.091 | 0.120 |
| conf_strat-banked | 0.878 | 0.923 | **0.204** | 0.093 |

Discrimination is untouched — even for the arm trained on its own judge approvals; the
self-reinforcement collapse hypothesis (falsifier c) is dead. But P(True) on own incorrect
candidates **doubles** (fixed-set drift only +0.043 → the inflation is distribution-specific).

**6. Banking reallocates the no-think proposal distribution; CoT shields the think one.** Banked
arms collapse depth-1 no-think coverage@16 (base 0.72 → exec 0.36 / conf_global 0.32 / rand 0.40;
conf_strat 0.76 does NOT collapse) while greedy stays flat. Per-task paired forensics: **zero-sum
mode reallocation, not capability loss** — correct d1 samples per 400 are conserved across arms
(126/115/130/106/108) while covered-task count crashes 18 → 9/8/10; the same correct mass gets
crammed onto the banked program family. All four banks' d1 pairs come exclusively from the
arithmetic-map/reorder/slice families and 0% from dedup_adjacent/unique_stable; exactly that
family goes from 9/10 tasks covered (66 correct samples) at base to 0/10 (0) under
exec/conf_global. Dose-dependent (exec
banks 43 d1 pairs → collapse; conf_strat only 15 via its d3-heavy score-mass quotas →
sub-threshold, 7/10 dedup tasks survive) and correctness-dependent (rand collapses by the other
route: mass shifts toward wrong programs, diversity retained). Training pairs are NO-THINK, so the
sharpening lives in the immediate-answer prior; **CoT re-derives the answer and shields the think
distribution** (think-mode unique-programs unchanged, 7.7 → 7.8–9.0). Registered prediction
(logged before running): the missing conf_global think eval should show d1 cov_any ≈ 0.80–0.88
with residual losses on dedup tasks, vs its 0.32 no-think collapse. **Test result: 3/3 — d1
cov_any 0.88 (top of the predicted band), uniq 8.6 (predicted 7–9), and all three residual
lost-vs-base tasks are exactly the dedup/unique family (`unique_stable`, `dedup_adjacent`,
`unique_stable`).** The out-of-sample test confirms the mechanism.

## Controls

- `rand` is draw-frequency-weighted (what "bank with no filter" means); uniform-over-unique would
  deflate the floor. Matched size is a hard trim (C23 count-confound), matched optimizer steps.
- Score-mass quotas for conf_strat exist because candidate-count quotas provably allocate slots
  where wrong candidates explode (d2: 915 wrong / 20 correct; d3: 1066 / 7) — the attempt-2 lesson.
- The fixed judge set isolates judge-change from candidate-change (every arm judges the identical
  777 candidates); the inflation headline is P(True)-on-INCORRECT drift, since overall mean drift
  confounds inflation with real ability gain.
- No-think judge secondary control: its within-task AUROC *improves* after banking (0.679 →
  0.71–0.82: exec 0.815, rand 0.800, conf_strat 0.792, conf_global 0.713) in every arm
  **including rand** (97% wrong pairs) — a generic task-domain-SFT effect
  on the readout, not learning of correctness. (It also shows the no-think judge is not uniformly
  at chance: on the cruder no-think error distribution it is partially informative, 0.679; on the
  plausible think-harvest near-misses it is chance, 0.471.)

## Oracle Versus Deployable Evidence

Oracle (`full_pass`) appears ONLY in: exec's keep-test (the ceiling arm by design), the
experiment-level gate, and post-hoc purity/calibration REPORTING. The conf arms' selection reads
nothing but the model's own think-P(True) logit + generator metadata (depth, draw frequency);
matching size to exec leaks one scalar (documented design constant). The deployable recipe this
experiment licenses: think-judge rank filtering (no oracle) — but for training data, its 0.43
purity was not enough to matter at this dose.

## Interpretation

The verifier is load-bearing at the TRAINING seat, not the JUDGING seat. Confidence transfers
across seats asymmetrically: as a *selector* (C41/C46) P(True) approaches execution; as a
*training filter* it fails not because the judge ranks badly (0.845 within-task) but because
banking tolerates almost no confidently-wrong data at this dose — and the wrongs a confidence
filter admits are precisely the plausible ones. Meanwhile the judge itself is robust to
self-training (ranks survive, scores inflate), so the flywheel's failure mode is not feedback
collapse but simple dose: at 70 pairs the exec ceiling is already at the bottom of C24's diversity
dose curve, and a 0.43-pure filter recovers none of it. Three laws sharpen: C46 is
substrate-scoped (single-token readouts work where verification is shallow; computational
correctness needs serial compute — C44 governs the judge seat too); C24's dose law now has a
matched-dose null confirming its bottom end (and retro-explains C18 as a low-dose overestimate on
a leak-permissive eval); and the banking arc gains a distribution rule — no-think SFT reallocates
the no-think proposal prior toward banked families while CoT shields the think distribution, so
flywheel harvests (think-mode) survive banking that would look catastrophic no-think.

## Next Experiments

1. **Scale the harvest, not the filter**: K=40 → K=160+ on more tasks, then take the top-rank
   slice at conf_strat's quota — does purity×diversity at larger dose cross the banking threshold
   verifier-free? (The direct test of "dose is the binding constraint".)
2. Threshold-vs-rank flywheel round 2 in anger: run one actual round-2 harvest+train with rank
   filtering (predicted fine) vs fixed-threshold filtering (predicted purity decay via the +0.113
   inflation).
3. Semantically-readable substrate (natural-language task descriptions, C37-style): does the
   NO-think judge's within-task discrimination return, making the cheap filter viable there?
4. Judge-budget dose-response: think-judge scores fell by depth (0.36/0.10/0.07 mean by depth) at
   budget 512, 99% forced-close — does budget 1024/2048 lift deep-candidate ranking?

## Artifact Manifest

`runs/verdict.json` (full stats incl. `.POSTHOC` labels), `runs/arms_summary.json` (gate + arm
composition), `runs/judge_think_diag.json` (Result 2 diagnostic), `runs/calib*.json` (Result 5:
fixed / think / self-think), `runs/eval_*.json` (10 arm×mode evals, 25 tasks/depth), `analysis/
verifier_free_banking.png`, `experiment_log.md` (full saga incl. two failed attempts and the
session crash), `reports/design_review.md` (pre-registration). Adapters (5 × 182 MB) moved out of
the repo — external paths + regenerate commands in `reports/artifact_manifest.yaml`.

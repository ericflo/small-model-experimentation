# Qwen3.5-4B: Can Confidence Replace the Verifier in the Banking Flywheel? Experiment Log

## 2026-07-07 design + review

Scaffolded from the C18 harness (harvest/train_lora) + C24 frozen-paired eval; the P(True) judge
(`judge_nothink`) was already present, unused, in every banking experiment's `gen_lib.py`. Adversarial
design review (reports/design_review.md): sound_with_fixes; all seven must-fixes applied (smoke/full
artifact separation, hard gate, stratified-conf primary arm, draw-weighted rand, matched-size hard trim,
self-distribution calibration, trichotomy decision rule). Two smoke passes green, second covering
adapter-loaded eval/calib and analyze.

## 2026-07-08 attempt 1: CUDA OOM (two stacked environment bugs)

Full harvest OOM'd at batch 48 fifteen minutes in. Causes: (1) launched without
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (mandatory per docs/compute_environment.md — C18 ran
this exact config WITH it); (2) torch 2.12 raises generation OOM as `torch.AcceleratorError`, which
gen_lib's six batch-halving catch sites (`except torch.cuda.OutOfMemoryError`) do not catch — the designed
graceful degradation became a hard crash. A third bug hid the failure: `cmd | grep | tee | tail` reports
the LAST pipe stage's exit code — the dead run looked like exit 0. Fixes: run.py sets the alloc env var
itself; gen_lib catches `OOM_ERRORS = (OutOfMemoryError, AcceleratorError)`; relaunches use `pipefail`.

## 2026-07-08 attempt 2: GATE STOP — the no-think judge reads difficulty, not correctness

Harvest succeeded (90 tasks × K=40 → 2,130 unique candidates). The pool is needle-in-haystack, unlike
MBPP: unique-candidate purity 0.033 (draws ~0.21) — wrong programs EXPLODE combinatorially at depths 2–3
(~1,000 distinct wrong each vs 20/7 correct), while correct answers concentrate + duplicate at depth 1.

The pre-registered gate stopped the run before training, and its numbers contain the experiment's first
finding: **pooled P(True) AUROC 0.749 but WITHIN-task AUROC 0.471 — chance.** All apparent signal is
between-task difficulty (mean P(True) by depth 0.38/0.25/0.20 tracks solvability). The C46 within-problem
discrimination (0.74 on MBPP) does NOT transfer to this substrate: here correctness is only COMPUTABLE
(mentally execute the candidate on 8 examples), not semantically readable from a docstring. Also caught:
pool-candidate-count quotas for conf_strat allocate slots exactly where wrong candidates are most numerous
(perfect-ranker purity ceiling ≈ 0.44; actual 0.10); conf_global still concentrated purity 8× over random
(0.243 vs 0.029) on the between-task component alone.

## 2026-07-08 diagnostic: serial compute RESCUES the judge (0.49 → 0.80 within-task)

Hypothesis from C44 (induction is a serial-compute limit) applied to the JUDGE seat: verifying a candidate
transform requires serially executing it; a single no-think forward pass cannot. Test: `judge_think`
(CoT before the A/B verdict) on 524 pool candidates (all 70 correct + 6 wrong/task).
**RESULT: within-task AUROC 0.49 → 0.802; pooled 0.719 → 0.924; mean P(True) correct/incorrect
0.48/0.26 → 0.58/0.10.** Verification of computational correctness is itself a serial computation —
C44's law governs metacognition too. Notable: thinking was budget-truncated (forced-close) in 99% of
judgments and still reached 0.80 — the first 512 thinking tokens carry the decisive execution steps.
(Tooling: `judge_think` returns `(p_values, forced_flags)` — a tuple, not a list.)

## 2026-07-08 attempt 3: pipeline relaunched on the think-P(True) filter

Pivot codified: new stage `judge_pool_think.py` (think-judge the full pool); `build_arms.py` ranks conf
arms on `p_true_think`, and conf_strat quotas switch from candidate-count-proportional to
JUDGE-SCORE-MASS-proportional (the count version provably allocates slots where wrong candidates explode);
calibration survival now measured on the THINK judge (fixed set for base/exec/conf_strat; self-distribution
for base/conf_strat) since that is the filter the flywheel would actually use. Same gate, same trichotomy.
Gate PASSED: pool think-AUROC 0.961 pooled, purity gap conf_strat-rand 0.400; arms at matched n=70:
exec purity 1.0 {d1:43,d2:20,d3:7}, conf_strat 0.429 {15,30,25}, conf_global 0.614 {36,29,5}, rand 0.029.

## 2026-07-08 session crash mid-pipeline (harness, not experiment)

The CLI session died with repeated EACCES while the pipeline ran; the orphaned run.py SURVIVED (trains +
evals kept writing). No permission error appears anywhere in the run logs -- the crash was the harness's
own session files. Recovery: verify PID alive, re-arm a log monitor from the new session, continue.
Lesson: launch long pipelines so they don't die with the session (they did not -- subprocess detach was
sufficient), and make every stage idempotent so a relaunch resumes from disk (it is).

## 2026-07-08 eval grid complete: trichotomy = conf~rand in the only cell exec moved

NO pre-registered cell clears the exec-gain >= 0.10 bar. The only significant pre-registered exec effect
is d2 think cov_frac +0.035 CI[+0.003,+0.072] (concentration on already-covered tasks). POST-HOC cell
(labeled .POSTHOC in verdict.json -- added after seeing raw evals): d2 think greedy base 0.08 -> exec
0.24 (CI[+0.04,+0.32], p=0.011); there conf_strat +0.04 = rand +0.04 exactly -> trichotomy **conf~rand**.
Purity 0.43 (~15x rand) bought zero deployable gain; only the 100%-pure exec arm moved anything.
Suggestive (multiple-comparison caveat): conf_strat is the only arm not collapsing d1 no-think coverage
and shows small d1 cov_frac wins over BOTH rand (+0.110) and exec (+0.090) in think mode.

## 2026-07-08 forensics (read-only agents, while calibration runs)

**d1 no-think coverage collapse (0.72 -> 0.36 exec / 0.32 conf_global / 0.40 rand, but 0.76 conf_strat)
is zero-sum mode reallocation, not capability loss.** Correct d1 samples/400 are CONSERVED across arms
(126/115/130/106/108) while covered tasks crash 18 -> 9/8/10; correct mass gets crammed onto the banked
program family. All four banks' d1 pairs come exclusively from the map/reorder/slice families (square/negate/drop_k/
rotate/sort/reverse/mul_k/abs) and 0% dedup_adjacent/unique_stable -- and the dedup/unique family goes
9/10 tasks covered (66 correct samples) at base to 0/10 (0) under exec/conf_global. Dose x correctness:
exec banks 43 d1 pairs (collapse), conf_strat only 15 (sub-threshold, 7/10 dedup tasks survive), rand
collapses by the other route (97% wrong pairs shift mass off-target with diversity retained). Training
pairs are NO-THINK, so the sharpening lives in the immediate-answer prior; CoT re-derives and shields the
think distribution (think uniq unchanged 7.7 -> 7.8-9.0). Flywheel-relevant: round-2 harvest is
think-mode, i.e. on the SHIELDED distribution. REGISTERED PREDICTION before running the missing cell:
conf_global_think d1 cov_any ~0.80-0.88, uniq ~7-9, residual losses concentrated on dedup/unique tasks
(vs its 0.32 no-think collapse). Will run post-pipeline when the GPU frees.

**"C18 fails to replicate" is the WRONG frame -- it is C24's dose law.** Audit of the lineage: C18's
0.15->0.45 was 3/20 vs 9/20 (Fisher p=0.082, n=20) on a leak-permissive eval (exact-ops dedup only; 3/20
d2 eval tasks train-identical under the canonical probe-set func-sig; leak-excluded 2/17 -> 7/17 p=0.118). The d2 expansion DID
survive frozen leak-proof evals -- C22 0.225->0.425 (n=40) and decisively C23 0.175->0.50 (n=40, p=0.004)
-- but at 3.5x the d2 dose (83 pairs / 26 distinct functions vs C18's 24 / 9). Our exec arm is the first
MATCHED-DOSE re-run (20 pairs / 8 distinct d2 functions) under the strict eval: flat cov_any 0.28->0.28
is the bottom of the C24 dose curve, not an anomaly. Correct statement: C18's effect size was a low-dose
overestimate; cite against C23 + C24, not as a standalone replication failure. Consequence for THIS
experiment: at matched dose the exec ceiling is intrinsically small, so the trichotomy had little
headroom -- the conf~rand verdict holds in the one cell that cleared the bar, with n=25 power caveats.

## 2026-07-08 calibration complete: the judge survives banking as a RANKER

Fixed judge set (777 identical base no-think candidates, 47 correct): think-judge within-AUROC base
0.872 -> exec 0.873 / conf_strat 0.883 -- discrimination untouched, even for the arm trained on its
own approvals (falsifier c dead). Scores inflate mildly on the fixed set (P(True)|incorrect 0.098 ->
0.149/0.141) and HARD on the self-distribution (base 0.091 -> conf_strat 0.204, doubled; own
pass-rate 0.120 -> 0.093, no capability gain). Flywheel rule: rank filters survive round 2, fixed
thresholds silently rot. Secondary: the NO-think judge's within-task AUROC improves after banking in
EVERY arm including rand (0.679 -> 0.71-0.82) -- a generic task-domain-SFT readout effect, not
correctness learning; and it is not uniformly chance (0.679 on crude no-think errors vs 0.471 on
plausible think-harvest near-misses -- the failure is distribution-dependent).

## 2026-07-08 registered prediction test: 3/3 CONFIRMED

Post-pipeline, ran the held-back conf_global think eval against the pre-logged prediction
(d1 cov_any ~0.80-0.88, uniq ~7-9, residual losses on dedup/unique tasks): measured cov_any 0.88,
uniq 8.6, and ALL THREE residual lost-vs-base tasks are the dedup/unique family (unique_stable,
dedup_adjacent, unique_stable). The mode-reallocation + CoT-shield mechanism survives an
out-of-sample test. Final analyze covers 10 eval tags / 20 paired cells; verdict.json + figure
final. Claim C47 filed; C18 annotated with the eval audit; playbook updated; adapters (5 x 182MB)
moved to scratchpad/verifier_free_banking_artifacts/ per reports/artifact_manifest.yaml.

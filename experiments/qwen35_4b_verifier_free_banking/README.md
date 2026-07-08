# Qwen3.5-4B: Can Confidence Replace the Verifier in the Banking Flywheel?

## Research Program

- Program: `posttraining_and_adaptation` (× `evidence_conditioned_selection`)
- Program question: the banking arc (C11–C24) requires an executable verifier at every rung; the confidence arc (C40–C46) built a calibrated verification-free judge. Can the second power the first — extending self-training to domains with no interpreter?
- Prior anchors: C18 (`qwen35_4b_coverage_banking` — the harness and effect this replicates), C46 (`qwen35_4b_code_confidence` + HumanEval — P(True) is the program-level confidence), C11 (banking law), C29 (read-only verifier 2AFC 0.81), C24 (banking gain is diversity-driven).

## Question

Does banking the model's own **high-P(True)** solutions — with NO execution anywhere in the training pipeline — recover the capability gain of banking **execution-verified** solutions? And does the calibration signal itself SURVIVE being trained on (flywheel viability), or does self-training inflate P(True) / collapse the judge?

## Hypothesis

C46 says top-P(True) selection approaches visible-test execution per-candidate, so a top-fraction filter over a large harvest should yield a training set pure enough (~0.8+) for C18-style banking, whose gain is diversity- not purity-driven (C24). Falsifiers, each a distinct autopsy: (a) banking is less noise-tolerant than selection — confident-but-WRONG pairs teach the confidently-wrong modes; (b) the filter biases toward easy tasks, collapsing the diversity that drives banking; (c) training on own confident outputs inflates P(True) and destroys calibration (feedback collapse — kills round 2 even if round 1 works).

## Setup

- Model: Qwen3.5-4B, QLoRA r32/α64 (C18-identical: epochs 3, lr 2e-4, no-think prompt→code pairs).
- Dataset/task source: contamination-free procedural list-DSL identification tasks (`families.py`), 90 train tasks at depths 1/2/3 (C18 schedule), K=40 think-mode samples each.
- Train/eval split: frozen held-out eval set (`eval_ladder.py`), behavioral func-sig + op-composition dedup vs ALL train tasks, 0 leakage, paired across arms.
- Baseline: base model (no training) and `rand` (same-size, **draw-frequency-weighted** sample of the same pool — what "bank with no filter" actually means; uniform-over-unique would deflate the floor).
- Arms (all filter the SAME candidate pool; matched size is enforced HARD = matched optimizer steps; identical training recipe; the ONLY variable is the keep-test): `exec` (execution-verified, C18-identical — the ceiling), `conf_strat` (PRIMARY: depth-stratified top think-P(True), per-depth quotas ∝ pool judge-score MASS — attempt-2 pivot: candidate-count quotas provably allocate slots where wrong candidates explode; verifier-free, isolates the keep-test from cross-depth slot allocation), `conf_global` (ablation: naive global top-P(True), the fully-deployable policy; its expected depth-collapse is reported, not conflated), `rand`.
- Gate (pre-registered, run.py hard-stops otherwise): pool P(True)-vs-full_pass AUROC ≥ 0.65 AND conf_strat−rand purity gap ≥ 0.10 AND matched n ≥ 60. Below the gate the finding is "the judge does not transfer to this substrate" and training is not burned. (The gate reads oracle purity — an experiment-level run/stop decision, documented like the matched-n scalar.)
- Primary metric & decision rule (pre-registered): per cell (mode × metric × depth, incl. pooled) where exec−base ≥ 0.10, classify by paired-bootstrap CIs into a TRICHOTOMY — `conf~rand` (conf_strat-vs-rand CI ≤ 0: filter adds nothing), `intermediate` (beats rand, below exec), `conf~exec` (beats rand, CI vs exec includes 0: verifier replaceable). The joint-bootstrap recovery ratio (conf−base)/(exec−base) is reported with CI as a magnitude, not a cutoff (n=25/depth cannot resolve 0.8 vs 1.0).
- Secondary: calibration survival — (a) fixed judge set (base's no-think eval-task samples; every model judges the SAME candidates → judge-change isolated), inflation headline = P(True)-on-INCORRECT drift (overall mean drift confounds inflation with real ability); (b) self-distribution pass (each model judges its OWN think-mode eval-task candidates — the actual round-2 flywheel number).
- Oracle-only metrics: `full_pass` grades all evals, builds the `exec` arm, and gates the run; in the conf arms' selection it never appears (post-hoc purity reporting only).
- Hidden-label boundary: conf-arm selection reads nothing but the model's own P(True) logit (no-think A/B judge, P(A) after "Answer: ") + generator metadata (task depth, draw frequency for tiebreaks). Matching size to the exec arm leaks one scalar (documented design constant).
- Known limits (pre-registered): single LoRA seed per arm (~30 optimizer steps — arm deltas include seed noise); no depth-4 cell (C18 comparability cell dropped for time); conf_global gets no think-mode eval (ablation, nothink only).

## Run

Smoke (~10 min end-to-end, all five stages):

```bash
python scripts/run.py --smoke
```

Full (~12 h on the RTX 4090: harvest ~2 h, pool think-judge ~50 min, 4 trains ~12 min, 9
pipeline evals ~5 h + 1 held-back conf_global think eval post-pipeline, 10 calib passes ~4 h —
the think-judge and self-distribution passes dominate):

```bash
python scripts/run.py            # idempotent; safe to re-run after interruption
python scripts/analyze.py
```

## Results

Full narrative in `reports/report.md`; stats in `runs/verdict.json`; saga in `experiment_log.md`.

1. **Gate stop first (kept):** the NO-think P(True) judge is at chance within-task on this
   substrate (0.471; pooled 0.749 = task difficulty). C46's within-problem law is
   substrate-scoped: it holds where correctness is semantically readable, not where it must be
   computed.
2. **Serial compute rescues the judge:** CoT judging (`judge_think`, budget 512, 99%
   forced-close) lifts within-task AUROC to 0.845 (pooled 0.961), P(True) correct/incorrect
   0.61/0.08 full-pool — C44's law governs the judge seat. The pivot re-ranked the conf arms on
   think-P(True); gate passed (purity gap 0.400).
3. **Trichotomy = conf~rand.** No pre-registered cell cleared exec−base ≥ 0.10. In the post-hoc
   cell where exec moved (d2 think greedy 0.08 → 0.24, CI [+0.04,+0.32], p=0.011), conf_strat
   +0.04 = rand +0.04 exactly, despite ~15× purity (0.429 vs 0.029). Recovery 0.25, CI [−1,+1].
   Banking is far less noise-tolerant than selection.
4. **The exec ceiling is dose-limited:** first matched-dose C18 re-run under the strict frozen
   paired eval → d2 think coverage flat (0.28 → 0.28). With the lineage audit (C18: p=0.082,
   3/20 d2 eval tasks leaked under the canonical probe-set func-sig; C22/C23 replications at
   3.5× dose; C24 dose law),
   C18's headline was a low-dose overestimate, not a false effect.
5. **Calibration survives banking as a RANKER:** fixed-set think-judge within-AUROC 0.872 →
   0.873 (exec) / 0.883 (conf_strat). But P(True) on own INCORRECT candidates doubles on the
   self-distribution (0.091 → 0.204). Rank filters survive round 2; fixed thresholds silently
   degrade. Own pass-rate 0.120 → 0.093 (no capability gain, consistent with 3).
6. **No-think SFT reallocates the no-think proposal prior; CoT shields think:** d1 no-think
   cov@16 collapses 0.72 → 0.32–0.40 for exec/conf_global/rand (conserved correct mass crammed
   onto the banked op-family; dedup-family tasks 9/10 → 0/10) but conf_strat's d3-heavy quota is
   sub-threshold (0.76); think-mode diversity untouched. Registered prediction for the held-back
   conf_global think eval: d1 cov_any ≈ 0.80–0.88, losses on dedup tasks — result: **0.88,
   uniq 8.6, all 3 residual losses in the dedup/unique family (3/3 confirmed)**.

## Interpretation

The verifier is load-bearing at the TRAINING seat, not the JUDGING seat. As a selector (C41/C46),
P(True) approaches execution; as a training filter it fails — not because the judge ranks badly
(0.845) but because banking at this dose tolerates almost no confidently-wrong data, and the
wrongs a confidence filter admits are precisely the plausible ones. The judge itself is robust to
self-training (ranks survive, scores inflate on the self-distribution), so the flywheel's binding
constraint is not feedback collapse but dose (C24): scale the harvest so the top-rank slice is
pure AND diverse, or keep the executable verifier. Deployment rules: think-judge (never no-think)
on computational substrates; filter by rank within depth strata (score-mass quotas), never by a
fixed threshold.

## Knowledgebase Update

- Program evidence updated: `posttraining_and_adaptation` × `evidence_conditioned_selection`
- Claim ledger updated: C47 (this experiment); C18 annotated with the eval audit + dose-law framing

## Artifacts

- `src/` — shared banking substrate (families list-DSL, gen_lib with the P(True) judge, code_env)
- `scripts/` — `harvest_pool.py` (shared pool + oracle + P(True) annotations, AUROC canary), `build_arms.py` (matched-size exec/conf/rand sets), `train_lora.py` (C18-identical), `eval_ladder.py` (frozen paired eval, C24 version), `calib_eval.py` (fixed-judge-set calibration survival), `run.py`, `analyze.py`
- `data/` — train_tasks, pool, per-arm training sets, frozen eval, judge set
- `runs/` — adapters (moved external before commit), eval/calib JSONs
- `reports/` — design_review.md, report.md, artifact_manifest.yaml

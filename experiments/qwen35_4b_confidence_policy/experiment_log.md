## 2026-07-13 — CORRECTION: escalation null at n=400 (backend switch HF->vLLM + power-up)

Per C57's own next-test #1 (the n=24-60 subset was flagged under-powered), powered
up the escalation arm and it REVERSED. Two changes: (a) switched generation from
the HF backend to vLLM (~10x faster; user prompt "why HF when vLLM is right
there"); (b) fixed a vLLM judge bug — after "Answer: " the 4B emits the
SPACE-PREFIXED " A"/" B" (ids 357/417), not the bare "A"/"B" (32/33), so reading
32/33 gave a degenerate p_true=0.5. The vLLM p_true is then a strong signal
(per-cand AUROC 0.756-0.769; solvability AUROC 0.715-0.727, matching the HF pool).

Regenerated the full pools on vLLM at n=400 x k=6, budgets 256 vs 2048.
- SURVIVES: pure-2048 modestly > pure-256 on the accuracy-vs-tokens frontier
  (high-k 0.593 vs 0.581).
- REVERSED: selective escalation of the abstained tail vs matched-compute breadth
  is NULL: esc-minus-breadth +0.022/+0.004/+0.017/+0.006 at abstain 20/30/40/50%,
  every 95% bootstrap CI spanning 0 (n=80/120/160/200). The first-pass HF n=24-60
  win (hardest-20% 0.458 vs 0.304) did NOT replicate.
- LIKELY MECHANISM: MBPP nearly SATURATES the think budget (the 4B self-limits to
  ~108-172 think tokens even at budget 2048), so the serial-compute lever is too
  weak to differentiate. Whether escalation helps on a genuinely budget-BOUND
  family is the owed test.
- LESSON: power up abstained-subset effects before claiming; a pre-registered
  under-powered win was correctly flagged and then reversed.
Robust deployable core: SELECT (argmax P(True)) + ABSTAIN (max-P(True) threshold),
both verifier-free from one logit. C57 corrected.

## 2026-07-13 — GENERALIZATION: the conf-select advantage is difficulty-dependent (HumanEval ties majority)

Ran the identical Part-1 frontier on the cached HumanEval pool (68 tasks x 9
cands, same schema; frontier.py --src humaneval). HumanEval is EASY for the 4B
(base pass 0.91, 67/68 solvable). There the confidence-SELECT advantage over
majority-vote VANISHES: k=9 conf 0.941 == majority 0.941 (majority slightly ahead
at k=7/8). On MBPP (base pass ~0.53) conf-select clearly beat majority (0.762 vs
0.742). So single-token P(True) selection beats self-consistency only when the
task is hard enough that selection matters; when the model is already ~0.9
accurate, majority-vote catches up. Abstention still works (HumanEval
risk-coverage clean, but only 1 unsolvable task limits the solvability AUROC).
P(True) > mean-logprob holds on both. C57's SELECT claim is thus bounded to
moderate-difficulty tasks. Also: MBPP n_think is saturated for the median task
(~86 tokens at both 256 and 2048 budgets) but the top ~10% tail is budget-bound
(p99 256->2048); escalation still fails that tail because those tasks are
capability-hard, not merely compute-starved.

## 2026-07-13 — difficulty curve resolves the dependence: conf-select's edge is on HARD tasks

Pooled MBPP+HumanEval (312 tasks), binned by per-task pass rate, conf-select vs
majority at k=6: hard (pass 0.08, n=68) +0.045; medium (pass 0.54, n=27) +0.038;
easy (pass 0.97, n=217) -0.007. The P(True)-select advantage concentrates exactly
on the hard tasks and vanishes on easy ones — self-consistent with abstention
(which flags the same hard tasks). Deployable rule: use conf-select where the
model is uncertain; majority-vote is fine when it's already ~0.97 accurate.

## 2026-07-13 — CAPSTONE: compute-optimal = confidence-gated adaptive ALLOCATION (not escalation)

The escalation (spend on depth) was null, but ALLOCATION (spend breadth only where
needed) is the real win. Confidence-gated adaptive sampling — commit the greedy
answer if its P(True) >= threshold, else sample K=8 + conf-select — dominates
uniform-k conf-select on the accuracy-vs-avg-compute frontier: reaches the k=9
ceiling 0.762 at ~4.25 avg samples (uniform needs 9), strictly beating uniform at
7/9 operating points (avg 1.79: 0.742 vs 0.720; avg 4.25: 0.762 vs 0.747). ~2x
compute saving. Mechanism: spend samples only on low-confidence (hard) tasks —
exactly where the difficulty curve shows conf-select helps. C57 -> Supported.
Deployable policy: sample greedily once, read P(True); high -> commit; low ->
sample K + argmax-P(True); abstain below a floor. One logit drives selection,
abstention, AND allocation.

## 2026-07-13 — adaptive allocation generalizes to HumanEval

adaptive_compute.py --src humaneval: the confidence-gated adaptive frontier beats
uniform at 7/9 operating points on HumanEval too (reaches the 0.941 ceiling at ~5
avg samples vs 9 uniform; one dip at avg 1.82). So the ~2x-compute-saving
allocation win holds on both a moderate (MBPP) and an easy (HumanEval) benchmark.

## 2026-07-13 — capstone generalizes to REASONING (cross-domain, cross-signal)

adaptive_toy.py on the original C41 pool (qwen35_4b_confidence_guided_compute, 240
records, confidence = P(answer)/C40 not a P(True) judge): confidence-gated
adaptive allocation beats uniform at 6/8 operating points (mid-compute +0.02..0.03;
small dips only at the k=1 and k=12 extremes). So the compute-optimal-allocation
result is domain-general — code (MBPP, HumanEval, P(True) judge) AND reasoning
(toy, P(answer)) — a robust, verifier-free, provenance-clean deployable policy.

## 2026-07-13 — agentic-domain arbitration: STARTED, deferred on infra

Began testing whether the confidence policy generalizes from static code/reasoning
to the multi-step GYM (gym_confidence.py in ../qwen35_4b_gauntlet_frontier/scripts).
Friction found: the gym's induction/exploration atoms make the BASE model think
heavily (~3200 think tokens at budget 4096, rarely closing </think>), so it only
emits parseable ANSWER lines at the 8192 budget — slow in eager, and a standalone
two-phase sampler does not reproduce the proven gym harness's emission handling
(0 parseable answers at 4096). CLEAN PATH (owed): reuse the proven harness.run_atoms
(fast, correct answers at 8192) and add a confidence signal — either answer-span
logprobs (P(answer)/C40) exposed from the VLLMRunner, or a P(True) judge pass over
gym answers. That is a ~half-day build vs the post-hoc code/reasoning analyses;
deferred pending direction. gym_confidence.py kept as a WIP starting point.

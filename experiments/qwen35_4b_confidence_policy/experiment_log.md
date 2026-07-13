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

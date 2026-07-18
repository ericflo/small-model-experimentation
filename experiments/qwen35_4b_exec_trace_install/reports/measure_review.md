# Measure Review

The transfer measurement of the exec_trace composite vs base on the
shared fitness harness (greedy pass@1) plus the agentic confirm.

Results (base -> exec_trace):
- HumanEval: 0.7622 (125/164) -> 0.7683 (126/164), +1 problem.
- MBPP: 0.5650 (113/200) -> 0.5500 (110/200), -3 problems.
- Agentic (duet-eval gen4, n=35, the real target): 8/35 -> 8/35, with
  5 base-only and 5 exec_trace-only discordant scenarios.

FROZEN FAST-EVAL RULE: by the letter, INSTALLED_CODING fires (HumanEval
strictly beats base AND MBPP within the 0.02 retention tolerance). But
this is a NOISE-LEVEL technicality — a single near-ceiling HumanEval
problem — and the pre-registered rule was too lenient (a >=1-problem
HumanEval bump should not count as installed capability). The agentic
eval, the REAL target (declared a follow-on confirm), is EXACTLY FLAT
(8/35 vs 8/35).

HONEST SUBSTANTIVE VERDICT: NULL. Execution-tracing did not move real
coding on any signal. Retention held (the forgetting risk was real; the
design beat it), but capability was RESHUFFLED (5v5 discordant agentic;
+1/-3 fast) not RAISED. The install!=convert law extends to coding:
installing a passive cognitive skill (execution modeling) does not
convert to coding capability. Lesson for future bets: tighten the rule
to require a meaningful delta (>= ~3 problems or a paired significance
test), and target the agentic LOOP (plan-act-verify-repair), not a
passive component.

**Verdict:** `PASS_MEASURE`.

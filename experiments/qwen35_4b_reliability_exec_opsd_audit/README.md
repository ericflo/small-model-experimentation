# qwen35_4b_reliability_exec_opsd_audit

Standalone experiment testing two reliability probes and one execution-grounded OPSD locality audit.

The experiment asks whether hidden-correct retrieval adaptations are already preferred by Qwen's own likelihood and whether behavioral counterexample evidence gives a hinted teacher task-specific branch signal that weak retrieval hints did not provide.

## Parts

1. **MAP likelihood selector**: score visible-pass retrieval-adapt candidates by raw code likelihood under the task prompt and select the highest mean logprob candidate.
2. **Low-temperature retrieval adaptation**: compare semantic adaptation at `T=0.0`, `T=0.1`, and the local `T=0.2` baseline.
3. **Execution-grounded locality audit**: at same-prefix code forks, compare teacher preference for the hidden-correct branch versus the hidden-wrong branch when the teacher is given a failing input, observed wrong output, and correct output.

## Gate

Run OPSD training only if execution-grounded hints add task-specific correct-branch preference beyond the no-hint student and shuffled-observation control. Full-reference hints are included only as a leakage ceiling.

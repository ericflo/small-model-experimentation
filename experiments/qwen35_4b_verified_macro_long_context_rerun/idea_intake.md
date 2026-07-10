# Idea intake: verified macro invention long-context rerun

## Program fit

- Primary program: `operator_and_skill_inventories`.
- Secondary connections: `structured_execution_and_compilers`, `benchmark_generalization`, and
  `test_time_reasoning_budget`.
- Existing or new program: existing. The scientific question is still whether a verified composite
  inventory improves search; the budget ladder is an anti-censoring correction, not a new program.
- Closest scorecard reviewed: `operator_and_skill_inventories`, whose current evidence says the
  bank is useful only if its entries can be called and selected reliably.
- Related-work query: `make related QUERY="verified macro long context thinking budget censoring"`.
- Related future work: the exactly-one-macro slot-conditioned sweep in the operator-program
  backlog. That is a broader interface redesign; this rerun first tests whether the original
  free-form design was simply starved of generation budget.

## Prior evidence

- `qwen35_4b_verified_macro_invention` is the direct parent. It froze a clean corpus and full
  control matrix, but never exposed its fresh v2 smoke or full tasks to the model.
- `qwen35_4b_thinking_budget_scaling` shows that Qwen3.5-4B's reasoning allowance is a real
  deployment variable and that cap behavior must be measured rather than assumed.
- `qwen35_4b_operator_inventory_scaling_stress` shows that fixed human-authored inventories remain
  searchable while composition cost grows; it did not learn reusable composite entries.
- `qwen35_4b_structure_search_scaling` shows exhaustive primitive search becomes exponentially
  expensive as depth grows, motivating representations that reduce surface decision depth.
- Closest near-duplicate: `qwen35_4b_verified_macro_invention`. This is deliberately a material
  follow-up rather than an amendment because the parent is result-bearing and its stop rule is
  immutable. The new uncertainty is whether its apparent interface wall survives a nonbinding,
  metadata-calibrated generation envelope.

## Novelty claim

No completed experiment has tested the frozen verified-macro matrix after calibrating a sufficiently
large reasoning and answer envelope without looking at task correctness; the only budgeted
plan-given evidence was fully thinking-capped and mostly answer-capped.

## Mechanism

If the parent failure was censoring, a larger allowance should let the model finish its own
reasoning, emit one exact macro surface for a supplied plan, and then reach the never-prompted
induction smoke. On the scientific tasks, recurring exact macros should reduce a true-depth-5
program to fewer surface decisions and shift Qwen's candidate distribution toward correct programs.

The explanation is false if a noncensored plan-given interface still fails exact literal expansion,
if the designed ceiling gives no induction oracle lift, if matched random composites tie mined
recurrences, or if any apparent lift disappears against a no-smaller-token base prefix. A run that
hits its cap is not falsifying evidence; it is censored setup evidence and must escalate.

## Control plan

- Baseline: `Qwen/Qwen3.5-4B` sampling over base primitives, K=24, with a prefix matched to each
  K=12 macro arm's measured token cost.
- Main treatment: deterministic train-only frequent verified macros (`mined`).
- Mechanism-falsifying controls: `mined_hint` exposes identical subsequences without callable
  aliases; five count/length/support-matched random libraries test semantic recurrence; and the
  designed library is an explicitly non-discovery ceiling.
- Model contribution: `qwen_ranked`, accepted only through exact local expansion verification and
  compared with its own support-matched placebo ensemble.
- Robustness: 80 recurring-motif tasks paired with 40 primitive-multiset-matched no-reuse tasks.
- Inference control: all arms and sample-more baselines use the same copied vLLM wrapper, settled
  ladder rung, answer cap, seeds, and runner version. A censoring trigger reruns the whole matrix;
  it never repairs one arm in isolation.
- Hidden-label boundary: budget selection reads runner token/finish metadata only. Macro
  construction sees train programs only; prompts see visible I/O only; hidden/probe outputs enter
  only the frozen analyzer.

## Evidence output

- Byte-identical frozen inputs with parent commit and SHA-256 provenance.
- Per-rung raw vLLM outputs and metadata showing thinking-cap, answer-truncation, and token
  headroom, without pooling rungs.
- A 16-record heldout train-only interface verdict before any scientific prompt.
- Fresh-smoke and full paired deployable/oracle metrics only after their gates pass.
- Program evidence/backlog and shared synthesis updates only if the result changes strategy; a
  claim-ledger update only if a registered scientific contrast clears.
- Reusable artifact: a self-contained vLLM macro harness whose large-context budget calibration and
  whole-matrix escalation can be copied into later experiments.

## Decision

- Run experiment: yes, after the adversarial review, source-hash checks, and CPU gates.
- Create program: no.
- Write synthesis only: no; the original scientific matrix remains entirely unrun.
- Stop/branch: budget-ladder exhaustion is setup-inconclusive. A clean heldout interface failure
  stops before smoke. A clean smoke failure localizes the interface/induction ceiling. Full runs
  only after every protected gate passes.

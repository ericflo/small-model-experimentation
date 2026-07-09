# Idea intake: verified macro invention

- Source: user-selected option from the 2026-07-09 forest review.
- Owning program: `operator_and_skill_inventories`; secondary connections are
  `structured_execution_and_compilers` and `benchmark_generalization`.
- Closest queued item: the operator-program backlog says to compare human-designed,
  mined, and model-discovered inventory entries, but no queued proposal or result-bearing
  experiment implements that comparison.

## Prior evidence

- `qwen35_4b_operator_inventory_scaling_stress`: fixed human-authored inventories remain
  searchable but two-hole cost grows quadratically.
- `qwen35_4b_inventory_shortlister_training` and `qwen35_4b_joint_shortlister_ladder`:
  directly naming pairs from a large fixed inventory floors at zero recall.
- `qwen35_4b_structure_search_scaling` (C35): base-primitive exhaustive search dominates
  model proposals while the complete skeleton space remains enumerable, then becomes
  exponentially expensive.
- `qwen35_4b_hypothesize_verify_wall` (C48): a taught search procedure improves practiced
  depths but does not transfer one composition step deeper.

Closest near-duplicate: `qwen35_4b_operator_inventory_scaling_stress`. It expands a fixed
bank of atomic, human-authored operators. This experiment instead derives verified composite
operators from train-only solved programs and tests reuse on behaviorally fresh programs.

## Novelty claim

The corpus has treated the hypothesis language as fixed. It has not tested whether a
solve -> abstract -> verify -> reuse loop can turn recurring deep motifs into reusable
operators and thereby make fresh deep programs shallow enough for Qwen3.5-4B to propose.

## Mechanism

If the depth wall is partly a search-language problem, exact reusable macros should reduce
the number of decisions needed on held-out combinations of familiar motifs. A real
abstraction effect must beat base-primitive sample-more and length/count/support-matched
random composite macros. It must also be carried by macro-using solutions and concentrate
on the recurring-motif split rather than a no-reuse control split.

The explanation is false if random composites tie mined macros, if gains come only from
oracle candidate coverage without visible-only selection, or if evaluation programs leak
into the macro-construction corpus.

## Control plan

- Baseline: Qwen/Qwen3.5-4B samples programs over the base primitive inventory.
- Main treatment: deterministic frequent-subsequence macros mined from the frozen,
  train-only proposal view.
- Highlight-only control: show the identical mined subsequences and demonstrations but
  require expanded base-primitive output, separating a useful search prior from a callable
  representational chunk.
- Model-discovery arm: Qwen/Qwen3.5-4B proposes macro expansions from exactly the same
  train-only proposal view; all proposals are expanded and execution-verified locally.
- Mechanism-falsifying control: count/length/support-matched random composite macros.
- Ceiling: generator-known recurring motifs, clearly labeled as an oracle/design ceiling.
- Robustness: a no-reuse split with the same base depth but no designed recurring motif.
- Hidden-label boundary: macro construction sees training programs only. Solver prompts see
  visible I/O only. Hidden and probe I/O are used only by the committed analyzer.
- Compute control: every model-facing arm, including base sample-more, uses the experiment's
  copied `src/vllm_runner.py`. Report sampled, prompt, logical-input, wall-time, and
  interpreter-call axes; include a base coverage-vs-token curve.

## Evidence output

- A frozen procedural dataset and split manifest.
- Exact verified macro libraries with provenance and support counts.
- Raw vLLM outputs plus runner metadata sidecars.
- Paired deployable and oracle metrics, macro-use diagnostics, and compute curves.
- Program evidence/backlog update, and a claim/synthesis update only if the registered gate
  is met.

## Decision

- Run experiment: yes, after adversarial design review and CPU/smoke gates.
- Create program: no; the operator-inventory program already owns abstraction banks.
- Expensive work: do not proceed past smoke unless the oracle macro interface is usable and
  all split/leakage checks pass.

## Post-smoke branch decision

Smoke v1 passed its parser-rate thresholds but failed on answer truncation, macro use, and the
designed-ceiling oracle comparison. Full generation was not run, so this is an interface failure,
not evidence about abstraction quality. The original intake question, closest duplicate, program
assignment, controls, and evidence target remain unchanged; this predeclared repair branch does
not warrant a second experiment directory.

The exact v1 state is preserved. A fresh v2 smoke uses seed `20260710`, new record ids, the
already registered full thinking budget, a shared surface-first alias procedure, and matched
base/designed generation. It may proceed only after disjointness and the train-only plan-given
probe pass. A post-failure line-local reparse of the v1 macro proposals found usable lines, but it
is explicitly exploratory and cannot be promoted into v1 evidence. If the fresh interface gate
fails, stop rather than inspect or tune on full tasks.

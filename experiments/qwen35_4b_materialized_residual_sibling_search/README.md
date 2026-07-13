# Qwen3.5-4B Materialized Residual Sibling Search

**Status:** in-progress · since 2026-07-13 · attempt 1 aborted in live preflight
before the first experimental request; append-only v2 repair is pushed and its
separate retry lock awaits commit, push, and CI.

## Research Program

- Primary: `structured_execution_and_compilers`.
- Secondary: `evidence_conditioned_selection`,
  `interpretability_and_diagnostics`, and `test_time_reasoning_budget`.
- Immediate parent: `qwen35_4b_early_text_hypothesis_forking`.
- Closest algorithmic near-duplicate: `qwen35_4b_decompose_compose_frontier`.
- Closest representation-negative: `qwen35_4b_partial_structure_search`.

## Question

On fresh exact-depth-three tasks, can Qwen3.5-4B complete a useful two-step
residual when an external interpreter materializes each candidate first
operation's public consequences? Does the resulting all-sibling explorer beat
candidate names and candidate-blind sampling at taskwise matched sampled and
logical model tokens? Secondarily, can a cheap no-think viability score retain
most of the all-sibling coverage with only four completions?

## Hypothesis

The parent found broad one-step semantic routing but almost no downstream
composition. Here each candidate `h` produces a concrete relation
`h(x_i) -> y_i`, reducing a depth-three inverse problem to a depth-two suffix.
The primary test does not pretend the model can discover `h`: it treats all 24
siblings symmetrically and measures whether materialized consequences change
the suffix proposal distribution. A shuffled state/target alignment preserves
the treatment's token multiset while destroying its semantics.

## Setup

- Model: only `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16.
- Backend: the experiment-local pinned vLLM runner for every model arm; no
  backend mixing and no training in this experiment.
- Tasks: fresh procedural integer-list functions whose output signature on a
  frozen common panel is realizable at depth three but not at depth zero, one,
  or two. Functions, concrete triples, and registered suffixes are disjoint
  across splits.
- Inputs: list lengths four through eight and IID discrete-uniform integer
  values from -9 through 9; within-list duplicates are allowed, while every
  visible, hidden, and probe input in a task is distinct.
- Splits: 24 public mechanics tasks, 48 qualification tasks in two frozen
  shards, and 192 untouched confirmation tasks in eight frozen shards.
- Public-live sibling: a candidate first operation for which at least one of
  the 24² legal suffixes fits every visible row. This is a deployable,
  model-free label, not hidden gold. Every 24-task block has exactly 8/8/4/4
  tasks with one/two/three/four public-live siblings.
- Primary treatment: generate one strict two-operation suffix from every one
  of the 24 materialized sibling prompts, assemble full programs, and select
  using visible executions and independently generated unlabeled probes.
- Primary controls: all-24 name-only siblings; all-24 task-hash-deranged
  state/target alignments; candidate-blind full-program sampling matched
  taskwise at conservative sampled-token and logical-token first-over points;
  exact public viability; and exhaustive CPU enumeration as an explicit
  dominance reference.
- Ranking secondary: no-think targeted raw log probabilities for binary
  materialized viability, candidate-name viability, the deranged relation, and
  a C25-style listwise original-I/O next-operation scorer. Any top-four policy
  gets its own real four-request generation run; all-24 outputs are not reused.
- Primary metrics: hidden-correct proposal coverage and visible-only selected
  hidden accuracy. Ranking uses within-task live recall@4 and live hit@4.
- Hidden boundary: hidden outputs may score only already frozen pools and
  selections. Prompts, stopping, resource matching, and selected IDs use
  visible examples, deterministic candidate states, unlabeled probe inputs,
  and frozen hashes only.

## Construction and execution boundaries

The DSL has one shared typed `INVALID` result for any illegal or safety-bound
execution. Empty or invalid target trajectories are rejected. Candidate
executions that become empty or invalid are simply ineligible; model-produced
code is never executed. The 24 concrete operation aliases and both two-op and
three-op parsers are frozen before model use.

Admission never filters on hidden candidate correctness or selector-probe
agreement. This keeps held-out selection fallible. A frozen common-input
function fingerprint, rather than a hash of each task's own rows, prevents the
same function from appearing in multiple splits.

## Stages and stop logic

1. CPU smoke proves exact minimum depth, common-panel and split disjointness,
   public-live enumeration, live-count balance, candidate-state distinction,
   partial-operation semantics, strict assembly/execution, selector blindness,
   taskwise resource matching, and key discrete threshold geometry.
2. Public mechanics runs two independent checks: a live-sibling suffix/direct
   ABI ceiling and no-think sibling ranking. Interface failure seals all model
   qualification. Suffix failure seals the all-24 explorer. Ranking failure
   seals only the top-four efficiency secondary.
3. Qualification compares the all-24 materialized explorer against every
   primary control on 48 new tasks using point and shard-consistency futility
   gates. It makes no claim-grade significance statement. The optional
   top-four decision is separate and cannot veto primary confirmation.
4. Confirmation repeats the untouched frozen protocol on 192 tasks. The sole
   claim-grade family is selected accuracy versus name-only, shuffled-state,
   and the two conservative direct first-over baselines, using exact paired
   one-sided McNemar tests with Holm familywise correction. Any top-four result
   is a descriptive operational secondary and cannot alter the primary result.
5. A replicated pass supplies positive prior evidence for a separate
   residual-policy supervision experiment. Failure seals this untrained
   interface, not the logically independent training question.

## Run

Model-free smoke:

```bash
.venv/bin/python experiments/qwen35_4b_materialized_residual_sibling_search/scripts/run.py --stage smoke
```

Mechanics retry command (sealed until the repaired implementation and separate
v2 lock are committed, pushed, and green in CI):

```bash
.venv-vllm/bin/python experiments/qwen35_4b_materialized_residual_sibling_search/scripts/run_mechanics.py --stage run
```

## Results

The model-free smoke passed on the frozen 264-task construction: 24 mechanics,
48 qualification, and 192 confirmation tasks. It found 3,525 eligible exact
depth-three function fingerprints, rebuilt the split deterministically,
independently re-audited public-live sets on 34 registered tasks, and obtained
0.966 compound pass probability at the registered confirmation alternative.
The 38,596 exact rendered prompts across every frozen task, candidate, and
condition span 259 to 941 tokens; the 259-token minimum is the short supplied
echo ceiling. The receipt records zero model
loads, zero model calls, and no benchmark reads. This is a construction and
design result only; no Qwen3.5-4B capability result exists.

The first live attempt initialized the exact engine but failed its cache receipt
before the first experimental generation request. No invocation transaction or
sampled output exists. The failed preflight is preserved as incident evidence;
it is not a capability result and its embedded PASS label is unauthenticated.

## Interpretation

The design remains eligible after an independently audited append-only preflight
repair and fresh versioned lock. Even a later model pass would be an external
structured-search result. CPU 24³ enumeration remains exact and cheap at this
depth.

## Knowledgebase Update

- Program evidence updated: pending a result.
- Program backlog updated: pending a result.
- Claim ledger updated: no; no result exists.

## Artifacts

- `src/`
- `scripts/`
- `configs/`
- `data/`
- `runs/`
- `analysis/`
- `reports/`
- `reports/artifact_manifest.yaml`
- `idea_intake.md`
- `reports/preregistration.md`
- `reports/design_review.md`

# Qwen3.5-4B Partial-Structure Recognition-Guided Search

**Status:** finished

Can Qwen3.5-4B recognize completable partial program skeletons well enough to prune depth-5 search and beat matched-compute sampling and unguided expansion?

## Research Program

- Primary: `structured_execution_and_compilers`
- Secondary: `evidence_conditioned_selection`
- Program question: can a fixed small model contribute a useful recognition heuristic when complete
  compositional proposal is rare and exhaustive structure search is becoming expensive?
- Prior anchors: C25/C26 (next-operation proposal and thinking), C35 (brute search through depth 4), C47
  (thinking P(True) on completed candidates), and C48 (partial-structure recognition left open).

## Question

Can the frozen `Qwen/Qwen3.5-4B` recognize whether an externally supplied, unfinished operation-type
skeleton can still be completed, and can that recognition prune true-depth-5 search better than direct
matched-compute sampling, proposal likelihood, and unguided expansion?

## Hypothesis

The model is a stronger recognizer than proposer on several completed-candidate tasks. Thinking may let it
apply that asymmetry one level earlier: score the semantic viability of a supplied type-prefix even when it
cannot generate the whole five-operation skeleton. If its scores preserve a live path while discarding most
siblings, a small beam can construct solutions absent from direct samples. The mechanism is false if scores
only recognize canonical-looking prefixes, correlate with task difficulty rather than within-task viability,
or fail to improve live-child retention and end-to-end search.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, pinned revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, frozen throughout.
- Backend: the pinned experiment-local vLLM runner for every model arm. No backend mixing.
- Dataset/task source: fresh procedural list transformations over 16 operation types. Parameters are filled
  only after a type skeleton is retained. No benchmark data or sources are read.
- Split: 48 fresh exact-min-depth depth-4 calibration tasks, 12 dedicated exact-min-depth depth-5
  oracle-development tasks, and 60 behavior-disjoint exact-min-depth depth-5 primary tasks. Each task has 8
  visible examples, 6 oracle-label probes, and 6 final hidden examples.
- Search state: a type-only skeleton prefix. It contains no parameters or interpreter-materialized state,
  keeping the recognition question distinct from C25's concrete next-operation proposal.
- Primary baselines: one frozen pool of 512 direct type-skeleton samples per task, independently and
  deterministically prefix-truncated to (a) the recognition arm's sampled-token cap and (b) its total logical
  model-token cap. Both use the identical parameter-fill cap, visible executor, and selector. The second,
  stronger arm was added before any primary search when the pre-run audit found that decode-only matching did
  not charge recognition's repeated prefills.
- Controls: no-think P(True), C25-style next-operation likelihood, score-shuffled thinking, seeded uniform
  beam, a model-free surface baseline, budget-truncated brute expansion, full brute enumeration, and an
  oracle-live beam.
- Calibration primary: task-macro, prefix-depth-stratified AUROC plus live-child recall@4 and complete live-
  path survival. Pooled AUROC is diagnostic only.
- Search primary: selected-candidate hidden success at fixed model-token and parameter-fill caps.
- Oracle-only metrics: semantic live label, successful-completion count, oracle beam, pool hidden coverage,
  min-depth audit, and final hidden correctness.
- Hidden-label boundary: model prompts contain only the DSL, visible I/O, prefix, and remaining slots. Oracle
  probes, hidden examples, target pipeline, live labels, and completion counts never enter a prompt. Search
  termination and candidate selection are visible-only; after exact-depth task construction, hidden data are
  grading-only.

## Pre-registered gates

1. **Exactness gate:** every scored task must have an exhaustion receipt proving no behaviorally equivalent
   shorter program exists. A seen-cap exhaustion is a hard failure, never a negative label.
2. **Oracle-state gate:** oracle-live beam must preserve a hidden-solving path on at least 90% of development
   tasks while using at least 10x fewer completed skeletons than full enumeration. Otherwise this prefix
   representation is not a useful search state and GPU scoring stops.
3. **Recognition gate:** thinking P(True) must reach macro within-task AUROC at least 0.65 with task-bootstrap
   lower 95% bound above 0.50, improve at least 0.05 over the strongest non-oracle baseline, and improve
   live-child recall@4 by at least 0.10 with lower bound above zero. AUROC without actionable retention is a
   stop result.
4. **Search success:** at depth 5, thinking-guided selected hidden success must exceed *both* direct
   matched-compute sample-more arms by at least 0.10 with paired CI lower bound above zero, beat shuffled
   scores and next-op likelihood, and agree in direction against both direct arms in both frozen task shards.
   Any exhausted direct pool or unmatched compute budget invalidates the frontier-advance verdict.

Only a search win licenses a separate banking follow-up. Banking is not part of this experiment.

## Run

Smoke:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_partial_structure_search/scripts/run.py --smoke
```

Full:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_partial_structure_search/scripts/run.py
```

The orchestrator is idempotent and stops at either failed gate. CPU oracle construction runs before any
substantial GPU judging.

## Results

**G1 stop: oracle-useful, model-unreadable.** The dedicated development oracle passed: width-4 live-prefix
search retained a hidden-solving path on 12/12 depth-5 tasks and compressed completed skeletons 262,144x.
The model gate then failed all six predicates on 48 calibration tasks / 7,200 children:

| method | macro within-task AUROC | live recall@4 |
|---|---:|---:|
| thinking P(viable) | **0.506** (CI 0.470--0.543) | **0.251** |
| no-think P(viable) | 0.556 | 0.303 |
| next-op likelihood | 0.504 | 0.228 |
| surface | 0.519 | 0.278 |
| random | 0.506 | 0.263 |

Thinking trailed the strongest baseline by -0.049 AUROC (CI -0.090 to -0.010) and -0.052 recall
(CI -0.110 to +0.007). Pooled thinking AUROC was a misleading 0.557; within-task discrimination was chance.
Wrong-task visible examples were no worse on the canary (original 0.450 vs shuffled 0.476), and 100% of
thinking rows hit the 256-token cap.

Exact visible-only full brute was also cheaper than the nominal tree suggested: 60 depth-5 primary tasks
finished in 111.85 seconds on eight CPU workers, covered a hidden solver on 60/60, and selected correctly on
56/60. The full model-guided primary search and banking were not run.

## Interpretation

The recognizer/proposer asymmetry does not extend to this state representation. A finished concrete candidate
can be verified by execution-like reasoning; an unfinished type-only prefix requires existential search over
missing parameters and operations. The exact oracle can do that from the transition system, but the prompt
does not expose parameter constraints or intermediate residual state. More thinking produced task-uncoupled
scores and was worse than the no-think readout.

The next step should first locate the real exact-search resource crossover at depth 6. Only if guidance is
economically needed should the model interface change: expose compact feasible-parameter domains and
per-example residual constraints, then rerun only the readability/actionability gate. See the [full
report](reports/report.md).

## Knowledgebase Update

- Program evidence and backlogs: updated with the G1 stop and residualized-state branch.
- Shared synthesis and program scorecards: updated; type-only prefix viability is retired.
- Claim ledger: no new claim added while the repository-wide claim re-grade remains incomplete.

## Artifacts

- `src/`
- `scripts/`
- `configs/`
- `data/`
- `runs/`
- `analysis/`
- `reports/`
- `reports/artifact_manifest.yaml`

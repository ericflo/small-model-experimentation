# Experiment Log: Qwen Candidate-Conditioned Trace Verifier

## Objective

Test whether candidate-conditioned Qwen reading can select correct executable
repair candidates from a fixed candidate pool without using the target
answer or target state at inference.

## Design Commitments

- Fresh experiment directory:
  `experiments/qwen_candidate_conditioned_trace_verifier/`.
- Large artifacts separated under:
  `large_artifacts/qwen_candidate_conditioned_trace_verifier/`.
- Candidate pool and prompt reconstruction are deterministic.
- Main report is standalone and does not require external experiment context.
- Raw accuracy is reported, but source-normalized oracle-gap capture is the
  primary selector-quality metric.
- Include no-repair, oracle, feature, trace-only, frozen Qwen, trained Qwen,
  ECHO-ablation, shuffled-label, prompt-only, candidate-only, trace-corrupted,
  and held-out-source diagnostics.

## Iterations

### 2026-06-25 Scaffold

Created the standalone directory, large-artifact directories, README, and
experiment log.

### 2026-06-25 Smoke Iteration 1

Ran the full smoke path with the initial text serialization. It completed, but
inspection showed that full prompt-plus-candidate examples were about 955 Qwen
tokens, so the configured reader context would have truncated the verifier
question and candidate tail.

Patched serialization before the main run:

- Strip internal register placeholders from the task text.
- Compact the natural task to semicolon-separated operations.
- Compact candidate traces to short `step:op=value` lines.

The corrected full prompt-plus-candidate serialization is about 367 tokens on a
length-24 example, fitting the configured reader context without cutting off the
verdict question.

### 2026-06-25 Smoke Iteration 2

Ran `smoke_candidate_conditioned_qwen_trace_verifier_v2` with compact
serialization and all controls enabled. Cache loading, prompt reconstruction,
Qwen embedding, feature baseline, trace baseline, Qwen ranking head,
ECHO-ablation, shuffled-label control, prompt-only, candidate-only,
trace-corrupted controls, metrics, charts, Markdown report, and HTML report all
completed.

### 2026-06-25 Main Iteration 1

Ran `main_candidate_conditioned_qwen_trace_verifier_v1` with three source
seeds, two critic seeds, shortlist size 64, and full candidate-conditioned Qwen
embeddings rebuilt from `Qwen/Qwen3-4B`.

Important results:

- Candidate coverage was high. On `standard_L24`, no repair was 44.4%,
  deployable-shortlist oracle was 75.0%, and full-pool oracle was 90.3%.
- Frozen Qwen yes/no scoring was destructive on `standard_L24`: 1.4% accuracy,
  97.2% changed fraction, and 96.9% damage rate.
- Trained candidate-conditioned Qwen heads also over-edited. `qwen_rank`
  reached 6.9%; `qwen_echo` reached 5.6-9.7%.
- Feature-only and trace-only baselines selected no repair and therefore tied
  the 44.4% base accuracy.

This showed that concrete candidate reading did not produce a safe selector in
the ungated setup; the dominant failure was high-confidence destructive edits.

### 2026-06-25 Main Iteration 2

Patched the runner to add validation-tuned base fallback gates for frozen Qwen,
`qwen_rank`, and `qwen_echo`, plus an `--embedding_run_name` flag so the second
main run could reuse the large Qwen embedding cache without loading the model
again.

Ran `main_candidate_conditioned_qwen_trace_verifier_v2` using the v1 embedding
cache. Runtime was 291.979 seconds and all trained/control heads completed.

Main result:

- Gating prevented most destructive edits but did not recover the reachable
  repair gap.
- On `standard_L24`, `qwen_echo_gated` tied the base at 44.4% with 0.0%
  oracle-gap capture; `qwen_rank_gated` fell to 41.7%.
- The validation-tuned gates mostly selected no-op. For `qwen_rank_gated` and
  `qwen_echo_gated`, validation accuracy equaled the base validation accuracy
  and validation recovery was 0.0%.
- The main report and HTML report were regenerated from v2 metrics.

Interpretation: the candidate set contains many answer-correct repairs, but the
tested Qwen-conditioned selectors did not learn to identify them safely without
target leakage. Safe gating converted the method into a conservative no-op
policy rather than a useful repair selector.

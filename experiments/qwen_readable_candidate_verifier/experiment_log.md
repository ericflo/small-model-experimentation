# Experiment Log: Qwen Readable Candidate Verifier

## Objective

Test whether a frozen or lightly trained Qwen reader can select correct repair
candidates when candidates are rendered as readable task code, candidate code,
claimed final values, and optional execution traces. The target answer and
target state are never provided to the verifier at inference.

## Design Commitments

- Fresh experiment directory:
  `experiments/qwen_readable_candidate_verifier/`.
- Large artifacts separated under:
  `large_artifacts/qwen_readable_candidate_verifier/`.
- Candidate pool and task reconstruction are deterministic.
- Main report is standalone and does not require external experiment context.
- Report both ungated and validation-gated decisions.
- Report selector accuracy, oracle-gap capture, shortlist-oracle capture,
  ranking AUC, changed fraction, damage rate, and recovery rate.
- Include readable trace-field ablations: task-value, program equivalence,
  program plus claimed final, program plus full trace, trace-corrupted,
  candidate-only, prompt-only, and majority-by-claimed-output controls.
- Split selector quality by easy versus hard candidate groups, where difficulty
  is determined by how many shortlist candidates share the target answer.

## Iterations

### 2026-06-25 Scaffold

Created the standalone directory, large-artifact directories, README, and
experiment log.

### 2026-06-25 Smoke Iteration 1

Ran `smoke_qwen_readable_candidate_verifier_v1` with readable task/candidate
rendering, frozen Qwen scoring, one trained Qwen head, feature/trace controls,
charts, Markdown, and HTML report generation.

The smoke path completed, but the primary value-recompute arm used a first-token
numeric approximation. Qwen's tokenizer split all residue strings in this setup,
which collapsed `zero_task_value` into an uninformative score. I patched the
runner to score exact numeric continuation log-likelihood over all residues
instead.

### 2026-06-25 Smoke Iteration 2

Ran `smoke_qwen_readable_candidate_verifier_v2` with rebuilt embeddings and
exact continuation scoring for `task_value`.

Smoke result on `standard_L24`:

- Base no-repair: 41.7%.
- Shortlist oracle: 50.0%; full-pool oracle: 100.0%.
- `zero_task_value`: 0.0%, ranking AUC 0.381, fully destructive on the tiny
  sample.
- `zero_program_final`, `zero_program_trace`, `zero_trace_corrupt`, and
  `qwen_program_final`: tied base with ranking AUC 0.500.
- Feature and trace controls showed high AUC on the tiny sample but did not
  improve commit accuracy.

Smoke verified the data path, exact value scoring, trained-head path, easy/hard
metrics, charts, and report generation.

### 2026-06-25 Main Iteration 1

Started `main_qwen_readable_candidate_verifier_v1` with all readable modes,
shortlist size 64, two critic seeds, and `max_length=1024`.

The exact `task_value` mode completed and cached, but prompt-length inspection
showed that full-trace prompts are about 1,400 tokens. A 1,024-token context
would truncate the trace arms, so I interrupted during `program_equiv` and
reran with a larger context.

Measured readable token lengths on the main selected set:

- `task_value`: max about 506 tokens.
- `program_equiv`: max about 990 tokens.
- `program_final`: max about 1,000 tokens.
- `program_trace`: max about 1,402 tokens.
- `trace_corrupt`: max about 1,406 tokens.
- `candidate_only`: max about 515 tokens.
- `prompt_only`: max about 505 tokens.

### 2026-06-25 Main Iteration 2

Started `main_qwen_readable_candidate_verifier_v2` with
`max_length=1536`, loading the completed `task_value` cache. `program_equiv`
and `program_final` completed and cached. The full-trace hidden-state embedding
path was too slow for a practical complete run, so I interrupted before
finishing `program_trace`.

### 2026-06-25 Main Iteration 3

Patched the runner so modes not listed in `--train_qwen_modes` are scored as
zero-only Qwen yes/no arms instead of full hidden-state embedding arms. This
keeps the trace and corrupted-trace controls while avoiding unnecessary learned
embedding caches. I first set `train_qwen_modes=program_final`.

The first v3 attempt revealed a control-flow bug: zero-only stores were being
overwritten by the old hidden-state branch. I fixed the branch and also changed
final-token yes/no scoring to call the base model and apply `lm_head` only to
the final hidden state, avoiding full-sequence vocabulary logits.

### 2026-06-25 Main Iteration 4

Reran after the scorer optimization. The optimized scorer reduced memory use
substantially, so I interrupted early and restarted with a larger Qwen batch
size.

### 2026-06-25 Main Iteration 5

Completed `main_qwen_readable_candidate_verifier_v5` with:

- `embedding_run_name=main_qwen_readable_candidate_verifier_v1`
- `max_length=1536`
- `qwen_batch_size=64`
- `train_qwen_modes=program_final`
- two critic seeds: 101 and 202

Runtime for the completed v5 run was 9,380.221 seconds. Large caches were
written under `large_artifacts/qwen_readable_candidate_verifier/embeddings`.

Main result on `standard_L24`:

- Base no-repair: 44.4%.
- Shortlist oracle: 75.0%; full-pool oracle: 90.3%.
- Majority-by-claimed-output: 9.7%, destructive.
- Frozen `zero_program_equiv`: 50.0%, 12.1% oracle-gap capture, AUC 0.677.
- Frozen `zero_program_trace`: 8.3%, destructive, AUC 0.620.
- Frozen `zero_trace_corrupt`: 1.4%, destructive, AUC 0.486.
- Trained `qwen_program_final`: 47.2%, 6.1% oracle-gap capture.
- Validation-gated `qwen_program_final`: 51.4%, 15.2% oracle-gap capture,
  zero damage rate, 12.5% recovery rate.
- Shuffled-label trained control was destructive on average.

Interpretation: readable program text does expose usable signal that was absent
from the dense/compact candidate channel, but the gain is modest. Full readable
trace text did not help; it made the frozen selector highly destructive, and
corrupting the trace made it worse. The easy/hard split was mostly degenerate
under the claimed-output frequency criterion: only one `standard_L24` group was
easy, so nearly all measured decisions were in the hard regime.

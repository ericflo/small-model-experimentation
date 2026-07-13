# Qwen3.5-4B Forced-Commit Jacobian Value Transport

**Status:** finished

This study first validates an explicit fixed-budget commit action, then asks
whether a scalar Jacobian continuation-value coordinate changes fresh answers
under that same action.

## Research Program

- Primary: `interpretability_and_diagnostics`.
- Secondary: `test_time_reasoning_budget`.
- Direct parents: `qwen35_4b_native_thought_jacobian_value_transport` and
  `qwen35_4b_native_thought_seam_budget_ladder`.
- Mechanism anchor: `qwen35_4b_jacobian_transport_control_replication`.
- Forced-interface warning: `qwen35_4b_answer_potential_trace_sft` (C51).

## Question

When Qwen3.5-4B does not naturally finish by a fixed thought budget, can an
explicit deployed “commit now” action expose a parseable, variable-quality
answer seam—and is continuation value at the last thought token both decodable
and causally writable in the replicated 24-coordinate J space?

## Hypothesis

The failed natural-close ladder showed active re-analysis through 1,024 tokens,
not exact looping. A budget controller that injects the standard `</think>`
token may therefore turn partially resolved thoughts into usable answers. If a
task-general value signal is present, fresh forced-policy continuations should
be rankable from J coordinates at the live prefix endpoint, and raising only
that scalar coordinate should improve new continuations beyond shuffled-axis,
exact random, answer-identity, raw, ActAdd, and non-J controls.

The policy is deliberately counterfactual to autonomous close. It is legitimate
only because calibration, value labels, causal evaluation, and any eventual
deployment use the exact same commit action. Nothing here may be described as a
natural seam.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Lens: exact replicated 24-coordinate context lens, SHA-256
  `e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`,
  layers 4--8.
- Backend: Transformers bf16 SDPA, unpadded batch one, audited KV cache.
- Fresh data: 16 seam-selection, 16 seam-confirmation, 32 value-fit, and
  32 causal-confirmation tasks; 96/96 unique and zero overlap with three parents.
- Task hardening: every visible set identifies one first-operation type and has
  no matching depth-one operation, so “exactly two” is behaviorally true.
- Sampling: temperature 0.6, top-p 0.95, top-k 20, three traces per task.
- Candidate caps: 256, 512, 1024. Selection uses paired trace prefixes; untouched
  confirmation opens only the smallest passing cap.
- Policy: if the model naturally closes before the cap, keep that answer;
  otherwise append one `</think>` token and generate at most 16 answer tokens.

## Frozen Gates

At selection and confirmation, policy parse and forced-only parse must each be
at least 90%, at least half the rows must actually require the forced action,
policy success must lie in 5%--95%, at least six tasks must mix correct and
incorrect policy outcomes, and answer-cap contact must be at most 5%.

Only `FORCED_COMMIT_SEAM_REPLICATED` opens value fitting. Prefixes at 0.5 and 1.0
of the selected cap receive three disjoint forced-policy continuations. The
held-out-by-task J readout must reach task-macro pairwise AUC 0.65, beat correct
alias activity by 0.03, and leave the within-task shuffled null near chance.

Only a value pass opens exact post-bf16 control calibration and one untouched
causal confirmation. The primary scalar clamp must improve paired success by at
least 0.10 with a positive bootstrap lower bound and beat exact random,
shuffled-axis, and matched non-J controls by frozen margins. Identity/full-donor
arms cannot rescue it.

## Run

CPU smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/qwen35_4b_forced_commit_jacobian_value_transport/scripts/run.py \
  --stage smoke
```

After the immutable design commit is anchored:

```bash
.venv/bin/python experiments/qwen35_4b_forced_commit_jacobian_value_transport/scripts/run.py --stage model-smoke
.venv/bin/python experiments/qwen35_4b_forced_commit_jacobian_value_transport/scripts/run.py --stage seam-selection
.venv/bin/python experiments/qwen35_4b_forced_commit_jacobian_value_transport/scripts/run.py --stage seam-confirmation
```

Later stages remain fatal placeholders until their audited implementations are
committed; the runner refuses to emit placeholder value or causal results.

## Status

Terminal `FORCED_COMMIT_SEAM_FAIL`. Design and the 46-threat adversarial review
were frozen before model calls; CPU and outcome-blind model smokes passed. The
complete 48-trace/144-policy-row selection found that appending close alone did
not reliably switch the model into answer mode. No cap was selected, and seam
confirmation, value fitting, controls, and causal outcomes remain sealed.

## Results

| cap | forced parse | exact success | mixed tasks | answer-cap contact | gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 6/48 (12.5%) | 1/48 (2.1%) | 1 | 44/48 (91.7%) | fail |
| 512 | 8/48 (16.7%) | 1/48 (2.1%) | 1 | 41/48 (85.4%) | fail |
| 1024 | 9/48 (18.8%) | 1/48 (2.1%) | 1 | 46/48 (95.8%) | fail |

All 48 traces required forced close at every cap. The run sampled 49,152 thought
tokens plus 2,225 answer tokens in 1,640.539 seconds. Typical post-close outputs
restarted analysis or emitted free-form reasoning instead of the requested slot.

A post-decision regex diagnostic tolerated aliases attached directly to special
EOS tokens. It raised parse counts only to 7/11/10 and correct counts to 1/2/2 at
256/512/1024. Those remain far below every frozen parse, success, mixed-task, and
answer-termination gate, so the parser edge case does not affect the decision.

## Scope

No value coordinate was fit and no activation was patched. The failure precedes
J space: a lone close token is not a usable answer-emission interface. The next
distinct experiment may supply syntax but not identity—append close plus the
fixed `First:` slot and read only the alias choice—while retaining free-form
forced answer as a control. It must use fresh tasks and a new adversarial review.

If a later value/causal stage becomes eligible, ground-truth continuations and
donors remain oracle. Capability still requires a non-oracle controller that
beats frozen Qwen, strongest controls, and matched-compute sampling.

## Knowledgebase Update

- Program ledgers: record terminal close-only interface failure.
- Shared synthesis: distinguishes close-token injection from an answer slot.
- Claim ledger: no claim ID while the repository claim re-grade is open.

## Artifacts

- `assets/context_lens.pt`: byte-identical replicated lens.
- `data/procedural/`: four frozen fresh splits and manifest.
- `runs/smoke/`: CPU/gate receipts.
- `reports/preregistration.md`: immutable decision rules.
- `reports/design_review.md`: adversarial review before model work.
- `reports/artifact_manifest.yaml`: reproduction/omission contract.

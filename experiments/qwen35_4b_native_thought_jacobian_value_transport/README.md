# Qwen3.5-4B Native-Thought Jacobian Value Transport

This experiment asks whether a task-general scalar continuation-value coordinate
inside the replicated 24-token J space is causally consumed from a natural token
inside Qwen3.5-4B's own `<think>` span.

## Research Program

- Primary: `interpretability_and_diagnostics`.
- Conditional after an oracle causal pass: `test_time_reasoning_budget` and
  `structured_execution_and_compilers`.
- Direct parent: `qwen35_4b_jacobian_transport_control_replication`, terminal
  `REPLICATED_J_TRANSPORT` on fresh lookup mappings with exact controls.
- Closest canceled design: `qwen35_4b_jacobian_value_transport`, whose native
  prefix stages were correctly forbidden when its late lens did not transport.
- Other anchors: `qwen35_4b_activation_steering`,
  `qwen35_4b_probe_to_prompt`, `qwen35_4b_thinking_separability_probe`, and
  `qwen35_4b_prefix_value_guided_search`.

## Question

Can continuation success be decoded from J coordinates at natural thought
prefixes and then changed by transferring only one learned certainty coordinate,
rather than copying an answer identity or merely sampling again?

## Fixed design

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Exact replicated 24-concept lens SHA-256
  `e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`.
- Frozen band `[4,5,6,7,8]`, natural thought checkpoints at 0.33/0.67,
  batch-one cache-free full-prefix recomputation.
- Eighty fresh, parent-disjoint procedural depth-2 list tasks. Exhaustive CPU
  enumeration guarantees the visible I/O has one identifiable first-operation
  type. A fixed one-token alias mapping connects the 12 operation types to lens
  concepts; 11 types are valid targets.
- Prefix value is the fraction of three disjoint-seed natural continuations that
  answer the correct first-operation alias. Whole-trace labels are not assigned
  to tokens.
- Primary G1 signal is a held-out-by-task scalar readout of concatenated J
  coordinates. Primary G2 intervention transfers only that scalar coordinate.
- Two exact post-bf16 random arms, shuffled value axis, answer-identity J clamps,
  logit-lens, raw donor, J/non-J donor decomposition, ActAdd, and wrong-task
  donor controls are frozen.

The full gates are in [preregistration.md](reports/preregistration.md), and the
24-threat pre-run review is in [design_review.md](reports/design_review.md).

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  experiments/qwen35_4b_native_thought_jacobian_value_transport/tests -q
.venv/bin/python experiments/qwen35_4b_native_thought_jacobian_value_transport/scripts/run.py --stage smoke
.venv/bin/python experiments/qwen35_4b_native_thought_jacobian_value_transport/scripts/run.py --stage model-smoke
.venv/bin/python experiments/qwen35_4b_native_thought_jacobian_value_transport/scripts/run.py --stage seam-calibration
.venv/bin/python experiments/qwen35_4b_native_thought_jacobian_value_transport/scripts/run.py --stage prefix-value
.venv/bin/python experiments/qwen35_4b_native_thought_jacobian_value_transport/scripts/run.py --stage control-calibration
.venv/bin/python experiments/qwen35_4b_native_thought_jacobian_value_transport/scripts/run.py --stage causal-confirmation
```

Every stage is fatal-gated. Model smoke and seam calibration are implemented;
later stages still refuse placeholders until their audited implementations land.

## Results

Terminal frozen decision: `NO_NATURAL_SEAM`. The generator produced 16 seam, 32 value-fit, and 32
causal-confirmation tasks: 80/80 unique fingerprints, zero overlap with the
direct Jacobian parent, balanced identifiable first-operation targets, and the
exact frozen lens hash.

The two-task model smoke validates revision, token IDs, one-token aliases, full
24-rank dictionaries, cache-free generation, and finite J coordinates without
recording correctness. Both traces hit the frozen 160-token cap without natural
close, and historical-token activations changed by up to 0.0625 across suffix
lengths, so causal invariance currently fails. These are scientific seam/control
risks.

The frozen 16-task seam confirmed the failure on all 48 traces:

- natural close: 0/48;
- parseable final alias: 0/48;
- exact success: 0/48;
- mixed correct/incorrect tasks: 0/16; and
- thought-cap contact: 48/48 at exactly 160 tokens (7,632 cache-free forwards).

The natural-close/parse/headroom gates fail, so prefix-value fitting, numeric
controls, and causal confirmation are canceled. This is an interface-budget
failure, not evidence that a J-space value coordinate is absent. The result is
preserved rather than raising the budget in place.

## Scope

Had the seam passed, ground-truth continuation outcomes would have fit the value
readout and selected causal donors, making later evidence oracle-only. Those
stages did not run. Any successor still requires a separate non-oracle controller
to beat frozen and matched-compute sampling on new tasks.

## Knowledgebase Update

- Idea intake and decision record connect this to the replicated mechanism.
- Program evidence changes only at a scientific gate.
- No claim number is reserved while the repository claim re-grade remains open.

## Artifacts

All procedural data, receipts, frozen weights, control geometry, and reports are
self-contained here. No adapter or external artifact is used at the design stage.

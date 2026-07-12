# Qwen3.5-4B Jacobian Value Transport

This experiment tests whether token-aligned Jacobian coordinates can carry
causal value from native thinking into fresh, exactly scored answers more
specifically than ordinary activation steering.

## Research Program

- Primary program: `interpretability_and_diagnostics`
- Secondary programs if the causal gate passes: `test_time_reasoning_budget`,
  `structured_execution_and_compilers`, and `posttraining_and_adaptation`
- Prior anchors: C19/C20 (decodable first operation but inert ActAdd), C30
  (decode-to-prompt works), C40/C42 (implicit confidence and step-local error
  signal), C51 (forced-close answer potential is an unreachable-state scorer),
  and C52 (token-local outcome labels do not make LoRA updates context-local).

## Question

Does Qwen3.5-4B expose sparse, token-aligned Jacobian coordinates that are not
only readable but causally transported from native thinking into later answers,
and can coordinate replacement change verifier-scored outcomes more specifically
than ordinary activation addition?

## Hypothesis

The residual stream contains two separable quantities: the strength of a
candidate intermediate representation and the local downstream gain from that
representation to the answer. C20 manipulated only the first with a global
mean-difference vector. A Jacobian-lens coordinate swap should be more effective
because it uses a model-native downstream-readable direction, removes a competing
coordinate while adding the target, and can be clamped through the layer band in
which transport is open.

The causal premise counts only if a J-coordinate intervention changes exact
future outcomes beyond matched random, wrong-donor, shuffled-label, raw-activation,
non-J/remainder, logit-lens, and ActAdd controls. Decoding or outcome prediction
alone is not a positive result.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16 Transformers inference.
- Substrate: fresh procedural list transformations generated inside this
  experiment. String and register transformations are held-family checks. No
  benchmark content is read or used.
- Splits: disjoint lens-fit corpus, positive-control calibration, value
  calibration, IID evaluation, held-family evaluation, and hard-depth evaluation.
- Primary baseline: the frozen model under the same HF backend, prompts, seeds,
  decoding budget, and full-prefix recomputation used by intervention arms.
- Primary metrics: intervention-induced target-concept rate on positive controls;
  within-task prefix-value AUROC; and exact verifier success after causal patching.
- Oracle-only evidence: correct-operation directions, reference-answer margins,
  and high-value donor selection. These establish an upper-bound causal mechanism
  and cannot support a deployable claim.
- Hidden-label boundary: exact operation and hidden examples may score or construct
  oracle diagnostics, but no non-oracle controller may receive them as inputs.

The frozen gates and analysis rules are in
[`reports/preregistration.md`](reports/preregistration.md); the adversarial review
is in [`reports/design_review.md`](reports/design_review.md).

## Run

CPU tests and data smoke:

```bash
.venv/bin/python -m pytest experiments/qwen35_4b_jacobian_value_transport/tests -q
.venv/bin/python experiments/qwen35_4b_jacobian_value_transport/scripts/run.py --stage smoke
```

Real-model plumbing smoke, after the immutable design commit is recorded:

```bash
.venv/bin/python experiments/qwen35_4b_jacobian_value_transport/scripts/run.py --stage model-smoke
```

Scientific stages are individually restartable and gate the next stage:

```bash
.venv/bin/python experiments/qwen35_4b_jacobian_value_transport/scripts/run.py --stage fit-lens
.venv/bin/python experiments/qwen35_4b_jacobian_value_transport/scripts/run.py --stage positive-control
.venv/bin/python experiments/qwen35_4b_jacobian_value_transport/scripts/run.py --stage prefix-value
.venv/bin/python experiments/qwen35_4b_jacobian_value_transport/scripts/run.py --stage causal-patch
```

`--stage full` runs the permitted sequence and refuses to cross a failed gate.

## Results

Terminal G0 verdict: **`NO_J_WRITING` under the frozen decision rule**, with a
more informative mechanistic split: **late direct writing worked; causal
transport did not**.

- Clean confirmation accuracy was 24/24 for direct concept report and 24/24
  for the prompt-local mapped consequence.
- At layer 24 and the selection-chosen alpha 4, the J swap changed the direct
  report to the target concept on 18/24 items (75%). Random stayed 0/24 and the
  ordinary logit-lens swap reached 5/24 (20.8%).
- The same J intervention changed the mapped downstream consequence on 0/24
  items at every tested layer. Its target-minus-source margin was essentially
  flat as alpha rose from 0.5 to 4.
- Earlier layers were inert; layer 20 moved direct report on only 1/24, so no
  adjacent-layer pair passed. G1 prefix value and G2 task patching were correctly
  cancelled.

The full terminal result is in [`reports/report.md`](reports/report.md).

## Interpretation

The averaged token pullback is a strong late output-control coordinate on this
model, not yet evidence of a reusable reasoning workspace. It can overwrite what
concept the model says, but the tested intervention does not make the model
recompute an arbitrary consequence from that concept. The next warranted test is
a separate experiment using context-local transport, set-to-target clamping across
an earlier layer band, and exact per-example delta-norm controls.

## Knowledgebase Update

- Program evidence: updated after synchronizing the terminal result to `origin/main`.
- Program backlog: branch to context-local clamped transport in a new experiment.
- Claim ledger: unclaimed while the repository-wide adversarial claim re-grade
  is open; no number is reserved.

## Artifacts

- Small split manifests, smoke receipts, metrics, and reports are committed.
- Full Jacobian matrices and activation caches remain external and checksummed in
  `reports/artifact_manifest.yaml`.
- No adapter is trained in this causal-premise experiment.

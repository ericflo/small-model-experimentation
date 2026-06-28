# End-to-End Structured Slot Executor Experiment Log

## Objective

Test whether a single learned structured slot executor can solve modular
belief-state programs without oracle initialization or exact transition.

Each example starts from:

```text
B = A + d (mod p), with A unknown
```

The model must place that support into weighted slots, then recurrently execute
arithmetic updates and observation filters. The main learned executor combines
a Sinkhorn-normalized cyclic initializer with a learned transition router over
equivariant modular update primitives.

## Primary Questions

1. Can the full structured executor train end-to-end from scratch?
2. Does the learned transition remain reliable when the initializer is learned
   rather than oracle supplied?
3. Does the Sinkhorn initializer remain reliable when transition gradients are
   trained jointly?
4. Which component is the remaining bottleneck: initialization, transition
   routing, or their interaction?
5. How far does the best learned executor get on held-out program lengths?

## Metrics

- `decoder_belief_target_mass`: probability assigned to the exact final
  `(A,B)` support.
- `decoder_query_target_mass`: probability assigned to the exact final query
  support after projecting the decoded belief.
- `init_belief_target_mass`: probability assigned to the exact initial
  `(A,B)` support before recurrent steps.
- `init_slot_unique_a_frac`: fraction of residues covered by distinct
  slot-level `A` argmaxes.
- `mean_route_accuracy`: whether the transition router's highest-probability
  route matches the true operation.
- `mean_route_entropy`: entropy of transition routing.

The strict headline metric is `decoder_belief_target_mass` at the first
evaluated `K >= L`.

## Artifact Layout

- Code and lightweight outputs:
  `experiments/end_to_end_structured_slot_executor/`
- Checkpoints:
  `large_artifacts/end_to_end_structured_slot_executor/checkpoints/`
- Run outputs:
  `experiments/end_to_end_structured_slot_executor/runs/<variant>/`
- Analysis outputs:
  `experiments/end_to_end_structured_slot_executor/analysis/`

## Planned Sequence

1. Smoke tests on modulus 7 to verify the combined learned executor and
   component controls.
2. Pilot runs on modulus 11 with held-out lengths up to 12.
3. Main runs on modulus 31 with held-out lengths up to 24.
4. Optional scale check if the main end-to-end row is strong.
5. Generate analysis tables, figures, checkpoint manifest, standalone report,
   HTML report, and final audit.

## Variant Plan

- `oracle_exact`: exact initializer and exact transition ceiling.
- `sinkhorn_exact`: learned Sinkhorn initializer with exact transition.
- `oracle_primitive_router`: oracle initializer with learned primitive router.
- `sinkhorn_primitive_router`: full structured end-to-end executor.
- `sinkhorn_cyclic_mixer`: softer learned transition control.
- `generic_primitive_router`: generic initializer ablation with structured
  transition.
- `sinkhorn_mlp`: structured initializer with unstructured transition ablation.

## 2026-06-21 Setup

Created the standalone experiment directory:

- `experiments/end_to_end_structured_slot_executor/src/`
- `experiments/end_to_end_structured_slot_executor/reports/`
- `experiments/end_to_end_structured_slot_executor/runs/`
- `experiments/end_to_end_structured_slot_executor/analysis/figures/`
- `large_artifacts/end_to_end_structured_slot_executor/checkpoints/`

Implemented the combined harness:

- Structured Sinkhorn cyclic initializer.
- Generic MLP and oracle initializer controls.
- Exact, MLP, cyclic-mixer, and primitive-router transitions.
- Full-prefix belief supervision.
- Initializer metrics and route diagnostics.
- External checkpoint writing.

Next action: compile the source and run modulus-7 smoke tests.

## 2026-06-21 Smoke Tests

Source compilation passed for both experiment scripts.

Ran modulus-7 smoke variants with evaluation lengths 2 and 3:

| Variant | Init | Transition | Steps | L=3 query | L=3 belief | Initial belief | Route acc |
|---|---|---|---:|---:|---:|---:|---:|
| `smoke_oracle_exact` | oracle | exact | 0 | 100.0% | 100.0% | 100.0% | n/a |
| `smoke_oracle_primitive_router` | oracle | primitive router | 500 | 100.0% | 100.0% | 100.0% | 100.0% |
| `smoke_sinkhorn_exact` | sinkhorn cyclic | exact | 500 | 98.4% | 96.0% | 96.9% | n/a |
| `smoke_sinkhorn_primitive_router` | sinkhorn cyclic | primitive router | 700 | 87.1% | 69.1% | 73.9% | 100.0% |
| `smoke_sinkhorn_primitive_router_lr01` | sinkhorn cyclic | primitive router | 900 | 99.4% | 98.6% | 98.9% | 100.0% |
| `smoke_sinkhorn_cyclic_mixer_lr01` | sinkhorn cyclic | cyclic mixer | 900 | 91.5% | 82.5% | 89.0% | 76.2% |
| `smoke_generic_primitive_router` | generic MLP | primitive router | 700 | 87.9% | 72.8% | 75.5% | 100.0% |
| `smoke_sinkhorn_mlp` | sinkhorn cyclic | MLP | 700 | 90.6% | 77.0% | 61.0% | n/a |

Smoke interpretation:

- The ceiling rows validate generation, decoding, K-indexed evaluation, and
  route diagnostics.
- Primitive routing is easy under oracle initialization and remains easy under
  learned Sinkhorn initialization once the initializer trains fast enough.
- Joint Sinkhorn+router training is sensitive to learning rate. At `lr=0.003`
  the router learns but the initializer lags; at `lr=0.01` the full executor
  reaches 98.6% strict belief at held-out length 3.
- The softer cyclic mixer underperforms because route selection remains
  diffuse.
- The generic initializer and unstructured transition ablations are both
  clearly below the tuned structured executor.

Pilot decision:

- Promote `sinkhorn_primitive_router_lr01` as the primary full executor.
- Keep `oracle_exact`, `sinkhorn_exact`, and `oracle_primitive_router` as
  component controls.
- Keep `generic_primitive_router` and `sinkhorn_mlp` as ablations.
- Do not promote the cyclic mixer unless the primitive-router row fails at
  larger modulus.

## 2026-06-21 Pilot Sweep

Ran modulus-11 pilot variants with training lengths 1-6 and evaluation lengths
3, 6, 9, and 12:

| Variant | Init | Transition | Steps | L=12 query | L=12 belief | Initial belief | Route acc |
|---|---|---|---:|---:|---:|---:|---:|
| `pilot_oracle_exact` | oracle | exact | 0 | 100.0% | 100.0% | 100.0% | n/a |
| `pilot_oracle_primitive_router` | oracle | primitive router | 600 | 100.0% | 100.0% | 100.0% | 100.0% |
| `pilot_sinkhorn_exact` | sinkhorn cyclic | exact | 900 | 98.2% | 97.3% | 98.8% | n/a |
| `pilot_sinkhorn_primitive_router` | sinkhorn cyclic | primitive router | 1400 | 99.2% | 98.9% | 99.5% | 100.0% |
| `pilot_generic_primitive_router` | generic MLP | primitive router | 1000 | 69.5% | 59.5% | 60.4% | 100.0% |
| `pilot_sinkhorn_mlp` | sinkhorn cyclic | MLP | 1000 | 78.4% | 70.2% | 49.2% | n/a |

Pilot interpretation:

- The learned primitive router remains exact under both oracle and learned
  Sinkhorn initialization.
- The full structured executor reaches near-ceiling held-out performance at
  modulus 11.
- The generic initializer ablation fails because it does not cover the initial
  support.
- The MLP transition ablation fails despite a structured initializer, showing
  that transition structure is still required.

Main decision:

- Run modulus-31 main rows for `oracle_exact`, `oracle_primitive_router`,
  `sinkhorn_exact`, `sinkhorn_primitive_router`, `generic_primitive_router`,
  and `sinkhorn_mlp`.
- Keep evaluation lengths 4, 8, 12, 16, and 24.
- Use the tuned high learning rate for Sinkhorn structured rows.

## 2026-06-21 Main Sweep

Ran modulus-31 main variants with training lengths 1-8 and evaluation lengths
4, 8, 12, 16, and 24:

| Variant | Init | Transition | Steps | L=24 query | L=24 belief | Initial belief | Route acc |
|---|---|---|---:|---:|---:|---:|---:|
| `main_oracle_exact` | oracle | exact | 0 | 100.0% | 100.0% | 100.0% | n/a |
| `main_oracle_primitive_router` | oracle | primitive router | 800 | 99.9% | 99.9% | 100.0% | 100.0% |
| `main_sinkhorn_exact` | sinkhorn cyclic | exact | 1800 | 97.7% | 96.8% | 99.6% | n/a |
| `main_sinkhorn_primitive_router` | sinkhorn cyclic | primitive router | 2400 | 98.7% | 98.1% | 99.7% | 100.0% |
| `main_generic_primitive_router` | generic MLP | primitive router | 1400 | 43.0% | 28.7% | 58.6% | 100.0% |
| `main_sinkhorn_mlp` | sinkhorn cyclic | MLP | 1600 | 19.9% | 5.0% | 58.9% | n/a |

Main interpretation:

- The full learned structured executor succeeds end-to-end at modulus 31,
  reaching 98.1% strict belief at held-out length 24.
- The primitive-router transition remains exact when trained jointly with the
  learned Sinkhorn initializer.
- Both learned pieces are necessary. A generic initializer with a perfect
  learned router fails, and a structured initializer with an unstructured MLP
  transition fails.
- The full learned row slightly exceeds the Sinkhorn+exact component control at
  length 24 in this run, likely because the learned primitive router preserves
  sharper slot-local distributions.

Scale decision:

- Run a modulus-97 full-program scale check for `sinkhorn_primitive_router`.
- Include `oracle_exact` as the standalone ceiling.
- Use a smaller batch/evaluation budget to keep the p97 dense pair target
  tractable.

## 2026-06-21 Scale Check

Ran modulus-97 scale variants with training lengths 1-8 and evaluation lengths
4, 8, 12, 16, and 24:

| Variant | Init | Transition | Steps | L=24 query | L=24 belief | Initial belief | Route acc |
|---|---|---|---:|---:|---:|---:|---:|
| `scale_oracle_exact` | oracle | exact | 0 | 100.0% | 100.0% | 100.0% | n/a |
| `scale_sinkhorn_primitive_router` | sinkhorn cyclic | primitive router | 1800 | 96.2% | 94.9% | 99.4% | 100.0% |

Scale interpretation:

- The full learned structured executor remains strong at modulus 97, reaching
  94.9% strict belief at held-out length 24.
- Initial support formation and route selection are not the limiting factors:
  initial belief is 99.4%, slot coverage is 100.0%, and route accuracy is
  100.0%.
- The remaining loss appears as gradual belief diffusion over long recurrent
  programs rather than a discrete routing failure.
- The p97 row is strong enough to include as a scale result in the standalone
  report, but it is not exact; the limitation should be stated directly.

## 2026-06-21 Final Audit

Final artifacts created:

- `reports/end_to_end_structured_slot_paper.md`
- `reports/end_to_end_structured_slot_paper.html`
- `checkpoint_manifest.csv`

Verification:

- Source compilation passed:
  `python -m py_compile src/end_to_end_structured_slot_experiment.py src/analyze_end_to_end_structured_slot.py`
- Checkpoint manifest validation passed for 19 saved checkpoints.
- Markdown and HTML report image references resolve.
- No `.pt`, `.pth`, or `.ckpt` files are stored inside the lightweight
  experiment directory.
- The standalone wording scan passed for the listed external-lineage phrases.
- Removed the compile cache after verification.

Artifact sizes:

- `experiments/end_to_end_structured_slot_executor/`: 8.8M
- `large_artifacts/end_to_end_structured_slot_executor/`: 3.8M

Conclusion:

The strongest learned row is `sinkhorn_primitive_router`. It reaches 98.1%
strict belief at modulus 31 and held-out length 24, then reaches 94.9% strict
belief at modulus 97 and held-out length 24. The component controls show that
both structured support formation and structured transition routing are needed.

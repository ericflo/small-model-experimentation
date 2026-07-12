# Qwen3.5-4B Specialist Policy Integration

Status: **stopped negative** on 2026-07-12 before best-of-8 or specialist
training. The tools core's incumbent score is 0.994, making its frozen
`S0 + 0.10` qualification target 1.094 on a score bounded by 1.0. Because all
four specialists were mandatory, teacher audit and integration are unlicensed.

This experiment tests whether independently execution-improved, same-origin
specialists can be consolidated on the student's own trajectories into one
`Qwen/Qwen3.5-4B` policy that composes their capabilities on held-out tasks.

## Research Programs

- Primary: `agentic_breadth_installation`.
- Supporting: `posttraining_and_adaptation`, `benchmark_generalization`.
- Closest near-duplicate: `qwen35_4b_interactive_policy_curriculum`, which
  trains one mixed policy and does not test specialist integration.
- Strategic source:
  [`knowledge/decision_records/2026-07-11_specialize_distill_compose.md`](../../knowledge/decision_records/2026-07-11_specialize_distill_compose.md).

## Question and Hypothesis

Does execution reward first create policies with real headroom, and can
same-observation on-policy multi-teacher distillation integrate that headroom
without the privileged-context shortcuts, exposure bias, or mixture see-saw
seen in adjacent methods?

The mechanism is supported only if qualified specialists improve exact
student-prefix continuations, correctly routed MOPD beats wrong routing and
matched integration controls, and the final student exceeds every individual
teacher on never-trained compound pairings/depths.

## Substrate

Primitive specialist domains:

- discovery/repair: `glyphgate`, `loomfix`;
- stateful control: `kilnrite`, `burrowmaze`;
- tools/provenance: `ferrier` plus permitted `foundry_ledger` atom replay.

Composition specialist training:

- `cipherkiln`: infer a cyclic code mapping, then execute a legal protocol;
- `mazeferry`: explore a partial map, find tools, and carry typed handles
  through a dependency chain.

No-new-exposure/held-out evaluation:

- primitive: `patchwheel`, `spindle`, `gatepost`;
- compound: `patchferry`, `tripleforge`, plus order reversals.

Every compound family has an exact oracle and explicit primitive-removal
policies. The CPU smoke currently records oracle score 1.0 and ablation full-
success 0.0 at every L1-L4 cell.

## Stages and Stop Rules

1. Regenerate and merge the C53 incumbent `S0`; pass HF/vLLM and nonzero-
   composite gates.
2. Verify that every frozen pass-one specialist gain bar has mathematical
   headroom under the environment score ceiling. This postmortem gate now
   stops before best-of-8 or training when a target is unreachable.
3. Produce four DAgger-to-execution-RL specialists. Every specialist must beat
   DAgger, extra SFT, shuffled reward, and `S0` best-of-8.
4. Audit correct versus KL-matched wrong teachers on exact `S0` prefixes and
   run the five-update exact-logit locality pilot.
5. Integrate qualified teachers with corrected top-50 MOPD. Compare end-to-end
   matched joint RL, off-policy SFT, parameter merge, and wrong routing.
6. Evaluate three seeds on individual domains, primitive transfer, and held-out
   compounds. Open the benchmark CLI only if every whitebox gate passes.

The exact thresholds, seeds, metrics, and interpretation are frozen in
[`reports/preregistration.md`](reports/preregistration.md) and
[`configs/default.yaml`](configs/default.yaml). The adversarial review is
[`reports/design_review.md`](reports/design_review.md).

## Firewall

- The only model is `Qwen/Qwen3.5-4B` at the pinned revision.
- Nothing under `benchmarks/` is imported, read, or used for training.
- Programmatic state labels DAgger but is never placed in model input.
- MOPD teacher and student see the identical observable prompt and prefix.
- All comparable generations use the same pinned vLLM backend.
- Transfer families are excluded from every new training and replay row.

## Run

CPU scientific smoke:

```bash
python3 experiments/qwen35_4b_specialist_policy_integration/scripts/run.py --smoke
```

The smoke writes `runs/smoke/summary.json` and verifies all compound oracles,
necessity ablations, state-aware experts, and split/replay invariants.

Reached model stages are resumable and fail closed on missing upstream receipts:

```bash
python3 experiments/qwen35_4b_specialist_policy_integration/scripts/run.py --stage model-smoke
python3 experiments/qwen35_4b_specialist_policy_integration/scripts/run.py --stage incumbent
python3 experiments/qwen35_4b_specialist_policy_integration/scripts/run.py --stage calibration-gate
python3 experiments/qwen35_4b_specialist_policy_integration/scripts/run.py --stage baseline-eval
python3 experiments/qwen35_4b_specialist_policy_integration/scripts/run.py --stage dagger-collect --domain discover
python3 experiments/qwen35_4b_specialist_policy_integration/scripts/run.py --stage dagger-train --domain discover
python3 experiments/qwen35_4b_specialist_policy_integration/scripts/run.py --stage rl-collect --domain discover
python3 experiments/qwen35_4b_specialist_policy_integration/scripts/run.py --stage specialist-train --domain discover
python3 experiments/qwen35_4b_specialist_policy_integration/scripts/run.py --stage controls --domain discover
python3 experiments/qwen35_4b_specialist_policy_integration/scripts/run.py --stage specialist-eval --domain discover
python3 experiments/qwen35_4b_specialist_policy_integration/scripts/run.py --stage specialist-analyze --domain discover
```

Replace `discover` with `control`, `tools`, or `compose`. Teacher-audit and
integration stages remain deliberately unavailable until all specialist gates
pass; a request cannot silently bypass the stop hierarchy. In the reached
result, `--stage baseline-eval` resumes the committed greedy baseline, rewrites
the negative headroom receipt deterministically, and stops before best-of-8.

## Current Results

Reached evidence:

- four compound families deterministic and JSON-safe;
- exact oracle 1.0 at every L1-L4 cell;
- generic random policy 0.0;
- every registered primitive-removal policy full-success 0.0; and
- state-aware live experts solve every cell at 1.0.
- the pinned model/runtime answered 4/4 generic semantic smoke prompts and
  honored the explicit CUDA-graph geometry on the live L40.
- the pinned Transformers path produced finite logits with both Qwen fast-path
  extensions, a two-step QLoRA made 128/128 nonzero mapped deltas, and vLLM
  loaded the explicitly merged local composite.
- a one-step QLoRA smoke exited successfully but yielded an all-zero adapter;
  the merge gate rejected and preserved it, demonstrating that Trainer exit
  status alone is not an installation check.
- the full incumbent ran all 333 optimizer steps over 2,117 encoded rows
  (123/2,240 rows skipped at the frozen 2,048-token cap), then produced
  128/128 nonzero explicitly mapped deltas with summed Frobenius norm 161.39;
  the CUDA FP32/no-TF32 merged composite is weight-hashed in its receipt.
- all 7/7 frozen visible-prefix canaries changed versus the pinned base under
  identical greedy prompts, runner hash, sampling, CUDA graphs, and runtime
  lock. The aggregate incumbent provenance/install gate passed every check.
- on a disjoint 288-episode compound calibration (four families, L2-L4,
  24/cell), the incumbent scored 0.135 macro: `cipherkiln` 0.227,
  `mazeferry` 0.296, `patchferry` 0.012, and `tripleforge` 0.005. The strict
  `<0.60` headroom gate and every scope/decode/seed/atom-firewall check passed.

The full paired greedy baseline then resolved feasibility:

| Specialist core | Frozen families | `S0` macro | Required score | Feasible under cap 1.0 |
| --- | --- | ---: | ---: | --- |
| discover | `glyphgate`, `loomfix` | 0.513 | 0.613 | yes |
| control | `kilnrite`, `burrowmaze` | 0.523 | 0.623 | yes |
| tools | `ferrier` | 0.994 | 1.094 | **no** |
| compose | `cipherkiln`, `mazeferry` | 0.180 | 0.280 | yes |

The all-process macro was 0.458 over 864 episodes; atom-retention macro was
0.681 over 1,344 items. Every baseline protocol check passed. The originally
scheduled all-family best-of-8 was interrupted during engine warmup before any
sampled output, and the new feasibility gate now deterministically refuses it.

This is a design-negative result, not evidence for or against MOPD/OPSD: the
experiment failed to provide a falsifiable four-teacher test because one
mandatory improvement bar was impossible before training. No DAgger, GRPO,
specialist, teacher-audit, integration, confirmatory, or benchmark stage ran.
Any follow-up must live in a new experiment, retain the 0.10 bar, and calibrate
headroom independently for every specialist domain before GPU production.

## Artifacts

- `src/gym/families/compound_core.py`: shared exact compound mechanics.
- `src/curriculum.py`: state-aware expert interface.
- `runs/smoke/summary.json`: committed CPU gate receipt.
- `analysis/specialist_headroom_gate.json`: terminal negative stop receipt.
- `runs/proxy_eval/incumbent_calibration/`: paired greedy/atom baseline.
- `reports/artifact_manifest.yaml`: external checkpoint policy.
- future large weights: `large_artifacts/qwen35_4b_specialist_policy_integration/`.

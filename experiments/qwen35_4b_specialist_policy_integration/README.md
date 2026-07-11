# Qwen3.5-4B Specialist Policy Integration

Status: the C53 incumbent was regenerated and passed structural plus paired
behavioral installation gates on 2026-07-11. Its disjoint compound-headroom
calibration is running; no specialist or integration result exists yet.

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
2. Produce four DAgger-to-execution-RL specialists. Every specialist must beat
   DAgger, extra SFT, shuffled reward, and `S0` best-of-8.
3. Audit correct versus KL-matched wrong teachers on exact `S0` prefixes and
   run the five-update exact-logit locality pilot.
4. Integrate qualified teachers with corrected top-50 MOPD. Compare end-to-end
   matched joint RL, off-policy SFT, parameter merge, and wrong routing.
5. Evaluate three seeds on individual domains, primitive transfer, and held-out
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
python3 experiments/qwen35_4b_specialist_policy_integration/scripts/run.py --stage calibrate
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
pass; a request cannot silently bypass the stop hierarchy.

## Current Results

Current pre-task evidence is limited to substrate and runtime validity:

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

This licenses incumbent evaluation, not a capability, teacher, integration,
transfer, or benchmark conclusion.

## Artifacts

- `src/gym/families/compound_core.py`: shared exact compound mechanics.
- `src/curriculum.py`: state-aware expert interface.
- `runs/smoke/summary.json`: committed CPU gate receipt.
- `reports/artifact_manifest.yaml`: external checkpoint policy.
- future large weights: `large_artifacts/qwen35_4b_specialist_policy_integration/`.

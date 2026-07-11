# Qwen3.5-4B Specialist Policy Integration Report

## Status

Implementation and the complete runtime/training/merge preflight passed. The
C53 incumbent was regenerated and passed its structural and behavioral
installation gates; compound-headroom calibration is running. No specialist
or integration capability claim is available.

## Research Program Fit

This is the registered beyond-C53 mechanism for `agentic_breadth_installation`:
execution-reward RL produces headroom and on-policy multi-teacher distillation
attempts to integrate it. It also directly tests post-training interference and
held-out composition.

## Reached Evidence

`runs/smoke/summary.json` records:

- exact oracle score 1.0 for `cipherkiln`, `mazeferry`, `patchferry`, and
  `tripleforge` at every L1-L4 cell;
- generic random policy score 0.0;
- all discovery/control/navigation/repair/tool removal policies at 0.0 full
  success; and
- state-aware live expert score 1.0 in all 16 family/level cells.

This establishes substrate validity only. It does not show that the fixed model
can learn any primitive or composition.

The generic runtime smoke loaded the pinned revision on the live NVIDIA L40,
resolved the requested full-decode CUDA-graph sizes exactly, and answered 4/4
format/semantic probes. It validates the inference path only.

The Transformers smoke found finite padded-vocabulary logits and both required
Qwen fast paths. A two-step rank-32 QLoRA produced 128 nonzero composite-mapped
deltas (summed Frobenius norm 8.742), and the merged checkpoint loaded through
the same vLLM path. The first one-step attempt is a preserved negative: Trainer
reported success, but every delta was zero and the merge correctly refused it.

The full incumbent then completed 333/333 optimizer steps in 3,054.4 seconds
on the live L40. The frozen 2,048-token encoder admitted 2,117/2,240 rows and
skipped 123 (5.49%): 116 skipped rows were forced-close atoms and seven were
episodes, a disclosed concentration inherited from the exact C53 recipe. The
explicit merge applied 128/128 nonzero deltas (summed norm 161.39, maximum
2.90) on CUDA in FP32 with TF32 disabled. All seven frozen visible-prefix
canaries changed versus the pinned base while prompt, runner, sampling, graph,
and environment-lock metadata matched. `analysis/incumbent_gate.json` passes
all source-data, encoding, hyperparameter, optimizer, merge, and installation
checks. This proves the intended checkpoint was installed; it does not yet
show compound headroom or improvement.

## Pending

- disjoint C53 incumbent compound-headroom calibration;
- four specialist/control runs and qualification receipts;
- same-prefix teacher/locality audit;
- MOPD and matched integration controls;
- three-seed confirmatory evaluation and, only if eligible, benchmark CLI;
- program/claim/synthesis updates based on the reached result.

## Artifact Manifest

See `artifact_manifest.yaml`. Large adapters and merged checkpoints remain
external and receive checksums and regeneration commands when produced.

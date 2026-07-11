# Qwen3.5-4B Specialist Policy Integration Report

## Status

Implementation and the complete runtime/training/merge preflight passed. No
gym baseline or result-bearing training result exists, so no capability claim
is available.

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

## Pending

- C53 incumbent regeneration and calibration;
- four specialist/control runs and qualification receipts;
- same-prefix teacher/locality audit;
- MOPD and matched integration controls;
- three-seed confirmatory evaluation and, only if eligible, benchmark CLI;
- program/claim/synthesis updates based on the reached result.

## Artifact Manifest

See `artifact_manifest.yaml`. Large adapters and merged checkpoints remain
external and receive checksums and regeneration commands when produced.

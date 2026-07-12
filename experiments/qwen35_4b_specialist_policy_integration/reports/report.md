# Qwen3.5-4B Specialist Policy Integration Report

## Status

**Stopped negative on 2026-07-12.** Runtime, incumbent installation, and
compound-headroom gates passed, but the paired greedy baseline made the tools
specialist's frozen gain bar mathematically unreachable. No specialist or
integration capability claim is available.

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

The disjoint compound-only calibration then evaluated 288 greedy episodes at
L2-L4 (24 per family/level) under episode seed base 80500, with atom generation
disabled. The family means were 0.2271 `cipherkiln`, 0.2961 `mazeferry`,
0.0115 `patchferry`, and 0.0053 `tripleforge`, for macro 0.1350 against the
pre-registered exclusive ceiling 0.60. Exact success was 5.56%, 1.39%, 0%,
and 0%. Every protocol check passed, and the compacted artifact contains no
simulator spec, expert label, message transcript, or raw top-20 payload. This
establishes the intended headroom and licenses specialist production; it is
not evidence that a training method improved the model.

## Terminal Feasibility Result

The full paired greedy baseline evaluated all 12 process families at L2-L4,
24 episodes/cell (864 episodes), plus 1,344 atom-retention items. It used the
committed clean runner/environment, 3,893,188 sampled episode tokens,
4,909,001 logical episode-input tokens, and 4,991.3 wall seconds. Process macro
was 0.4582 and atom macro 0.6806.

| Core | Incumbent macro | Frozen `+0.10` target | Maximum possible gain | Feasible? |
| --- | ---: | ---: | ---: | --- |
| discover | 0.5127 | 0.6127 | 0.4873 | yes |
| control | 0.5230 | 0.6230 | 0.4770 | yes |
| tools | **0.9940** | **1.0940** | **0.0060** | **no** |
| compose | 0.1797 | 0.2797 | 0.8203 | yes |

All registered environment scores are bounded in `[0, 1]`. Therefore no
possible `ferrier` specialist can satisfy the frozen `S0 + 0.10` requirement
on these paired cells. This is not sampling uncertainty: the decision rule
uses the observed paired baseline, and its required score is above the score
contract's hard ceiling. Because the preregistration requires all four
specialists, the experiment stops before best-of-8, DAgger, execution RL,
controls, teacher audit, or integration.

The old all-family best-of-8 subprocess was interrupted during vLLM warmup,
before any sampled request or output artifact. A committed fail-closed analyzer
now reproduces the negative receipt and blocks that spend automatically.

## Interpretation

This experiment did **not** test whether specialist RL creates headroom or
whether MOPD integrates it. It found a prior design error: compound-level
headroom does not imply headroom for every mandatory teacher. The adversarial
review correctly guarded average-masking after training but missed the simpler
upper-bound feasibility check before training.

The most promising follow-up is a new experiment—not a threshold amendment:

1. keep the same regenerated incumbent, discovery/control/compose cores,
   installation gates, corrected MOPD loss, and matched controls;
2. replace or structurally harden the saturated tools/provenance core using a
   disjoint calibration pool (an existing provenance candidate such as
   `gatepost` has measured room, but requires a new split and held-out family);
3. require every core's `1 - S0_macro` to exceed every frozen absolute gain bar
   before best-of-8 or any training; and
4. freeze fresh paired seeds only after this domain-level feasibility gate.

Lowering the `+0.10` bar, dropping the tools teacher, or reusing the current
evaluation cells to tune a replacement would answer a different question.

## Unreached by Design

- execution-filtered best-of-8 (zero sampled outputs);
- all four specialist/control pipelines;
- same-prefix teacher/locality audit;
- MOPD and matched integration controls;
- confirmatory and benchmark evaluation.

## Artifact Manifest

See `artifact_manifest.yaml`. Large adapters and merged checkpoints remain
external and receive checksums and regeneration commands when produced.

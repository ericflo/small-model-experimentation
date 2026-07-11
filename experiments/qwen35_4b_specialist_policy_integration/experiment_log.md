# Qwen3.5-4B Specialist Policy Integration Experiment Log

## Scaffold

Created as a new experiment scaffold attached to agentic breadth installation,
post-training/adaptation, and benchmark generalization.

## 2026-07-11 — preregistration and CPU substrate gate

- Accepted the specialize -> distill -> compose decision record.
- Copied the prior interactive-policy harness into this standalone experiment.
- Added four procedural compound families with exact oracles and explicit
  primitive-removal policies.
- Locked the preregistration and adversarial design review before model output.
- CPU scientific smoke passed all L1-L4 oracle, random, necessity, live-expert,
  split, and replay-exclusion checks. No model baseline or training was run.

## 2026-07-11 — runtime implementation checkpoint

- Discovered that the live GPU is an NVIDIA L40 (46,068 MiB), not the RTX
  6000 Ada recorded by the previous pod; updated the shared environment docs.
- Recreated separate pinned vLLM and Transformers environments from committed
  locks. The first pinned vLLM load resolved the explicit CUDA-graph geometry
  and answered four generic semantic probes correctly. These probes contain no
  gym item and license no capability claim.
- Fixed the merged-composite path so the current runner genuinely loads and
  fingerprints a local checkpoint instead of accepting an unusable harness
  argument.
- Added resumable domain-isolated DAgger, GRPO, extra-SFT, shuffled-reward,
  paired evaluation, diagnostic, and gate stages.
- Pre-baseline amendment: increased extra-SFT from 120 to 300 steps because
  GRPO has multiple forward passes per optimizer step. This preserves the
  preregistered compute-overmatched control; no task model output existed.
- The first one-step QLoRA preflight ran for 110.1 seconds and exited normally,
  but all 128 reconstructed LoRA deltas were zero; explicit merge refused it.
  The adapter remains under the external smoke artifacts and its compact
  failure receipt is committed.
- A two-step, accumulation-one rerun exercised a nonzero optimizer update:
  all 128 mapped deltas were nonzero (summed Frobenius norm 8.742), explicit
  FP32/no-TF32 merge succeeded, and vLLM loaded and generated from the local
  composite. HF/vLLM prompt-token counts matched 4/4.
- A concurrent main-branch environment update introduced the repository's
  canonical PEFT 0.19.1, bitsandbytes 0.49.2, accelerate 1.14.0, and xFormers
  pins. The work was rebased rather than overwritten, a full dependency lock
  was regenerated, and the entire finite-logit/train/merge/local-vLLM preflight
  passed again under that exact lock.
- Primary-paper correction before any specialist output: MOPD equation (5)
  adds `-p_student + p_teacher` to each teacher-top-k reverse-KL summand. The
  earlier “corrected tail mass” wording was inaccurate; no lumped tail bucket
  will be implemented. The registered top-50 choice and all gates are unchanged.
- A second pre-output orchestration audit found that the compound headroom
  call would have inherited `seeds.proxy_eval_base` instead of the separately
  frozen `split.calibration_seed_base`. Before any gym-model generation, the
  call and its analyzer were made explicit and fail-closed on seed namespace;
  unrelated atom generation was also disabled for this compound-only gate.
- The same audit strengthened the frozen shuffled-reward control: advantage
  vectors are now deranged within each family/level cell with no fixed-point
  groups. A plain permutation would retain roughly one correctly routed group
  per cell in expectation and unnecessarily dilute the negative control.
- Specialist qualification now also fails closed on partial or guard-stopped
  DAgger/GRPO runs, stale evaluation-to-merge fingerprints, adapter hash
  mismatches, zero/partial merge mappings, and compute-short controls. Stopped
  checkpoints remain preserved and evaluable for diagnosis but cannot qualify.

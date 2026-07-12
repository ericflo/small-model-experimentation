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
- The preregistered behavioral installation rule is enforced before downstream
  use for every DAgger, extra-SFT, shuffled-reward, and real-reward composite.
  Reusable seven-prefix canaries are hash-bound to the source and candidate
  merge receipts; a candidate must change at least one greedy token sequence
  under identical runner, sampling, graph, prompt, and environment metadata.

## 2026-07-11 — incumbent regeneration and installation gate

- The frozen C53 recipe completed all 333 optimizer steps over 2,117 encoded
  rows in 3,054.4 seconds on the L40 (12.82 GB peak allocated CUDA memory).
- The independent encoder audit found 123/2,240 skips (5.49%): 116 forced-close
  atom rows and seven episode rows. No row was truncated or silently relabeled.
- Explicit CUDA FP32/no-TF32 merge applied 128/128 nonzero deltas, with summed
  Frobenius norm 161.39 and merged-weight SHA-256 `56e2bec45199ebcc...`.
- All 7/7 frozen visible-prefix greedy canaries changed relative to the pinned
  base with identical prompt, runner, engine, sampling, graph, and environment
  metadata. The aggregate incumbent gate passed every registered check.
- These receipts license the disjoint compound-headroom calibration only; they
  do not establish capability improvement.

## 2026-07-11 — disjoint compound-headroom gate

- Ran 288 greedy episodes over `cipherkiln`, `mazeferry`, `patchferry`, and
  `tripleforge`, L2-L4 with 24 episodes/cell, at the separately frozen episode
  seed base 80500. Atom generation was disabled.
- Family means were 0.2271, 0.2961, 0.0115, and 0.0053 respectively; macro
  0.1350 passed the preregistered exclusive `<0.60` ceiling by a wide margin.
- Every protocol check passed. Generation used 1,942,775 sampled tokens and
  2,229,494 logical input tokens across 3,742 turns in 2,225.5 wall seconds.
- The post-run firewall check confirmed that neither hidden specs/expert labels
  nor raw top-20 logprob payloads survived in the committed evaluation rows.
- This is a headroom/measurement result only. It licenses matched baselines and
  specialist production; it does not show a capability gain.
- Before matched baseline output, unused atom passes were removed from the
  best-of-8, DAgger, extra-SFT, and shuffled-reward evaluations. Qualification
  reads atom retention only from the paired incumbent and real specialist;
  episode prompts, seeds, decoding, and every registered comparison are
  unchanged.
- Before best-of-8 output, its qualification scope was narrowed to the seven
  specialist-training families and its token ledger made own-domain paired.
  Primitive/compound transfer best-of-8 is not consumed by specialist gates;
  the held-out compound comparison remains a separate 128/cell confirmatory
  arm if integration is reached. This preserves the stop hierarchy and avoids
  spending eight rollouts on five families before a teacher exists.

## 2026-07-12 — terminal specialist-headroom stop

- Completed the paired greedy baseline on all 12 process families: 864
  episodes, macro 0.4582, plus 1,344 atom items at macro 0.6806. The clean
  committed runner generated 3,893,188 episode tokens in 4,991.3 seconds.
- Core macros were discover 0.5127, control 0.5230, tools 0.9940, and compose
  0.1797. With the frozen `+0.10` pass-one bar and score ceiling 1.0, the tools
  target is 1.0940 and its maximum possible gain is only 0.0060.
- Interrupted the just-started old-scope best-of-8 subprocess during engine
  warmup. It generated zero sampled requests and wrote no result artifact.
- Added `analyze_specialist_headroom.py`; the resulting terminal receipt passes
  every baseline protocol check, marks only `tools` impossible, and authorizes
  `stop_before_best8_and_specialist_production`. A stage resume reproduces the
  stop before GPU load.
- No DAgger, GRPO, specialist, control, teacher-audit, MOPD, confirmatory, or
  benchmark stage ran. The OPSD/MOPD mechanism remains untested here.
- Durable design lesson: calibrate theoretical pass-one headroom independently
  for every mandatory teacher before sampling baselines or producing teachers.
  Any replacement tools core or split is a new preregistered experiment; the
  current threshold and outcome are not amended.

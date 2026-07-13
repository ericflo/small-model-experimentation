# NF4/BF16 Training–Deployment Parity Diagnostic Protocol

**Status:** interpretation-only diagnostic, frozen before its first execution.

This protocol was added after the experiment's preregistration and after some
primary training diagnostics were visible. It is not an amendment to the
frozen design, creates no authorization, and cannot stop, rescue, select, or
retroactively reclassify an experiment arm. The final same-backend procedural
comparison remains authoritative.

## Question and scope

The MOPD trainer evaluates and differentiates a 4-bit NF4 QLoRA surrogate,
whereas the next rollout round serves an explicitly merged bfloat16 composite.
This diagnostic quantifies whether the registered objective and logit movement
in the training surrogate are preserved in that merged checkpoint.

The bfloat16 readout uses Transformers SDPA on the exact checkpoint served by
vLLM. This isolates the quantization-plus-merge seam. It is **not** an
HF/vLLM-kernel parity claim: the standard vLLM interface exposes at most 20
prompt log-probabilities here and therefore cannot reconstruct the registered
teacher-top-50 full-normalizer objective. Every result-bearing rollout arm
continues to use the same vLLM backend.

## Fixed cohort

- Primary seed: `42` only.
- Rounds: all four completed rounds, `0,1,2,3`; no round may be omitted or
  selected by its result.
- Units: the trainer's already-existing `_probe_units` cohort in each round,
  reconstructed from the exact consumed unit ledger and target cache.
- Selection: sort consumed units by sample ID inside target buckets, then take
  the first six deep capability units and first two soup anchors.
- Prefixes: all eight units must retain their full prompt prefix.
- Objective positions: every registered natural target position in each unit.
- Full-logit position: the middle registered target position,
  `positions[len(positions) // 2]`, matching the locality convention.

No new state, continuation, teacher output, verifier score, task score, or
benchmark information is generated or read.

## Model views

For each round, score the same eight units under four views:

1. `nf4_before`: exact round base loaded through the trainer's NF4 double-quant,
   bfloat16-compute, SDPA, k-bit-preparation, zero-initialized PEFT path;
2. `nf4_after`: that exact in-memory surrogate after loading the saved adapter,
   with evaluation mode and LoRA dropout disabled;
3. `bf16_before`: exact round base loaded bfloat16 with Transformers SDPA;
4. `bf16_after`: exact explicitly merged round checkpoint loaded identically.

The saved adapter's complete A/B key, shape, dtype, and value inventory must
match the attached PEFT inventory exactly after load. The merge receipt must
report the same number of applied and nonzero modules as saved A/B pairs and
the exact registered `alpha / rank` scale.

The diagnostic replays the NF4 pre/post unit losses against the training
receipt. Replay tolerances (`atol=1e-5`, `rtol=1e-3`) are engineering checks on
whether the training view was reconstructed; they are not scientific parity
thresholds.

## Fixed measurements

At every target position, evaluate the canonical cached-teacher corrected
top-50 reverse-KL objective. Report per-unit means and equal-unit cohort means,
not only a position-pooled mean. For each unit report:

- NF4 and bfloat16 pre/post objective means;
- bfloat16-minus-NF4 endpoint gaps;
- NF4 and bfloat16 pre-to-post objective gains;
- gain difference, sign agreement, and position-vector hashes.

Across units report mean gains, mean and maximum absolute gain errors, sign
agreement, and Pearson gain correlation.

At the fixed midpoint, remove each full-vocabulary logit's global mean and
report for both pre and post endpoints:

- median, RMS, 95th-percentile, and maximum absolute centered-logit error;
- total-variation and Jensen–Shannon divergence in nats under full softmax;
- top-1 agreement and top-50 overlap;
- mean and maximum teacher-support log-probability error.

Also compare centered pre-to-post logit movements using cosine similarity,
both L2 norms, the bfloat16/NF4 norm ratio, and median/RMS/p95/maximum absolute
movement error. Zero-norm movements are reported explicitly; cosine similarity
and the norm ratio are null rather than assigned a misleading numeric value
when their denominator is degenerate. This separates a static quantization
offset from failure to transfer the learned movement through the merge.

No parity magnitude or direction is a pass/fail criterion.

## Provenance and receipt

The output receipt must bind, by SHA-256 where applicable:

- this protocol, the diagnostic script and helper, target-cache/unit helpers,
  the canonical MOPD loss, trainer, and explicit merge implementation;
- the frozen config and preregistration receipt;
- each target cache and cache receipt;
- each training receipt, adapter config, and adapter weights;
- each before/after merge receipt, config, and actual weight-file inventory;
- the completed seed-42 integration receipt and each round entry it binds;
- the exact probe identities, per-position objective vectors, and midpoint
  full-logit vectors;
- Python, PyTorch, Transformers, PEFT, bitsandbytes, CUDA, GPU, and training
  lock metadata.

The receipt's status is `interpretation_only`; its downstream authorization is
always `null`. Exact provenance, cohort, finiteness, and NF4 replay checks may
invalidate the diagnostic itself. Scientific parity measurements never gate
the frozen experiment. A large discrepancy can qualify final mechanism claims
or motivate a separately preregistered successor only.

Every bound artifact is rehashed after scoring and must still match its
pre-allocation inventory. The output path must not already exist; a diagnostic
receipt is never silently overwritten.

## Execution boundary

Commit this protocol and implementation before observing parity output. Run
the diagnostic only after the live seed-42 integration process has exited and
all four round adapters and explicit merges are complete. It is intentionally
not wired into `scripts/run.py`.

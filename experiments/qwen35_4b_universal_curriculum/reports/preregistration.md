# Pilot Preregistration

Hardware-budget clarification (frozen before any score): see
[`protocol_amendment_001.md`](protocol_amendment_001.md). Quick uses think@1,024 and
medium uses think@2,048 because the next power-of-two cap fails the suite estimator's
public tier budget on the restored RTX 4090.

## Frozen Question

Does a truth-audited, surface-varied synthetic curriculum add broad held-out transfer
beyond the existing broad emission-policy install?

## Arms

1. `base`: pinned unmodified Qwen/Qwen3.5-4B.
2. `replay`: the existing C53 `blend` install, used as the strong frozen control.
3. `designed_only`: the clean designed curriculum trained from base at the fast tier.
4. `designed_plus_replay`: the same designed rows co-trained with the frozen
   `sft_blend.jsonl` corpus at the fast tier.
5. `blend_then_designed_fast`: continue the immutable C53 `blend` adapter for one
   low-learning-rate epoch on the frozen 800-row designed search tier. This is the
   cheapest test of whether the new curriculum can add procedures without rebuilding
   the already-installed emission policy.

The first pilot starts with arms 1, 2, and 5. If the sequential arm fails its local
installability/retention gates, no benchmark seed is consumed. If it passes locally but
fails transfer, arms 3 and 4 remain the preregistered co-training alternatives in this
factorial. Hyperparameters, seeds, corpus hashes, tokenizer dose, and skipped-row count
must be written to receipts before evaluation. The fast tier is exactly 800 designed
rows, one epoch, max length 2,048, `w_think=0.2`, and zero skipped targets; a promoted
configuration is retrained at the full registered dose in a successor experiment.

## Evaluation and Firewall

- Qwen/Qwen3.5-4B only, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Candidate adapters are explicitly merged into the full composite model; runtime vLLM
  LoRA is forbidden by C49.
- Benchmark access is exclusively through `scripts/run_benchmark_aggregate.py`.
- The initial event is Menagerie quick on a fresh seed, canonical tier budget, same
  `qwen_vllm` backend for every arm.
- Raw benchmark material never enters this experiment. Only aggregate, public
  per-family scores, budget status, provenance hashes, and wall time are retained.

## Pilot Gates

- Corpus: deterministic, duplicate-free, executable-truth valid, behaviorally
  non-collapsed where depth is claimed, and zero target contradictions.
- Training: zero skipped rows; finite loss; nonzero adapter delta; exact pinned model.
- Candidate promotion: combined-minus-base aggregate > 0 and every one of the ten public
  family deltas >= 0 on the initial quick event. Otherwise stop benchmark exposure and
  diagnose locally.
- Universal-feature confirmation is not part of this pilot. It requires a successor
  experiment with fresh quick seeds, strict positive mean delta on every family, medium
  transfer, paired uncertainty, and matched-compute sampling.

## Interpretation Rules

- Same-surface improvement proves only installability.
- Combined > base but combined ~= replay attributes the gain to replay, not designed data.
- A subset-only family lift is axis transfer, not universality.
- Any target contradiction, truncation, backend mix, or raw benchmark exposure
  invalidates the run regardless of score.

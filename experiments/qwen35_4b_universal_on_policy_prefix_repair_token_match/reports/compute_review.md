# Second Adversarial Review: Exact-Compute Training Freeze

**Date:** 2026-07-14
**Scope:** deterministic stream materialization, exact Qwen token accounting, and
training authorization only
**Verdict:** `PASS_CONTROL_TRAINING`.

This review authorizes exactly one next event: train the replay control from the
authenticated `close_xi` parent. It does not authorize candidate training until the
control receipt is committed, rebased, pushed to `main`, and both required workflows
are green. It authorizes no local capability evaluation and no benchmark access.

## Authentication and contamination boundary

- The only tokenizer and prospective training model is `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- The common warm start remains the exact `close_xi` adapter: weights/config
  `16e9dc75...c179` / `de953bd5...7ff`.
- The 60-row repair source is the already-published quota-satisfying inventory,
  SHA-256 `30141538...d84b8`. Selection was not changed after token measurement.
- The replay source and inherited partition are authenticated before selection.
  Shared, candidate-filler, and control-variable replay source indices are pairwise
  disjoint. The 200 shared replay rows are byte-identical at the same stream
  positions in both arms.
- Materialization and validation read only experiment-owned data and the exact model
  tokenizer. They make zero model calls and do not read or import `benchmarks/`.

The frozen source-token, stream-manifest, control-stream, candidate-stream, and final
token-receipt hashes are respectively:

```text
2ae6aded50fb4ad649bf69eea01e03aee58b73e58083276e2ab5f188b3ff654d
f836d0a192adfd1e85e4b3514b4854515b239be5faae0c33cebea46530593cd3
541805df2d817707c1e76213e50c8f08fd9caff10d0a3887e1196424b6820be6
9a43f3bea7699af4899678042623a90ef1b6cfc0f17defe069570be908cc03f1
eb08026ffcf82b8780819a26a522f04d69358ffdfd4797dd4c603dd1fbbe0cfc
```

The actual training encoder is `scripts/train_think.py:encode_row`, SHA-256
`0cfb126f...2cc4`. It remains byte-identical to the original design checkpoint. Both
the validator and trainer call that implementation rather than a token-count
approximation; the validator separately reconstructs and checks semantic boundaries
without changing trainer outputs.

## Exact arm geometry

Both arms contain 320 rows, exactly 304,313 unpadded forward tokens, zero rejected
rows, and 200 position-aligned shared rows. Because microbatch size is one, padding
does not create an unrecorded forward-compute difference. At one epoch, batch size
one, and gradient accumulation eight, each arm performs 40 optimizer steps.

| Block | Candidate rows | Control rows | Forward tokens |
|---|---:|---:|---:|
| shared replay | 200 | 200 | 199,360 each |
| on-policy prefix repair | 60 | 0 | 76,953 |
| candidate replay filler | 60 | 0 | 28,000 |
| disjoint control replay | 0 | 120 | 104,953 |
| **total** | **320** | **320** | **304,313 each** |

All 60 selected repairs fit the 4,096-token limit and remain balanced at ten per
registered class. Across the final arms, minimum/maximum encoded lengths are
329/2,991 tokens. The repairs retain all 47,123 exact parent-prefix tokens as masked
context; none receives loss.

## Gradient-exposure audit

Equal forward tokens are not equal target composition. The intervention deliberately
conditions on long parent failures, so the repair arm converts part of what replay
would train as a target into masked context.

| Per-epoch exposure | Replay control | Prefix-repair candidate | Candidate − control |
|---|---:|---:|---:|
| forward tokens | 304,313 | 304,313 | 0 |
| masked context | 119,305 | 152,726 | +33,421 |
| parent-prefix context | 0 | 47,123 | +47,123 |
| think target | 181,580 | 147,631 | −33,949 |
| close target | 640 | 640 | 0 |
| answer target | 2,788 | 3,316 | +528 |
| total target span | 185,008 | 151,587 | −33,421 |
| nonzero-weight tokens | 145,404 | 111,983 | −33,421 |
| absolute weight mass | 31,311.2 | 25,049.4 | −6,261.8 |

This is a disclosed intervention property and residual causal ambiguity, not a token
matching failure. A positive result would show that targeted conditioning beats more
replay despite fewer loss-bearing tokens and lower aggregate loss weight; it could
not isolate prefix state from target-composition effects. A negative result could be
caused by the repair mechanism, reduced supervised exposure, long-context difficulty,
or their interaction. Training loss is diagnostic only and cannot adjudicate the
mechanism.

## Frozen training event

Each arm is an independent continuation from the same authenticated parent, never a
continuation from the other arm. Frozen hyperparameters are one epoch, learning rate
`1e-5`, rank/alpha `32/64`, batch size `1`, gradient accumulation `8`, maximum length
`4096`, think/close weights `0.2/0.2`, and seed `47`.

The training wrapper authenticates the receipt, dataset bytes, parent adapter,
zero-skip geometry, exact output path, and all hyperparameters. It captures clean Git
HEAD and status before opening any output, refuses overwrite, preserves a normalized
log and failure receipt, and requires a complete adapter. This avoids the earlier
self-created-log dirty-tree failure.

## Adversarial findings

- **Long severe prefixes:** 42 of 60 selected repairs end at the generation cap.
  They remain because cap prevalence was observed only after frozen selection. All
  rows fit, but learning from late failure states may be intrinsically difficult.
- **Masked-context dominance:** the candidate has fewer loss-bearing tokens and less
  loss mass than control. Exact forward compute is the registered match; any report
  must preserve the exposure table above.
- **Replay-distribution mismatch:** the variable replay blocks are token-matched but
  not family- or target-span-matched to repairs. This is inseparable from replacing
  replay with on-policy repairs at fixed forward compute and must not be presented as
  a pure prefix-mask ablation.
- **One-epoch sensitivity:** one epoch limits overfitting and preserves the frozen
  40-step comparison, but it may underdose the repair mechanism. No dose adjustment
  is permitted in this result-bearing experiment.
- **Interface validity:** exact generated prefix token IDs are appended after the
  assistant generation prompt and fully loss-masked. The correction, close, and
  answer remain targets. Encoder tests cover invalid IDs, forbidden boundary tokens,
  prefix/text mismatch, and exact normal/prefix span reconstruction.
- **Evidence boundary:** stream validity is not capability evidence. Local promotion
  still requires the sole candidate to beat both parent and replay overall and on
  execute/induct/probe. The aggregate gateway stays sealed unless that happens.

## Authorized sequence

1. Commit, rebase, push, and CI-verify this model-free freeze.
2. Run only `--stage train-control` from that clean published checkpoint.
3. Preserve and publish the control receipt; wait for both workflows.
4. Run only `--stage train-candidate` from the next clean published checkpoint.
5. Preserve and publish the candidate receipt, then perform a separate local-eval
   design/checkpoint before any capability event.

Any hash drift, skipped row, overwrite condition, dirty preflight, incomplete
adapter, or failed training stops the sequence and is preserved as a failure rather
than repaired in place.

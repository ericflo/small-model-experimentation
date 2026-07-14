# Second Adversarial Review: Exact-Exposure Training Freeze

**Date:** 2026-07-14
**Scope:** replay lineage, deterministic stream construction, exact Qwen token
accounting, and control-training authorization only
**Verdict:** `PASS_CONTROL_TRAINING`.

This review authorizes exactly one next model event: independently continue the
authenticated replay adapter on the frozen replay-control stream. It does not
authorize candidate training until the control log and receipt are committed,
rebased onto `origin/main`, pushed to `main`, and both required workflows are green.
It authorizes no merge, local capability evaluation, or benchmark access.

## Authentication and contamination boundary

- The only tokenizer and prospective training model is `Qwen/Qwen3.5-4B` at
  revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Both arms independently warm-start from the published predecessor replay adapter.
  Its weights/config hashes are `bb59d3bd...5154d` / `0dfd9bda...120f`.
- The copied 2,240-row replay source is byte-identical to its predecessor source,
  SHA-256 `25a9595f...f0c2`. The copied predecessor stream manifest is
  `abf8b505...0966f`; it fixes the inherited 200-row shared replay core.
- The 52-row selected restart source remains `022b1ea4...d951f`. It contains four
  examples per each of 13 skills, including 40 hard correctness/cap failures and 12
  correct but over-budget cases. Every example begins at the original prompt and no
  example contains a parent prefix.
- Stream construction, tokenization, and validation read experiment-owned data and
  the exact Qwen tokenizer. They make no model call and do not read or import
  `benchmarks/`.

The frozen source-token, stream-manifest, replay-control, restart-candidate, and
independent token-receipt hashes are:

```text
ac9b9c8a3c9bfc66699781c96792ea72c37701b11719772764e74b35dba10bd6
7ba55045e72371e3675ba67bcf0bd72f6a0bf645c3ad7d0e92f7282e59d91de1
7a8d45666000cbb6bffabf6faab8f9d61006bf3a80275a631238a23cd03b5078
28deb20e6bfca81f760549b071d0d0df39bfa561c4d09fde0580d81699413190
52a761ef8fd37f3eac88abf8f090013f571a47511daeb26820ca030201b1c170
```

The actual training encoder is `scripts/train_think.py:encode_row`, SHA-256
`0cfb126f...2cc4`. It is byte-identical to the published predecessor trainer. Source
measurement and final-stream validation both call that implementation, then
independently reconstruct prompt, thinking, close, and answer boundaries.

## Exact arm geometry

The integer feasibility problem chooses disjoint 68-row candidate filler and 120-row
control blocks from the copied replay pool while preserving the inherited core. It
was solved to an exact integral optimum by SciPy 1.18 HiGHS in 4.43 seconds, 801
nodes, and zero reported gap. No objective-based quality ranking or model output is
used. The selected indices and solver receipt are frozen in `stream_manifest.json`.

Both arms contain 320 rows, exactly 297,731 unpadded forward tokens, 126,796
loss-bearing target tokens, absolute loss mass 27,632.8, zero rejected rows, and 200
position-aligned byte-identical replay rows. Because microbatch size is one, padding
does not create hidden forward-compute differences. At one epoch and gradient
accumulation eight, each arm performs 40 optimizer steps.

| Block | Candidate rows | Control rows |
|---|---:|---:|
| shared replay | 200 | 200 |
| clean counterfactual restarts | 52 | 0 |
| candidate replay filler | 68 | 0 |
| disjoint control replay | 0 | 120 |
| **total** | **320** | **320** |

All final rows fit the 4,096-token limit. Candidate sequence lengths span 128–2,991
tokens; control lengths span 308–2,991. Targets were not changed, masked, truncated,
padded, or duplicated to create the match. The replay subsets are disjoint from each
other and from the shared core.

## Gradient-exposure audit

The three preregistered axes are equal exactly in the final encoded streams.

| Per-epoch exposure | Replay control | Restart candidate | Candidate − control |
|---|---:|---:|---:|
| forward tokens | 297,731 | 297,731 | 0 |
| nonzero target tokens | 126,796 | 126,796 | 0 |
| absolute loss mass | 27,632.8 | 27,632.8 | 0.0 |
| answer target | 2,842 | 2,842 | 0 |
| close target | 640 | 640 | 0 |
| masked prompt/context | 127,084 | 110,670 | −16,414 |
| think target | 167,165 | 183,579 | +16,414 |
| total target span | 170,647 | 187,061 | +16,414 |
| parent-prefix context | 0 | 0 | 0 |

The target-span difference is real despite equality of loss-bearing tokens and loss
mass. Some replay rows are forced-close examples whose thinking span is context with
zero loss; the solver selected different forced-close composition in the variable
blocks. Thus 16,414 candidate tokens move from zero-weight prompt/context to
zero-weight thinking span. Answer and close counts remain exactly equal, and the
actual loss-bearing count and absolute weight mass remain exactly equal. This is a
residual sequence-composition difference, not extra supervised exposure.

## Causal unit and remaining ambiguity

The package intervention replaces replay examples with balanced, failure-selected,
clean oracle recomputations. It tests task-level parent failure selection plus restart
supervision as a package; it does not isolate selection from oracle content or skill
balance. The 12 budget-only rows also mean the package targets both correctness and
bounded computation. A positive result supports that combined curriculum, while a
negative result rejects this dose and selection rule rather than all clean restarts.

The replay variable blocks are exactly exposure-matched but differ in family and
forced-close composition. Matching those properties too was not preregistered and
would condition the control on semantic properties observed after selection. The
three exact axes, shared core, same parent, same update count, and fixed candidate
contents are the primary causal controls.

## Frozen training event

Each arm is an independent continuation from the same authenticated adapter, never a
continuation from the other arm. Hyperparameters are one epoch, learning rate
`1e-5`, rank/alpha `32/64`, batch size `1`, gradient accumulation `8`, maximum length
`4096`, think/close weights `0.2/0.2`, and seed `48`.

The training wrapper authenticates dataset bytes, exact token receipt, parent
adapter, output path, all hyperparameters, and a clean pushed `main` before opening
outputs. It refuses overwrite, records package versions and Git provenance, streams
a durable log, preserves failures, requires 320 encoded rows with zero skips, and
requires a complete adapter. Candidate launch additionally authenticates the
committed control receipt, log, and external adapter.

Training loss is diagnostic only. Stream validity is not capability evidence, and
the aggregate gateway remains sealed until a separately frozen fresh-local design
passes its absolute and relative promotion rules.

## Authorized sequence

1. Commit this model-free freeze, fetch and rebase onto `origin/main`, rerun smoke
   and `make check`, push directly to `main`, and verify both workflows green.
2. Run only `--stage train-control` from that clean published checkpoint.
3. Preserve the control log and receipt, freeze their hashes, and repeat the full
   publish/CI gate.
4. Run only `--stage train-candidate` from the next clean published checkpoint.
5. Preserve and publish the candidate result before designing any merge or local
   capability event.

Any hash drift, skipped row, overwrite condition, dirty or unpushed preflight,
incomplete adapter, or failed training stops the sequence and is preserved rather
than repaired in place.

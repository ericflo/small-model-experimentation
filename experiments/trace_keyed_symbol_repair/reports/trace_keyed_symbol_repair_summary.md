# Trace-Keyed Symbol Repair Summary

Generated: `2026-06-20 10:09:05 UTC`.

## Core Results

| Split | Condition | Repair@1 | Visible pass | Patch apply | Expected-token copy | Wrong-token removed | Successes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IID | Frozen base + trace | 0.0% | 0.0% | 0.0% | 18.3% | 0.0% | 0/60 |
| Format holdout | Frozen base + trace | 0.0% | 0.0% | 0.0% | 3.3% | 0.0% | 0/60 |
| IID | Final-patch SFT + final patch | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0/60 |
| Format holdout | Final-patch SFT + final patch | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0/60 |
| IID | No-trace SFT + no trace | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/60 |
| Format holdout | No-trace SFT + no trace | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/60 |
| IID | Shuffled-trace SFT + trace | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/60 |
| Format holdout | Shuffled-trace SFT + trace | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/60 |
| IID | Trace SFT + trace | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 60/60 |
| Format holdout | Trace SFT + trace | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 60/60 |

## Ablations

| Split | Condition | Repair@1 | Visible pass | Patch apply | Expected-token copy | Wrong-token removed | Successes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IID | Trace SFT + no trace | 0.0% | 0.0% | 100.0% | 0.0% | 95.0% | 0/60 |
| Format holdout | Trace SFT + no trace | 0.0% | 0.0% | 100.0% | 0.0% | 90.0% | 0/60 |
| IID | Trace SFT + shuffled trace | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/60 |
| Format holdout | Trace SFT + shuffled trace | 0.0% | 0.0% | 100.0% | 0.0% | 95.0% | 0/60 |

## Artifact Split

- Downloadable directory: `/workspace/experiments/trace_keyed_symbol_repair`.
- Large adapters/checkpoints: `/workspace/large_artifacts/trace_keyed_symbol_repair`.

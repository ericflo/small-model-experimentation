# Counterexample Rule Repair Summary

Generated: `2026-06-20 20:16:10 UTC`.

## Core Results

| Split | Condition | Repair@1 | Visible pass | Hidden pass | Patch apply | Marker match | Input literal | Successes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IID | Frozen base + trace | 0.0% | 0.0% | 0.0% | 4.4% | 2.2% | 48.9% | 0/45 |
| IID | Final-patch SFT + final patch | 0.0% | 0.0% | 0.0% | 100.0% | 17.8% | 64.4% | 0/45 |
| IID | No-trace SFT + no trace | 8.9% | 8.9% | 8.9% | 100.0% | 22.2% | 53.3% | 4/45 |
| IID | Shuffled-trace SFT + trace | 4.4% | 4.4% | 4.4% | 100.0% | 17.8% | 53.3% | 2/45 |
| IID | Trace SFT + trace | 91.1% | 91.1% | 91.1% | 100.0% | 95.6% | 55.6% | 41/45 |
| Format holdout | Frozen base + trace | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 48.9% | 0/45 |
| Format holdout | Final-patch SFT + final patch | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 42.2% | 0/45 |
| Format holdout | No-trace SFT + no trace | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 62.2% | 0/45 |
| Format holdout | Shuffled-trace SFT + trace | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 60.0% | 0/45 |
| Format holdout | Trace SFT + trace | 53.3% | 53.3% | 53.3% | 100.0% | 68.9% | 66.7% | 24/45 |
| Rule-family holdout | Frozen base + trace | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0/45 |
| Rule-family holdout | Final-patch SFT + final patch | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/45 |
| Rule-family holdout | No-trace SFT + no trace | 2.2% | 2.2% | 2.2% | 100.0% | 2.2% | 100.0% | 1/45 |
| Rule-family holdout | Shuffled-trace SFT + trace | 0.0% | 0.0% | 0.0% | 100.0% | 2.2% | 100.0% | 0/45 |
| Rule-family holdout | Trace SFT + trace | 0.0% | 0.0% | 0.0% | 100.0% | 6.7% | 100.0% | 0/45 |

## Trace Adapter Ablations

| Split | Condition | Repair@1 | Visible pass | Hidden pass | Patch apply | Marker match | Input literal | Successes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IID | Trace SFT + no trace | 0.0% | 0.0% | 0.0% | 100.0% | 11.1% | 53.3% | 0/45 |
| IID | Trace SFT + shuffled trace | 0.0% | 0.0% | 0.0% | 100.0% | 8.9% | 53.3% | 0/45 |
| Format holdout | Trace SFT + no trace | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 57.8% | 0/45 |
| Format holdout | Trace SFT + shuffled trace | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 55.6% | 0/45 |
| Rule-family holdout | Trace SFT + no trace | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0/45 |
| Rule-family holdout | Trace SFT + shuffled trace | 0.0% | 0.0% | 0.0% | 100.0% | 2.2% | 100.0% | 0/45 |

## Trace Adapter Family Breakdown

| Condition | Family | Repair@1 | Visible pass | Hidden pass | Successes |
| --- | --- | --- | --- | --- | --- |
| Trace SFT + trace / IID | affine_int | 73.3% | 73.3% | 73.3% | 11/15 |
| Trace SFT + trace / IID | slug_affix | 100.0% | 100.0% | 100.0% | 15/15 |
| Trace SFT + trace / IID | threshold_label | 100.0% | 100.0% | 100.0% | 15/15 |
| Trace SFT + trace / Format holdout | affine_int | 0.0% | 0.0% | 0.0% | 0/15 |
| Trace SFT + trace / Format holdout | slug_affix | 60.0% | 60.0% | 60.0% | 9/15 |
| Trace SFT + trace / Format holdout | threshold_label | 100.0% | 100.0% | 100.0% | 15/15 |
| Trace SFT + trace / Rule-family holdout | parity_offset_holdout | 0.0% | 0.0% | 0.0% | 0/45 |

## Artifact Split

- Downloadable directory: `/workspace/experiments/counterexample_rule_repair`.
- Large adapters/checkpoints: `/workspace/large_artifacts/counterexample_rule_repair`.

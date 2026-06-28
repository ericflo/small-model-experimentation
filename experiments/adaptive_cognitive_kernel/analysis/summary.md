# Adaptive Cognitive Kernel Analysis Summary

- Main run: `main_ack_v1`.
- Seeds: `101,202,303`.
- Arms: dynamic ACK, ACK no-delta, fixed GRU, direct transformer.
- Training uses online-generated programs at lengths `2..6`.
- Evaluation includes held-out lengths `8,10,12` and held-out adjacent operation compositions.

## Main Readout

Dynamic ACK learned an ordered conditioning signal but did not outperform a conventional fixed recurrent controller.

| Metric | ACK ordered | ACK shuffled | ACK no-delta | Fixed GRU | Direct transformer |
|---|---:|---:|---:|---:|---:|
| Length-12 final answer | 7.7% | 5.7% | 6.5% | 8.3% | 5.8% |
| Length-12 exact pair | 0.7% | 0.5% | 0.5% | 0.7% | 0.3% |
| Length-12 state step | 11.6% | 2.0% | 4.0% | 14.4% | 10.5% |
| Composition-12 final answer | 7.1% | 6.2% | 5.7% | 7.0% | 6.6% |

## Interpretation

The dynamic weight-edit path carries real signal: disabling deltas and shuffling the conditioning stream both damage the ACK runtime, especially on intermediate-state prediction. The stronger claim fails: the temporary weight-edit runtime does not beat a standard fixed recurrent controller, and all models degrade sharply beyond the training length range.

## Artifacts

- Report: `reports/adaptive_cognitive_kernel_report.md`
- HTML report: `reports/adaptive_cognitive_kernel_report.html`
- Metrics: `analysis/metrics.csv`
- Summary CSV: `analysis/summary_by_arm.csv`
- Figures: `analysis/figures/`
- Checkpoints: `large_artifacts/adaptive_cognitive_kernel/checkpoints/main_ack_v1/`

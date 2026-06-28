# Experiment Log

## 2026-06-28

- Created standalone live tool-state controller package.
- Copied source cases and sandbox utilities into the package.
- Configured a family-disjoint balanced pilot split with train/dev/test families and fresh trace generation.
- Ran a 9-case smoke pass with one repair round to exercise trace generation, safe execution, report writing, and chart writing.
- Ran an initial full pass, but its held-out split had no oracle headroom on test, so it was retained only as a diagnostic artifact and not used for the final readout.
- Tightened split construction to seed recovery-positive families across train/dev/test before filling neutral families.
- Ran the corrected `split2` pilot with 60 fresh traces: 36 train, 12 dev, 12 test.
- Final held-out result on `split2`: direct-only 0/12, learned rule 2/12, sequential LoRA 2/12, shuffled-label LoRA 0/12, oracle 2/12.
- The sequential LoRA matched the oracle on the held-out split while using 6 program generations, versus 12 for the best fixed/rule policies and 2 for the oracle.

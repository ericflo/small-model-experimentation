# Qwen3.5-4B Answer-Potential Trace SFT Report

## Status

Pre-run design freeze. No scientific result has been observed. See `preregistration.md` for the
decision rules and `design_review.md` for the integrated adversarial fixes.

## Planned Evidence

The report will headline the G0 scorer gate before any SFT result, then compare matched empty,
length-matched random, binary-success RFT, answer-potential, and shuffled-potential arms. Deployable
accuracy, parse rate, thought tokens, family macros, matched-forward-token sample-more, and oracle
ceilings will remain separate.

## Result Boundary

No placeholder result is implied by this design-stage report. Terminal negative and stopped outcomes
will be preserved with the same prominence as a positive result.

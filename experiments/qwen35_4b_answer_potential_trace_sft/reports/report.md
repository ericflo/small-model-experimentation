# Qwen3.5-4B Answer-Potential Trace SFT Report

## Status

Pre-calibration implementation complete. No scientific result has been observed. The four-trace GPU
smoke and the HF/vLLM plumbing parity diagnostic passed; see `preregistration.md` for the dated
pre-result readout amendment, decision rules, and `design_review.md` for adversarial fixes.

## Smoke Evidence

- Frozen procedural split construction produced 64 calibration, 600 train, 400 IID, 100 held-family,
  and 100 hard items with zero ID, prompt, digest, or generator-seed overlap.
- Twenty-five CPU tests pass across the firewall, verifier-equivalent answers, controls, statistics,
  and vLLM geometry.
- The corrected real-model smoke produced four finite trace scores and eight fresh answer
  continuations with exact registered CUDA-graph geometry and a passing live-KV capacity receipt.
- HF bf16 SDPA versus vLLM bf16 targeted likelihood differed by at most 0.060 nats per answer token,
  below the pre-calibration plumbing tolerance of 0.15. HF rows are diagnostic only.
- The 64-token smoke traces all contacted the thought cap; their zero rollout score is intentionally
  non-scientific and does not enter G0.

## Planned Evidence

The report will headline the G0 scorer gate before any SFT result, then compare matched empty,
length-matched random, binary-success RFT, answer-potential, and shuffled-potential arms. Deployable
accuracy, parse rate, thought tokens, family macros, matched-forward-token sample-more, and oracle
ceilings will remain separate.

## Result Boundary

No placeholder result is implied by this design-stage report. Terminal negative and stopped outcomes
will be preserved with the same prominence as a positive result.

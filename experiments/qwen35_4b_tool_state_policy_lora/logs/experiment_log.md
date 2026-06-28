# Experiment Log

## 2026-06-28

- Created a standalone tool-state action-policy package.
- Copied table-transformation cases and precomputed Qwen3.5-4B tool-environment traces into `data/`.
- Chose a family-disjoint split seed (`6137`) that keeps direct-miss program-recovery labels present in train, dev, and test.
- Planned a two-stage run: first validate non-neural baselines and oracle ceilings, then run the LoRA action-policy arm if the package compiles and the split is sane.
- Ran the fast baseline/rule stage. The selected rule was `PROGRAM` only when the final program passed the visible example and disagreed with the direct output.
- The selected rule reached 33/50 on held-out test, exactly matching the oracle action ceiling and recovering 5 direct misses with 0 losses.
- Ran a one-step LoRA smoke test. Model loading, adapter attachment, training, scoring, and artifact writing all worked.
- Started an 80-step LoRA run with autoregressive action generation, then interrupted it during post-train evaluation because generation was too slow for a two-action classifier.
- Patched evaluation to score `DIRECT` vs `PROGRAM` by next-token logits. Re-ran the full 80-step real-label LoRA plus 80-step shuffled-label control.
- Final LoRA result: real-label LoRA reached 32/50, recovering 4 direct misses with 0 losses. Shuffled-label LoRA reached 23/50, with 10 direct-correct losses. The adapter learned useful state signal but did not match the simpler rule/oracle ceiling.

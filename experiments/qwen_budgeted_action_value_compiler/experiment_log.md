# Experiment Log

## 2026-06-23

- Created standalone experiment directory with `src/`, `runs/`, `analysis/`, `reports/`, and a separate large-artifact checkpoint root.
- Design choice: keep the frozen-Qwen compiler and typed bytecode VM, but train an action-value model from bounded suffix search returns instead of a binary prefix verifier.
- Planned conditions:
  - constrained greedy decoding from the compiler;
  - local answer-verified repair over complete programs;
  - compiler-logprob typed beam search;
  - exact-prefix value guided beam search;
  - binary found-value guided beam search;
  - graded budgeted-Q guided beam search.
- Implemented `src/qwen_budgeted_action_value_compiler_experiment.py` as a standalone harness with exact, found, and qvalue target modes trained from the same prefix-action sample pool.
- Ran `smoke_budgeted_action_value`, an end-to-end Qwen-backed smoke test with tiny data. It loaded `Qwen/Qwen3-4B`, extracted frozen features, trained a tiny compiler, collected budgeted action-value samples, trained exact/found/qvalue models, wrote metrics, and saved the checkpoint under the large-artifact root.
- Smoke target stats: train exact positives were 12.5%; found positives were 54.4%; q-positive rate was 54.4%; mean Q target was 0.248; mean nonzero Q target was 0.456; mean correct rank was 3.6. This confirms the graded value target is materially sharper than the binary found target.
- Ran `pilot_budgeted_action_value_s128`. The weak 128-example compiler reached 12.5% quick bytecode. Train exact positives were 4.3%; found positives were 68.7%; mean Q was 0.210; mean nonzero Q was 0.306; mean correct rank was 8.7. Fresh paired greedy/logprob were 18.8%; binary found beam reached 25.0%; qvalue beam reached 20.3%; local answer repair reached 50.0%. Hard-composition greedy/logprob were 18.8%; exact beam reached 20.3%; qvalue beam matched 18.8%; local answer repair reached 62.5%.
- Interpretation: binary found-value can improve a weak compiler, but absolute Q values are too low and broad for direct decoding. Patched a sibling-normalized `advantage` target: each action's Q return divided by the best Q among actions from the same prefix, so the model learns the local action choice rather than only absolute recoverability.
- Ran `pilot_budgeted_action_value_s128_advantage` with the same seed and scale. Train mean advantage target was 0.384. Fresh paired binary found beam again reached 25.0%; advantage reached 21.9%; qvalue reached 20.3%; greedy/logprob were 18.8%; local answer repair was 50.0%. Hard-composition advantage did not beat logprob. Interpretation: advantage is useful to measure but not the main bet; binary found and exact-prefix remain the strongest weak-compiler controls.
- Ran `main_budgeted_action_value_s512`, a stronger 512-example main run. The compiler reached 71.9% quick bytecode accuracy. Train prefix labels were 4.3% exact positive, 72.1% found positive, mean Q 0.259, mean advantage 0.415, and mean correct rank 7.7.
- Main value-model best held-out AUCs: exact 0.950, found 0.814, qvalue 0.628, advantage 0.666. The exact-prefix label is cleanest; the graded budgeted targets remain difficult for the lightweight scorer.
- Main decoder results: fresh paired greedy/logprob were 67.2%; best learned budgeted-value beam was advantage at 69.5%; exact and found beams reached 68.8%; answer-verified repair reached 82.0%. Hard composition greedy/logprob were 56.2%; best exact and advantage beams reached 57.0%; answer-verified repair reached 78.9%.
- Interpretation: bounded suffix search exposes a large recoverable action set and learned value guidance produces small no-answer gains, but the main bottleneck is ranking complete candidates. The next experiment should train a deployable complete-program reranker from answer-verified candidate sets rather than another lightweight prefix classifier.
- Generated standalone analysis artifacts:
  - `analysis/summary.md`
  - `analysis/final_metrics.csv`
  - `analysis/best_family_metrics.csv`
  - `analysis/value_train_logs.csv`
  - `analysis/prefix_sample_stats.csv`
  - `analysis/figures/*.png`
  - `reports/qwen_budgeted_action_value_compiler_paper.md`
  - `reports/qwen_budgeted_action_value_compiler_paper.html`

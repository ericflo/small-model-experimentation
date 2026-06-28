# Experiment Log

## Design

This standalone experiment tests silent/internal execution on a Python-shaped
mini-language. Programs are executable by a deterministic harness, but prompts
present them as readable Python-like code. The model is evaluated in three
roles:

- frozen direct final-answer scoring;
- explicit chain-of-thought generation with intermediate state reporting;
- trained private latent compute with direct final/state heads.

The decisive comparisons are held-out program length, held-out operation
composition, K-private-compute scaling, and a shuffled-compute control.

Large trainable artifacts are stored under
`/workspace/large_artifacts/qwen_python_shaped_silent_executor`.

## Iteration Notes

1. `smoke_python_shaped_silent_executor_v1` tested the first 128-value version.
   It completed end to end, but raw prompting made the CoT baseline ramble and
   truncate.

2. `smoke_python_shaped_silent_executor_v2` switched model-facing prompts to
   Qwen chat formatting. CoT outputs became concise but remained weak.

3. `smoke_python_shaped_silent_executor_v3` enabled Qwen thinking mode for the
   CoT baseline. This made CoT a stronger and more expensive baseline.

4. `pilot_python_shaped_silent_executor_v1` and `v2` showed that the 128-value
   target space was too diffuse: QLoRA learned frequent-answer priors, not
   execution, and ordered latent positions did not beat shuffled positions.

5. The substrate was revised to small integers `0..31` with smaller operation
   constants. `smoke_python_shaped_silent_executor_v4` validated the corrected
   generator and faster frozen scoring.

6. `pilot_python_shaped_silent_executor_v3` tested the corrected substrate with
   the default latent recipe. It learned a weak signal but still showed no
   ordered-vs-shuffled separation.

7. `pilot_python_shaped_silent_executor_v4` removed K=0 from training and
   increased state supervision. This was more aligned with the execution
   hypothesis but did not improve the gate.

8. `main_python_shaped_silent_executor_v1` ran the final multi-seed experiment:
   three QLoRA seeds, train lengths `4,8,16`, held-out lengths `24,32`,
   held-out operation compositions, thinking-CoT baseline, ordered latent
   K-sweep, and shuffled-compute control.

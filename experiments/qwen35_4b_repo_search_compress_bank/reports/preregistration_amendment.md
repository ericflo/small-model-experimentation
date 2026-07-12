# Preregistration implementation amendment 1

Frozen after GPU smoke and before the registered harvest seed was used.

The original preregistration requires equal INSPECT/PATCH/VERIFY/COMMIT loss mass. Deterministic oracle smoke initially operationalized this as equal weighted row counts. The first real-trajectory smoke—explicitly non-scientific and on separate seeds—showed why that is insufficient: an exact patch JSON target can be tens of times longer than `test` or `submit`.

The implementation is therefore tightened, without changing any task, model, seed, arm, dose multiplier, success threshold, or outcome gate:

- tokenize every compact row with the pinned Qwen3.5-4B tokenizer and the exact training chat template;
- choose shared per-operator row weights so supervised **action-token** loss mass is equal across all four operators;
- choose compact-only per-operator plan-span weights so supervised **plan-token** loss mass is also equal;
- copy the calibrated contexts, actions, and row weights byte-for-byte into `action_only`, then set only its plan-span loss to zero;
- stop before training if any row crosses the frozen 4,096-token limit or either exact token-mass equality check fails.

The smoke also revealed that a private edge-case test can pass while a visible regression fails. Repository success and submission success are now explicitly the conjunction of final visible and private tests. This corrects scoring to the preregistered phrase “fresh replay passes both visible and hidden tests”; it does not use an experimental outcome.

This amendment makes the frozen intended design executable rather than changing its hypothesis. The pre-amendment smoke remains implementation evidence only and consumed no registered task, training, evaluation, or Menagerie seed.

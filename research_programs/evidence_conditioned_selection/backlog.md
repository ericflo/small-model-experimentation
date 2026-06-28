# Backlog

## Next Experiments

- Train visible-only selectors on candidate pools with explicit false-pass labels held out by family.
- Compare public-test augmentation, generated counterexamples, consensus, and code/verifier reranking on the same pool.
- Build an abstaining selector benchmark that reports precision, recall, and coverage separately.
- Stress selectors under intentionally adversarial visible examples.
- Convert oracle ceiling reports into deployable-gap scorecards.

## Required Controls

- First-visible or shortest-visible baseline.
- Hidden oracle ceiling clearly labeled as non-deployable.
- Random or shuffled candidate ordering.
- Family-held-out evaluation.

## Stop Conditions

Do not continue selector variants that improve selected accuracy only by silently reducing commit rate. Precision, recall, and abstention must be visible.

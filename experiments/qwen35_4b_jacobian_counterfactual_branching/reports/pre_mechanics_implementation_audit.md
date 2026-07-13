# Pre-Mechanics Implementation Audit

Completed after mechanics implementation and before model invocation. It
authorizes one label-free four-task alpha mechanics run only after a pushed hash
boundary. Continuations and correctness remain unavailable.

## Verdict

Mechanics is implementation-complete with eight passing tests. Commit/push,
anchor exact runner/model/geometry/test hashes and the passing smoke-005 hash,
then run once.

## Firewall assertions

1. Mechanics reads only `mechanics_public.jsonl`.
2. Every public row has exactly `task_id` and `visible` fields.
3. Public SHA-256 is frozen in the fresh-data manifest.
4. `first_op`, target pipeline, hidden examples, correct alias, and correctness
   do not exist in the file or runner objects.
5. Mechanics boundary must be anchored to an ancestor commit before model load.
6. Boundary validates exact runner, model-ops, branch-geometry, and test hashes.
7. Boundary also locks the passing smoke-005 bytes and verifies `passed=true`.
8. Pending/stale boundary fails before model construction; a subprocess test
   proves this path.
9. Only the exact permitted model/revision and frozen lens load.
10. Confirmation and qualification files are never opened.

## Prefix and intervention assertions

11. Exactly four public mechanics tasks run.
12. One trace per task uses frozen task-index seed.
13. Every trace must remain live for exactly 512 thought tokens.
14. Any natural close, EOS, cache-contract miss, or wrong token count is fatal.
15. The identical saved prefix feeds baseline, J, and non-J for every alpha.
16. Alpha order is fixed `[0.5,1.0,2.0]` and all members run before selection.
17. Each alpha uses all 12 public alias branch targets exactly once.
18. Branch target identity is its public row index, never a task answer.
19. J and non-J use identical layers, token position, alias order, close, slot,
    readout, and full-prefill backend.
20. Non-J starts from exact-Gram controls and receives only geometry repair.
21. Target norms are paired to realized J row/layer deltas.
22. Every task/alpha independently recomputes the full live receipt.

## Metric and gate assertions

23. J target selection is chosen alias equals supplied branch alias.
24. Non-J target selection uses the same arbitrary row-to-alias mapping as a
    specificity canary.
25. Target probability lift subtracts the unpatched prefix probability for that
    same alias.
26. Metrics aggregate 4 x 12 = 48 supplied targets per alpha.
27. Smallest alpha is selected only if all four frozen mechanics gates pass.
28. Required gates remain J selection >=0.60, mean lift >=0.15, J-minus-non-J
    selection >=0.35, and 100% numeric validity.
29. Failure at every alpha is terminal `NO_NATIVE_J_BRANCH_CONTROL` and cannot
    open continuations.
30. Alpha selection sees no task outcome or gold.
31. Full baseline/J/non-J probabilities and choices are stored for audit, all
    explicitly marked `correct_alias_loaded=false` and `outcome_loaded=false`.
32. Complete trace token IDs and hashes are stored for deterministic replay.
33. Summary records peak memory, public-data hash, smoke hash, and boundary.
34. No secondary alpha, subset, target family, or layer can rescue a miss.
35. Runner still raises for qualification/confirmation; cache-fork continuation
    is not implemented.

## Tests

Eight tests pass: branch rank/zero-sum/Gram/span across all layers and alphas,
determinism, prompt label mutation, fresh/public data schema, one-shot hook
receipt, wrong batch failure, exact quantized-control case, and pending mechanics
firewall.

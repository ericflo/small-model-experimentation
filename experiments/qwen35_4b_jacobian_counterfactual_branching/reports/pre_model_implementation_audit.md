# Pre-Model Implementation Audit

Completed after cache-free native-prefix branch readout implementation and
before any model load. This boundary authorizes **model smoke only**. Mechanics,
continuations, qualification, and confirmation remain unavailable.

## Verdict

Commit and push the implementation, then hash-anchor it in config. One
outcome-blind model smoke may test live bf16 geometry and one-shot hook mechanics.
Failure permits only geometry/cache implementation repair; it cannot inspect a
correct alias or continuation outcome.

## Assertions

1. The only accepted model ID is `Qwen/Qwen3.5-4B`.
2. The exact revision is frozen in config.
3. Model smoke checks an ancestor implementation commit and four exact file
   hashes before model construction.
4. Pending or incomplete boundary fails before model load.
5. Lens SHA-256 is rechecked before model load.
6. Public aliases must be the first 12 frozen lens concepts in exact order.
7. Model smoke loads only the first mechanics row.
8. The prompt is constructed from visible examples and the public alias menu;
   mechanics `first_op`, target pipeline, hidden examples, and correctness are
   discarded before rendering.
9. Model smoke generates exactly 32 live-thought tokens only for memory/cache
   feasibility, not scientific mechanics.
10. Natural close or EOS fails the smoke rather than being silently removed.
11. The unpatched baseline and both branch arms use the identical prompt,
    thought token IDs, close, slot, alias set, and cache-free backend.
12. J and non-J branch width must equal 12.
13. Every branch matrix is computed from the committed lens/config, never an
    outcome.
14. The hook patches the exact final live-thought token position.
15. It registers only on layers 4--8.
16. Each layer must apply exactly once; repeated application raises.
17. Wrong batch width or out-of-range position raises.
18. Only the selected token changes at each hook output.
19. Requested and realized deltas plus input activations are retained in memory
    for geometry audit but branch logits/probabilities are not written by smoke.
20. Model smoke records only finite state, dimensions, hook counts, resource
    use, and geometry maxima—not choices, target selection, or correctness.
21. Live J norm error is measured after bf16 addition.
22. Non-J norm error is paired row-by-row to realized J, not float requests.
23. Non-J projection uses the complete 24-coordinate span at each layer.
24. Realized full 12 x 12 Gram matrices are compared.
25. Realized zero-sum residue is audited for both arms.
26. All numeric gates are conjunctive at preregistered tolerances.
27. A numeric miss makes `passed=false`; it cannot be averaged across layers.
28. Mechanics remains unimplemented/unavailable even after a passing model
    smoke until a second audit and pushed boundary.
29. Six unit tests pass, including exact branch formula/Gram/determinism,
    prompt-label mutation, fresh-data manifest, one-shot token patching, and
    wrong-batch failure.
30. CPU smoke already proves 76 fresh tasks, rank-24 lens dictionaries, rank-11
    zero-sum branch banks, Gram error <=1.14e-6, and float non-J leakage
    <=3.05e-7; model smoke is specifically the live-bf16 adversary.

## Expected failure mode

Naively adding float Gram-matched deltas to bf16 activations will likely violate
the 1e-5 realized norm/Gram or zero-sum tolerance even if span leakage passes.
That is not a reason to weaken controls. Preserve the first smoke receipt and
repair only with outcome-blind joint quantization geometry, following the
independent transport replication's lattice lesson.

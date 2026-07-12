# Pre-Selection Adversarial Implementation Audit

Completed after the outcome-blind model smoke and before any scientific outcome.
No task correctness, chosen alias, trace text, or arm comparison was available
to this audit. It checks whether the implemented seam runner actually enforces
the frozen 58-point design review.

## Verdict

Proceed with the one registered seam-selection run after this audit is committed
and pushed. The audit found four implementation omissions—coherent-content
control, mask-strength diagnostics, machine-config anchoring, and explicit
completeness/control-finiteness receipts—and repaired all four before outcomes.
No threshold, task, model, cap, or success rule changed after model behavior was
observed.

## Assertions

1. **Immutable design:** runtime verifies the ancestor design commit plus exact
   README, preregistration, adversarial-review, and semantic-config hashes.
2. **Immutable data:** runtime verifies the CPU receipt, 96-row manifest, exact
   hashes of all four splits, lens hash, and model ID before model loading.
3. **No benchmark path:** the self-contained task module has no benchmark import;
   the data receipt records the procedural firewall.
4. **Exact model:** model smoke passed pinned revision, 32 layers, width 2,560,
   bf16 SDPA, and batch one.
5. **Exact lens:** layers 4--8 each have rank 24 and the replicated byte hash.
6. **Exact aliases:** all 12 public aliases are distinct leading-space single
   tokens and the slot tokenizes as `[271, 5170, 25]`.
7. **Outcome-blind smoke:** the committed smoke receipt contains no correctness,
   chosen alias, trace text, or comparative metric.
8. **Trace cache:** native and close-only generation require one full prefill
   followed only by single-token forwards; one failure aborts the stage.
9. **Slot parity:** real, shuffled, and no-thought slot arms all use the same
   cache-free full-prefill function, close token, literal slot, alias IDs, dtype,
   backend, and deterministic argmax.
10. **Exact shuffle:** permutation is a deterministic hash order over positions;
    runtime requires identical length and sorted token multiset and stores source
    and shuffled sequence hashes plus moved-position rate.
11. **Unmasked diagnostics:** one forward supplies both constrained choice and
    full-vocabulary top token, total alias probability mass, and per-alias
    unmasked probabilities; there is no second numerically different replay.
12. **No-thought pairing:** one deterministic no-thought slot is computed per
    task and reused only as the task-level interface baseline.
13. **Free-form separation:** close-only generation uses the exact real thought
    prefix but is diagnostic and cannot enter `seam_gate`.
14. **Natural close:** a natural close at or before a cap replays only the thought
    before it and is labeled. It is not extended or counted as a fresh trace.
15. **Malformed termination:** EOS/short generation before a cap remains a
    nonfinite incorrect row in every denominator and is never force-replayed.
16. **Cardinality:** selection must finish exactly 48 traces, 144 real slots,
    144 shuffled slots, 144 free-form controls, and 16 no-thought controls before
    any files or summary are written. Confirmation has the analogous 48/48/48/
    16 counts at one cap.
17. **Control validity:** every actually evaluated shuffled and no-thought slot
    must be numerically finite; any nonfinite control aborts without a summary.
18. **Gate reachability:** CPU arithmetic passes. Each observed no-thought and
    shuffled baseline also receives an explicit receipt showing whether its
    required gain remains possible under the frozen 0.80 ceiling.
19. **Selection rule:** all three cap metrics are computed before choosing the
    first/smallest passing cap. No diagnostic arm can rescue a cap.
20. **Holdout lock:** confirmation verifies all five selection row-file hashes,
    including shuffled thought, and opens only the selected cap.
21. **No partial peeking:** row arrays remain in memory and scientific summaries
    are written only after cache, numeric, and cardinality checks complete.
22. **Later-stage firewall:** value, numeric-control, and causal CLI stages still
    raise a fatal unavailable error; a seam result cannot accidentally emit a
    J-space placeholder.
23. **Test coverage:** six CPU tests cover exact-depth freshness, gate arithmetic,
    natural/malformed boundaries, separated metrics, deterministic exact
    shuffle, and semantic-config hash sensitivity.
24. **Repository publication:** syntax, links, text, generated catalogs, site
    rendering, briefs, and dates all pass before the scientific run.

## Remaining limitations (accepted, not repaired post hoc)

- The alias mask deliberately changes the task to closed choice.
- A shuffled token sequence is an out-of-distribution bag-of-tokens control, not
  proof that every aspect of coherent reasoning is necessary.
- No-thought still performs transformer computation inside the fixed prefill.
- Cap selection uses gold labels and is followed by only one untouched
  confirmation split.
- A slot-seam pass is constrained elicitation, not a J result or matched-compute
  capability gain.

These limitations are already explicit in the immutable preregistration and do
not justify changing a gate after selection.

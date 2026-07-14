# Adversarial Design Review

**Verdict:** pass after claim narrowing and harness hardening. Expensive work is
authorized only after this checkpoint is committed, rebased, fully checked, pushed
to `main`, and both GitHub workflows pass.

## Primary threats and resolutions

1. **The parent was selected after observing a predecessor result.** `close_xi` is a
   post-result near-miss, so its prior local score is design input, not confirmatory
   evidence. Seed 88,006 is retired. The new run uses fresh seed 88,007, reports the
   parent again, and trains an active replay control from the same bytes and weights.
2. **This could merely repeat C56's oracle-trace failure.** The original curriculum
   narrated one dead branch and the correct decomposition only inside a full answer.
   This intervention separately scores apply, fit, reject, and execute interfaces,
   then includes full search. Any effect is attributed to the five-stage package,
   not to generic trace narration alone.
3. **The final ledger is not exhaustive search.** Its `1/2` and `2/2` entries show
   one known-dead first operation and the true pair even though the allowed universe
   is larger. The design therefore cannot establish exhaustive enumeration or a
   general search algorithm. The preregistration and report now call it a bounded
   two-branch demonstration; the separately supervised substates are the mechanism.
4. **Targets could be executable-looking but false.** The generator computes every
   transition over abstract integer indices. Independent tests reconstruct each
   operation universe from `_audit`, require exactly one fitting pair, require a
   genuinely dead first candidate, recompute every stage answer, and balance reject
   labels 8/8. The source is byte deterministic.
5. **Construction metadata could leak answers.** `_audit` fields are present only as
   ignored JSON keys. The trainer renders `messages`, `think`, and `answer`; no audit
   field enters a prompt or target. Local tasks are regenerated independently at the
   reserved fresh seed.
6. **Surface shortcuts could masquerade as procedure.** Eighty lessons span six
   surface families, including independently generated nonce tokens, four separators,
   sequence lengths 4--6, and cycle sizes 6--9. Canonical operation codes remain a
   deliberate interface choice, so a result supports transfer from that scaffold,
   not invariance to arbitrary operator language.
7. **Another 40 updates could explain the result.** `replay_after_close` starts from
   the identical parent and matches the candidate's 320 rows, 286,814 forward tokens,
   40 steps, optimizer, loss weights, and seed. It is an active treatment and a
   mandatory aggregate comparator.
8. **The replay contrast could silently use a different core.** Both streams reuse
   the predecessor's authenticated 200-row core. Tests locate exactly 200 identical
   line positions after deterministic shuffling. Candidate filler is disjoint from
   both inherited core and inherited control indices.
9. **Equal rows and steps could hide compute mismatch.** Both arms have exactly
   286,814 tokenizer-measured forward tokens and zero skips; batch size one removes
   padding-compute differences. The candidate and control have different length
   orderings after bucketing, which is an unavoidable part of changing examples and
   is not described as tokenwise gradient matching.
10. **Equal forward compute could hide objective-mass differences.** Candidate has
    8,209 more prompt tokens, 9,100 fewer thought tokens, and 891 more answer tokens;
    close tokens are equal. These exact allocations are published prospectively.
    The comparison isolates a curriculum package under matched forward compute, not
    an equal-label-token intervention.
11. **The predecessor's close-weight treatment could persist accidentally.** Both
    new arms inherit the same fixed parent but use ordinary thought and close weights
    of 0.2. Target-specific close CLI options were removed. Tests lock prompt, thought,
    close, answer, forced-close, negative-row, and overlength behavior.
12. **A forged or stale parent could invalidate lineage.** Training rejects any
    parent whose config/weight hashes differ from `de953bd5...7ff` and
    `16e9dc75...c179`. Merge accepts only the exact parent/control/candidate paths and
    authenticates trained arms against their receipts.
13. **Training one arm after seeing the other could invite adaptation.** Both data,
    wrappers, hyperparameters, and tests are frozen now. Control trains first only for
    operational checkpointing; no local generation occurs between training runs and
    no candidate setting can change without a new experiment.
14. **The local gate could reward formatting while missing the mechanism.** The
    unchanged aggregate bars are supplemented by `u_execute >= 0.50` and
    `u_induct >= 0.50`, plus parse, cap, and route-abstention checks. Only the one
    registered candidate can promote; parent and replay cannot substitute for it.
15. **A single fresh local seed is noisy.** It is a cheap mechanism screen, not a
    breadth claim. The parent, control, and candidate share the exact 26 cases, every
    completion is preserved, and failure seals the aggregate event rather than
    triggering threshold or seed tuning.
16. **Backend drift or benchmark leakage could manufacture breadth.** Benchmark
    access is conditional and aggregate-only through `run_benchmark_aggregate.py`.
    Exactly six explicitly merged models enter one `qwen_vllm` event; source inventory
    and runner signatures must match. Benchmark items, sources, transcripts, and raw
    streams remain unread.
17. **A quick pilot could be overstated.** The sole candidate must improve every
    reported family versus base and beat `blend`, inherited replay refresh, parent,
    and active replay in aggregate. Even that is exploratory: fresh quick replication,
    medium@2,048, paired uncertainty, and matched-compute sample-more remain mandatory
    in a separate confirmation experiment.
18. **Operational drift could make the scientific record unauditable.** Each natural
    stage must start from the published frozen checkpoint and end with a commit,
    fetch/rebase, full `make check`, push to `main`, and verification of both GitHub
    workflows. Training receipts capture `git_head` and `git_status`; no later stage
    begins from an unpublished worktree.

## Frozen artifact identities

- Parent weights/config: `16e9dc75...c179` / `de953bd5...7ff`.
- Scaffold source: `5854c218...a093`.
- Source-token receipt: `2d48f5e2...8200`.
- Predecessor partition: `abf8b505...0966f`.
- Stream manifest: `8035657a...dd8`.
- Token receipt: `eeb12b95...e4a0f`.
- Replay/candidate streams: `c157fb13...355d` / `79a8d7c9...0b90`.
- Construction seed 77,111; training seed 45; local seed 88,007; conditional
  aggregate seed 78,137.

All 43 experiment tests and the complete CPU smoke harness passed. No GPU model load,
training, local evaluation, merge, or benchmark event ran during this review.

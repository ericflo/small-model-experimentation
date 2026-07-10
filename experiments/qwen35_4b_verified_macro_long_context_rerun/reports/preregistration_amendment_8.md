# Preregistration amendment 8: resumable external full matrix

Date: 2026-07-10. Frozen while scientific smoke was still running, before its verdict and before
any train-only Qwen proposal or full-evaluation prompt. This amendment changes full-run execution
and artifact durability only; smoke still decides whether full is allowed.

## Motivation

The frozen full matrix contains 120 tasks. Without a Qwen-proposed library it has 9 arms and 14,400
completions; with one it has 15 arms and 23,040 completions. Completed 16k calibration/interface
metadata project roughly 102--125 million or 163--201 million sampled tokens respectively, already
22--28 or 35--45 generation hours before any higher selected rung. At observed JSON density, one
15-arm tier is about 2.68 GB, and the current physical promotion would duplicate it.

Whole-arm atomic calls would also lose two or more hours on interruption. These are infrastructure
risks, not scientific reasons to reduce tasks, K, controls, or the matched-compute baseline.

## Frozen full-only sharding protocol

1. Sort the 40 `no_reuse` and 80 `reuse` task ids independently. Form 40 canonical triplets
   `[no_reuse_i, reuse_2i, reuse_2i+1]` in that exact order.
2. Base uses K=24 and 20 shards, each containing two consecutive triplets (6 tasks, 144 stage-one
   sequences). Every K=12 arm uses 10 shards, each containing four consecutive triplets (12 tasks,
   144 sequences). Thus every shard has the exact 2:1 reuse mix, and each K=12 shard nests two base
   task shards. Task order, shard membership, K, prompt hashes, and batch shape are part of the Ada
   inference protocol and may not drift on resume.
3. Each shard writes into a temporary sibling directory, then atomically commits a directory
   containing `preflight.json`, `rows.jsonl`, `runner.meta.json`, and a last-written `receipt.json`.
   The receipt binds the shard-plan hash, ordered record ids, ordered prompt hashes/token counts,
   sampling/engine/runner/model/environment identity, and SHA-256 plus byte size of every file.
4. A final shard is reusable only after exact receipt validation. Never salvage individual records
   or samples from an interrupted shard; rerun the whole first incomplete shard. A malformed final
   shard fails closed and must be explicitly quarantined, never silently overwritten.
5. The fixed 144-sequence shard plan is a new prespecified vLLM protocol. Monolithic, differently
   ordered, or differently sized calls are not replay-equivalent on Ada and cannot be pooled.

## Canonical external artifacts

Full raw shards are canonical under
`/workspace/large_artifacts/qwen35_4b_verified_macro_long_context_rerun/full/`. The repository keeps
an atomic checksum catalog and compact summaries/verdicts, not multi-GB JSONL or a duplicate
promoted copy. The selected tier is a logical pointer in the catalog. Reanalysis must verify the
catalog, every expected receipt, path containment, sizes, hashes, selected budget, and the
`full_budget_selection.json` hash before reading rows.

Derived per-task JSON must omit full reasoning text and raw token arrays. It retains completion
hashes, aggregate counts/tokens, selected sample/program/grades, and the CSV/summary/verdict needed
to audit the estimand. Raw text and exact tokens remain available only in checksummed external
shards.

## Failed-rung short circuit

An operational error, OOM, interruption, missing cache, or validation failure never escalates the
thinking budget. Termination evidence can reject a rung early only after an irreversible full-arm
integer bound is crossed:

- base K=24, N=2,880: reject at 144 unresolved cap contacts or answer-limit contacts, or 721 exact
  periodic loops;
- every K=12 arm, N=1,440: reject at 72 unresolved cap contacts or answer-limit contacts, or 361
  exact periodic loops.

These are the first counts that cannot recover below the strict 5% cap/answer gates or at-or-below
the 25% loop gate even if every remaining completion is adequate. Once crossed, mark remaining
shards/arms skipped and start the next rung from shard zero for every arm. Lower rows remain
diagnostic and unscored. A selected rung still requires every registered arm, task, and sample to be
complete and adequate at one budget.

## Unchanged scientific boundary

No task, split, prompt, library, arm, seed derivation, K, model, answer limit, loop detector,
visible-only selector, hidden grader, bootstrap, matched-compute rule, or confirmatory threshold
changes. Sharding and external storage buy resumability and a landable repository; they cannot buy a
favorable score.

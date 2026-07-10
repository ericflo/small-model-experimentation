# Preregistration amendment 11: fail-early full-run audit hardening

Date: 2026-07-10. Frozen after the complete base-smoke think@32,768 call returned and while its
amendment-7/9 cache replay was still withheld, before any think@49,152 generation, train-only Qwen
proposal, full-evaluation prompt, or full artifact existed. This amendment follows an adversarial
code review of the amendment-8 implementation. It strengthens durability and validation only; no
scientific arm, task, sample count, budget, threshold, or estimand changes.

## Audit findings

The first amendment-8 implementation had several fail-late paths that could waste many GPU-hours or
allow offline analysis to trust more state than it independently verified:

- cross-shard identity compared volatile Git commit/dirty fields;
- existing later shards were not all validated before a missing earlier shard began generation;
- the shared receipt verifier checked fewer prompt/context/sample invariants than the live runner;
- the full selection did not independently prove registered, contiguous tiers beginning at the
  smoke-selected budget;
- hidden evaluation labels, libraries, thresholds, and analysis sources were not all bound into the
  catalog;
- the catalog existed only after selection, orphaning valid diagnostic receipts after interruption
  or an all-rungs-inconclusive outcome;
- stale repository-local raw files, malformed external-root entries, and duplicate `--full`
  processes could fail only after model construction or generation.

These are implementation defects, not observations about model output.

## Frozen hardening contract

1. **Protocol and evaluation binding.** The full catalog must bind exact byte hashes for the frozen
   config, complete tasks including hidden/probe labels, post-proposal libraries, demonstrations,
   dataset manifest, interface gate, passed smoke selection and smoke verdict, and the run,
   analyzer, artifact-store, harness, and domain sources that construct or score the comparison.
   Analyzer verification occurs before reading a raw shard. A hidden-label-only mutation must fail.
2. **Registered selection proof.** Full selection records its starting thinking budget and the
   passed smoke-selection hash. Tiers must be unique, ordered, and the exact contiguous prefix of
   the registered ladder from that start. The selected tier must be the first complete adequate
   tier; every earlier tier must have crossed a registered irreversible integer bound. Offline
   analysis rechecks these facts and recomputes selected-arm termination adequacy before scoring.
3. **Protocol identity versus provenance.** Every receipt retains complete provenance, including
   Git commit and dirty state. Equality-critical cross-shard protocol identity excludes only those
   incidental Git fields while retaining model/revision, runner, adapter, sampling, resolved
   sampling, engine and engine arguments, package/lock, GPU/CUDA/Python/vLLM environment,
   termination/token ids, and RNG settings. Current protocol identity is checked before generation
   and on cache reuse; inference-affecting drift fails before a new shard.
4. **Two-pass resume.** Under the fixed plan, scan and exact-validate every existing final shard in
   the active rung, including downstream arms, before generating any missing shard. A malformed,
   prompt-drifted, identity-incompatible, or unexplained final fails without a model call. A valid
   rename that beat a catalog checkpoint is reconciled into the inventory and never regenerated.
5. **Independent receipt trust boundary.** The shared validator checks preflight schema/pass/count,
   complete context-reserve arithmetic, ordered prompt hashes and token counts, row prompt
   identity, task/arm identity, exact sample indices `0..K-1`, summary request/completion counts,
   and structural token accounting. Runtime and analyzer use this same boundary.
6. **Inventory-first catalog.** The catalog inventories every valid diagnostic or selected receipt,
   not only a successful tier. Logical selection and the budget-selection hash are optional. It is
   checkpointed after startup reconciliation, every committed shard, every selection update, each
   rejected rung, and the final setup-inconclusive state. The one-way dependency remains
   `plan/config/data -> receipts -> budget selection -> catalog -> derived analysis`; selection
   never hashes the catalog.
7. **Fail-before-model filesystem audit and lock.** Hold a nonblocking OS `flock` on a persistent
   sibling lock file for the entire full stage, acquired before frozen-data checks, proposals,
   model construction, or artifact mutation. Before loading vLLM, reject any repository-local full
   JSONL, symlink, unknown budget/arm/shard, malformed final, or unexplained root entry. Temporary
   interruption directories are non-reusable and explicitly inventoried. A second invocation must
   fail before constructing the runner.
8. **Complete arm-set boundary.** The full plan contains exactly the nine registered non-Qwen arms,
   or all fifteen arms after an exactly-eight-entry Qwen-ranked library and all five matched Qwen
   random controls exist. Partial Qwen ensembles fail before generation.

## Required adversarial tests

Tests must cover Git-only provenance change versus package/GPU protocol drift; hidden-label
mutation; unregistered, skipped, or reordered budget tiers; prompt/reserve/sample-index mutation;
missing shard zero followed by malformed shard one with zero generation calls; interruption after
rename but before catalog; all rungs rejected with a selected-null catalog; duplicate invocation;
stale local raw data; unknown/symlinked root entries; operational error without budget escalation;
and the exact 20 base plus 10-per-macro happy-path shard geometry.

## Unchanged scientific boundary

The only model remains `Qwen/Qwen3.5-4B` through the frozen vLLM runner. The 40 canonical triplets,
144-completion shard shape, K=24 base, K=12 macro arms, ladder, 512-token answer allowance,
termination rules, visible-only selector, hidden grader, matched-compute sample-more baseline,
controls, bootstrap, and confirmatory conjunction remain exactly as preregistered. Hardening can
only reject invalid execution state earlier; it cannot improve a task score or choose a favorable
rung.

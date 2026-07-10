# Qwen3.5-4B Verified Macro Invention Long-Context Rerun Experiment Log

## Scaffold and branch decision

- Created as a new experiment after review of the stopped, result-bearing
  `qwen35_4b_verified_macro_invention`; the parent's outputs and historical numbers are not
  rewritten. After the corrected interface result, additive forward links were added to its
  README/report so readers do not mistake the old setup stop for a durable model limit.
- Attached to `operator_and_skill_inventories`, with secondary connections to structured execution,
  generalization, and test-time reasoning budgets.
- Related-work search named the parent as the closest near-duplicate. The operator-program backlog's
  exactly-one-macro slot sweep is a separate future interface redesign; this follow-up first tests
  the narrower claim that the parent's inference envelope was binding.
- Reclassified the parent's budgeted interface observation as setup censoring rather than a
  negative macro result: 16/16 samples hit think@768 and 12/16 hit the 128-token answer cap. The
  fresh v2 smoke and full tasks were never prompted, so the scientific hypothesis remains open.

## Frozen inheritance

- Copied the data and harness from parent commit
  `1c8c5bbb81d2a67618891597205ceb2f40f498d8` into this self-contained directory.
- Verified byte identity for all seven active data files and all seven historical
  `smoke_v1_frozen/` files against that commit. The archived v1 inputs remain part of the CPU
  contamination gate even though their seen smoke rows are excluded from scoring.
- Recorded all 14 parent paths and per-file hashes in `data/source_provenance.json`; `--prepare`
  independently checks both frozen sets against constants in the runner. The two headline active
  file hashes are
  `tasks.json = 82fbbd57e26fd392aa8f30ec6f26d370dc08dd78b3279bed6ee2e2174aea5073`
  and
  `libraries.json = a2ae3663753a3a0d0c9614a5d7c1d250506c74fd7879e11e99b66f5c1e43f865`.
- Did not copy parent model outputs into the new result. The historical v1 smoke remains seen and
  excluded; the copied v2 smoke and full splits retain their never-prompted status.

## Design freeze, before follow-up GPU generation

- Kept exactly one model, `Qwen/Qwen3.5-4B` at the repository-pinned revision, and exactly one
  inference backend: the experiment-local vLLM wrapper under the uv-managed `.venv-vllm`.
- Raised the engine envelope to `max_model_len=65536` and required prompt-plus-reserve token
  preflight. Concurrency may adapt to GPU memory; scientific token allowances may not.
- Froze the budgeted-thinking ladder at 2,048, 4,096, 8,192, 16,384, and 32,768 tokens, with a
  512-token answer allowance at every rung.
- Froze metadata-only calibration to four deterministic train-only plan-given records at n=16.
  The smallest rung must have cap and truncation each below 5%, p99 thinking use at most 75% of B,
  and p99 answer use at most 50% of A. Output text and correctness are unavailable to selection.
- Froze a separate heldout train-only interface gate of 16 records at n=4. At least 12 records need
  one strict macro-using optimal surface whose literal expansion exactly equals the supplied plan;
  cap and truncation must each remain below 5%.
- Froze whole-stage escalation: any censoring violation reruns the complete gate, proposal batch,
  smoke matrix, or full matrix at the next rung. Lower rungs are diagnostic only and are never
  pooled. Exhausting the ladder is setup-inconclusive.
- Preserved the parent's scientific arms, K=12/K=24 sampling, fresh v2 smoke, full tasks, visible-
  only selection, analyzer, hidden-label boundary, and confirmatory effect/interval thresholds.
- Saved `idea_intake.md`, `reports/preregistration.md`, and `reports/design_review.md` before the
  first follow-up model call.

## GPU runs

### Auto-async calibration excluded before science

The first train-only calibration produced complete tiers at 2,048, 4,096, and 8,192 tokens. The
first two tiers were fully cap-bound (64/64 each). The 8,192 tier reduced cap contacts to 12/64,
but the registered prefix audit found 32/64 mismatches against think@4,096, all in the two records
scheduled after the first 32 logical sequences. Engine logs confirmed that vLLM 0.24 had
auto-enabled asynchronous scheduling.

Stopped before heldout interface or fresh induction smoke. Archived the auto-async rows and exact
runner, added `async_scheduling=False` to the experiment-local and template wrappers, bumped the
runner schema, and froze `reports/preregistration_amendment_1.md` before restarting calibration.

The non-async 2,048→4,096 rerun then reproduced 32/64 prefix mismatches on the same second
scheduling wave. This refuted the async-specific attribution: the Ada GPU cannot enable vLLM's
Hopper-only batch-invariant mode, so cross-budget common random numbers are unavailable. Frozen
`reports/preregistration_amendment_2.md` before continuing. Prefix audits remain recorded but no
longer gate the metadata-only budget selector; valid non-async tiers are retained.

The complete ladder then found 64/64, 61/64, 18/64, 9/64, and 9/64 raw cap contacts at budgets
2,048 through 32,768. At 32,768, the 55 naturally closing samples all finished by 12,564 tokens;
the other nine had exact periodic tails over the final 8,192 tokens (periods 5–318), and no answer
truncated. Because the registered calibration was setup-inconclusive, inspected only these
train-only loop tails, froze the token-only detector and thresholds in
`reports/preregistration_amendment_3.md`, and kept every scientific prompt untouched.

The 32k heldout interface then passed 16/16 records with all 64 samples valid and macro-using,
confirming the parent's apparent interface failure was setup-induced. The first fresh base-smoke
call ran for about 22 minutes at full utilization but returned no rows before manual interruption;
no content or correctness was inspected. Calibration showed think@16,384 already covers every
non-loop trace (natural p99 12,564) with 3,820 tokens of headroom. Froze amendment 4: 80% natural
headroom, loop handling from 16k, max-num-seqs 64, targeted recalibration, and complete interface /
smoke reruns under the faster protocol.

The targeted max-seqs-64 calibration selected think@16,384, and the independent heldout interface
again passed 16/16 records. The complete fresh base-smoke arm then showed that the scientific
workload has a different termination distribution: all 144 samples contacted the thinking cap,
13 were exact periodic loops, 131 remained unresolved (90.97%), and 60 answers truncated at 512
tokens (41.67%). vLLM sampled 2,391,698 tokens in 2,138.606 seconds (1,118.34 sampled tokens/second).
No generated text, parser output, correctness, oracle result, or task score was inspected. The
automatically started designed-ceiling arm was interrupted before it returned any row.

Froze `reports/preregistration_amendment_5.md` before another GPU call. Base alone makes the 16k
rung ineligible, so failed lower rungs are diagnostic only and may short-circuit after a completed
arm proves the full rung cannot pass termination gates. Extended the ladder to 16,384, 32,768,
49,152, and 61,440 with the answer allowance and 65,536-token engine context unchanged. At the
largest rung, the observed 990-token maximum prompt plus 61,440 thinking tokens, the two-token
injected close, and 512 answer tokens total 62,944. The next complete termination-adequate matrix,
not any lower censored row, is eligible for smoke scoring.

Preflight arithmetic correction, found while think@32,768 was still running: 990 is the maximum
base prompt, while the frozen designed preflight reaches 1,060. The true matrix-wide largest-rung
bound is 63,014 tokens, leaving 2,522 below the unchanged 65,536 context. No protocol changed.

The same CPU-only tokenizer audit found that the frozen train-only proposal prompt is the overall
maximum at 3,478 tokens. Its largest-rung prompt plus reserves totals 65,432, leaving 104 tokens.
The registered ladder fits, but any later extension must raise `max_model_len` rather than consume
the guard band.

Before complete base@32,768 returned, froze `reports/preregistration_amendment_6.md`. If either
completed K=12 arm rejects the 32k matrix, remaining higher rungs use all 12 base-smoke prompts at
K=4 as termination-only workload probes. Only the first probe-adequate rung receives a fresh
complete K=12 base/designed matrix; probe rows never enter scoring. If both 32k arms are adequate,
this conditional branch is unused.

The active Python process had loaded the pre-amendment code before that branch was frozen. Added a
temporary non-model-facing sentinel at the 49,152 K=12 preflight path: if the 32k matrix rejects,
the old process fails closed on frozen-artifact mismatch before it can generate a 49k row. Remove
the sentinel before any amendment-6 restart; if the 32k matrix passes, it is never consulted.

Before base@32,768 returned, a runner-code audit found that naturally closed stage-1 answers were
not directly capped at 512 even though forced-close stage-2 answers were. Froze
`reports/preregistration_amendment_7.md`: any `n_answer_tokens >= 512` is now an answer-limit
contact, regardless of stage or finish reason. Existing rows are reclassified from metadata; a
post-process cache rerun is mandatory before accepting the active process's result.

While smoke was still unresolved and before any full/proposal prompt, froze
`reports/preregistration_amendment_8.md`. If smoke passes, full uses fixed balanced
144-completion shards with atomic receipts, exact ordered cache binding, canonical external raw
storage, logical (non-copying) promotion, compact derived results, and only irreversible-count
failed-rung short circuits. Tasks, arms, K, and scientific thresholds are unchanged.

An independent code audit before base@32,768 returned then found that the legacy contact rule
conflated a raw stage-1 length finish with exhausted reasoning even when `</think>` had occurred
earlier and the runner regenerated the discarded partial answer under the fresh 512-token allowance.
Froze `reports/preregistration_amendment_9.md`: reasoning contact is now forced intervention or
`n_thinking_tokens + 1 >= B`; raw stage-1 length, forced intervention, final-slot boundary contact,
and earlier-close answer restart are separate diagnostics in runtime selection and analysis. The
same audit froze a CPU proposal-envelope regression: record hash
`df4735015e69149acba33eab02156ed56252ddb512a7bc669efc99b3a1c51e7d`, 3,478 prompt tokens,
65,432 tokens at the largest rung, and 104 tokens of headroom. No model output informed either
repair, and exact-valid cached rows must be replayed through the amended classifiers.

Implemented that full-only durability path without launching inference. The runner now freezes the
40 canonical no/reuse/reuse triplets, nests 20 base shards inside 10 macro-arm shards, commits only
last-receip-valid external directories, and refuses to overwrite malformed finals. The analyzer
verifies the plan, selection hash, root containment, every expected receipt/payload hash, and exact
sampling/engine/runner identity before reading selected rows. Model-free tests cover task balance,
atomic commit shape, corruption/path-escape failures, exact irreversible thresholds, rung
short-circuiting, catalog completeness, multi-shard accounting, and compact derived output.

## External scientific-smoke durability implementation

After base@32,768 returned atomically but before amendment-9 classification or any output-content
inspection, froze `reports/preregistration_amendment_10.md`. Scientific matrix and termination-probe
bundles now use one external root with a preflight-only resumable state and a receipt-last complete
state. Each receipt binds all three runner files, ordered prompt/input identities, task order, K,
arm, role, budget, model/revision, experiment-local runner hash, sampling, and engine identity.

The tracked deterministic catalog binds every external file plus the exact tasks, demonstrations,
config, base/designed library payloads, analyzer, domain, harness, runner orchestration, and storage
implementation. Logical selection points directly at complete matrix receipts; probes cannot be
selected and no `runs/smoke/` promotion copy is created. The analyzer verifies the catalog and all
selected receipts before reading a row, and derived smoke output now omits completion prose and
token arrays just like the full path.

Added a model-free `--migrate-scientific-artifacts` path that stages and exactly validates legacy
local caches before atomic installation. It preserves local sources by default; the explicit
`--remove-local-scientific-artifacts` follow-up revalidates the installed tree before deleting only
canonical local tier/probe/promotion directories. Model-free tests cover idempotence, delayed local
removal, guard rejection, protocol mutation, analyzer verification-before-parse, logical promotion,
and fresh-clone preparation. No migration or inference was performed as part of this implementation.

## Fail-early full-run audit hardening

Before any train-only Qwen proposal, full prompt, or full artifact existed, froze and implemented
`reports/preregistration_amendment_11.md`. Scientific migration, smoke, and full now share one
persistent nonblocking sibling `.full.lock` before frozen-data verification, smoke replay,
proposal work, model construction, or artifact-root/catalog mutation. Its model-free preflight rejects local full JSONL, symlinks, unknown
tree entries, malformed finals, partial Qwen ensembles, and stale exact bindings. The only valid
full arm boundaries are the nine registered non-Qwen arms or all fifteen arms after the exactly
eight-entry Qwen-ranked library and five matched Qwen controls exist.

Full resume is now two-pass: it validates every existing final in the active rung, including
downstream arms, before generating any missing shard. Every cache hit and generation is checked
against the current inference-critical protocol identity; Git commit/dirty remain audit provenance
but package/lock, GPU/CUDA/Python/vLLM environment, model, runner, adapter, sampling, engine,
termination-token, and RNG fields remain strict. Operational failures propagate without silently
escalating the reasoning budget.

The schema-2 full catalog is inventory-first and exists before selection. It checkpoints startup
reconciliation, each committed rename, selection updates, rejected rungs, and final selected or
selected-null setup-inconclusive states. It binds exact bytes for config, complete hidden/probe
tasks, post-proposal libraries, demonstrations, manifest/provenance, passed interface and smoke
evidence, and all construction/scoring sources. Offline analysis verifies this graph and the
registered contiguous-prefix selection proof before reading raw rows, then recomputes every
selected arm's termination evidence before task scoring. Added adversarial model-free coverage for
hidden-label mutation, selection-order/bound failures, duplicate invocation, stale/unknown/symlink
state, two-pass zero-generation failure, operational-error non-escalation, and exact 9/15-arm
geometry. No inference or scientific artifact was created or modified during this implementation.

The final orchestration audit also made local cleanup subset-safe and catalog recovery monotonic.
An interrupted cleanup may resume when every remaining local bundle is byte-identical to a subset
of the external inventory, including a preflight that has since completed externally. Under the
shared lock, a stale catalog may reconcile only an existing preflight-only entry advancing to an
exact receipt-valid completion with unchanged protocol/source and preflight bytes. New bundles,
modified files, partial rows/metadata, source drift, or selection-semantic drift still fail closed.

## Scientific cache migration and amendment replay

After the old process exited, removed one byte-identical notebook checkpoint and migrated the
complete base@16k/base@32k bundles plus the genuine designed@16k preflight-only state through the
model-free two-pass path. The canonical external tree initially contained 9 files and 92,301,251
bytes with tree SHA-256 `fdba92ffc046c7710e72bb6a772f253c7c5a059027a4110f49018e00a189b38d`.
Both row hashes matched their source audits (`f4d243db...` at 16k and `6f3c6036...` at 32k), both
receipts reverified, and the second pass removed the entire local `runs/smoke_tiers/` namespace.

The amendment-7/9 replay then rejected complete base@32,768 from content-blind finish/count
metadata plus the preregistered token-ID periodicity test. All
144 samples used the full reasoning allowance and required forced intervention; there were no
earlier-close answer restarts. Eighty-one tails were exact periodic loops, 63 contacts remained
unresolved (43.75%), and 37 fresh answers hit the 512-token limit (25.69%). The arm sampled
4,739,527 tokens in 5,971.182 seconds (793.73 tokens/s). Base alone irreversibly rejected the rung,
so designed@32k was never generated or inspected. No decoded output, parser result, task score,
hidden grade, or oracle field informed the decision.

The frozen amendment-6 branch therefore started the all-12-record, non-scored K=4 base probe at
think@49,152 through vLLM. Probe rows are receipt-bound outside the matrix namespace and are
ineligible for selection, scoring, promotion, or prefix pooling. Its termination result remains in
progress; a probe pass would authorize a fresh K=12 base/designed matrix at 49,152, while a probe
failure would advance only the same termination-only probe to 61,440.

## Pre-result capacity-fit scheduler branch

While the max-seqs-64 K=4 base probe at think@49,152 was still inside vLLM and before its receipt or
termination result existed, froze `reports/preregistration_amendment_12.md`. Engine-start evidence
had already measured 995,328 KV-cache tokens, and the installed vLLM scheduler source confirms that
decode-time cache exhaustion preempts a request by freeing its blocks and resetting its computed
prefix to zero when prefix caching is disabled. The registered long-rung sequence envelopes admit
only 19 concurrent sequences at 49k and 15 at 61k; max-seqs 64 therefore permits avoidable full
recomputation.

Armed a receipt watcher that stops the current process group only after the 49k probe commits its
last-written receipt. The completed max-seqs-64 probe will remain verbatim but is now irrevocably a
non-scored, non-selectable `scheduler_overcommitted` diagnostic. It cannot authorize a matrix or a
61k call. Fresh capacity-fit selection moves to the independent
`qwen35_4b_verified_macro_capacity_fit_rerun`, with max-seqs 19 at 49k and 15 at 61k, a new external
namespace, and fresh K=4 rows before any same-protocol K=12 base/designed matrix. No active probe
token, decoded output, parser result, or score informed this branch.

## Scheduler-overcommitted 49k diagnostic completion

The receipt watcher stopped the old process group immediately after the max-seqs-64 K=4 probe at
think@49,152 committed its last-written receipt, before it could validate/advance into another model
call. Model-free catalog reconciliation verified the complete 13-file external tree at 123,389,281
bytes and SHA-256 `add7cc9ef14f1a47b9cd23fd645cfbf65cbe1347c3e5dd2130d82c9a70d79612`.

Content-blind replay found 48/48 stage-1 length finishes, forced interventions, and reasoning-boundary
contacts; 34 were exact periodic loops, 14 remained unresolved, and 13 stage-2 answers reached 512
tokens. The call sampled 2,366,620 tokens in 4,035.356 seconds (586.47 tokens/s), continuing the
throughput decline from 1,118 tok/s at 16k and 794 tok/s at 32k. It failed all three termination
gates, but its eligibility had already been revoked before the receipt: with block size 528, 48
admitted worst-case contexts require 2,433,024 cache tokens against the measured 995,328-token cache,
and vLLM recomputes evicted prefixes when prefix caching is off.

No decoded output, parser status, visible/hidden score, or candidate correctness was inspected.
The reconciled receipt remains external and the compact audit is
`analysis/scientific_smoke_49k_scheduler_overcommit_audit.json`. After verification, the stopped old
engine was terminated and released the GPU; no 61k/max-seqs-64 call or K=12 matrix began.

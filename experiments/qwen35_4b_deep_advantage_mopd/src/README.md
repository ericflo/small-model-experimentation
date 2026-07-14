# Source

Experiment-local routing, replay, target-cache, loss, and training helpers.

Completion-target sequence fitting is deliberately centralized in
`training_units.fit_prompt_around_completion`. The entire generated completion
and its registered natural target positions are preserved; if the concatenated
sequence exceeds the frozen training `max_length`, only the oldest prompt
tokens are removed and the cut is receipted. That fitted representation may be
cached for a matched control, but `capability` and `anchor` cache creation fails
before policy scoring if either prefix was shortened. Every trainer independently
rejects shortened samples selected by its arm. A completion that leaves no
causal prompt token remains a hard failure.

All-policy cache provenance is checked at creation, orchestration resume, and
training. The receipt must bind the frozen config/top-k and the resolved quick,
deep, and soup model paths, model configs, and merge receipts.

The non-advantage-route control has an additional full-prefix overlay path.
`route_control_matching.py` defines the frozen tier/identity order, while
`control_rematch.py` replays that matcher after filtering candidates solely by
whether the complete observed prefix and completion fit `max_length`. The
unfiltered replay must reproduce the original manifest, and the filtered replay
may change only controls whose cached prompt was shortened. The derived cache
keeps the primary manifest/cache immutable, copies every unaffected sample at
the same index, and rescales no target: only replacement states are scored
again under the same quick/deep/soup policies. Selection and cache receipts
bind all candidate artifacts, tokenizer files, source hashes, replacement
identities, copied-sample semantic hashes, and the zero-truncation inventory.
The overlay writer commits JSON/cache files atomically and resumes only from a
validated manifest-only or manifest-plus-cache publication prefix. It never
deletes published prefix evidence; cache-only, receipt-only,
manifest-plus-receipt, unknown, and symlinked states all fail closed.

`control_receipts.py` is the single semantic validator for trained controls.
Both orchestration resume and the independent authorizers require the exact
primary initial-loss pressure, its implied loss scale, the registered 60/20
unit and 6/2 probe geometry, full prefixes, consume-once IDs, and arm-specific
targets. The validator reconstructs the exact assignment and shuffled ledger
from the bound round manifest and target cache (or the manifest and local base
tokenizer for off-policy SFT), so a forged self-rehashed ledger cannot pass. A
separate no-clobber controls authorization reuses the benchmark audit, proves a
stable recursive inventory of runner/trainer/builder/auditor code before and
after the audit, and runs before any sealed confirmation model is loaded.

`model_provenance.py` is the shared checkpoint-byte authenticator for controls,
confirmation, and benchmark execution. It permits exactly the seven root files
emitted by this pipeline, rejects every symlink, nested entry, and non-regular
artifact, authenticates the complete weight and inference inventories, and
accepts only the frozen source or local tokenizer load profile. The committed
quick/deep/soup receipt and its ancestor commit are trust roots. Before
confirmation, the controls authorizer seals one exact 13-arm map; orchestration
rehashes it before and after global `ADMISSION`, and each evaluator rehashes its
one authorized arm immediately before `STARTED` and after generation.

Sealed confirmation adds two isolated modules without changing the shared
acquisition runner or harness. `confirmation_protocol.py` fingerprints the
exact pinned vLLM/Python/package/lock/GPU/CUDA and engine protocol, proves live
live-derived hybrid-cache capacity with
`ceil(tokens/528) + 3*ceil(tokens/16384)` (35 blocks at full context), and
journals every returned generation
call before harness scoring. `confirmation_artifacts.py` owns the no-clobber
`STARTED -> GENERATED -> COMPLETE` transaction, score-last publication,
verification-only resume, and terminal quarantine. It authenticates sampled
token totals from retained stage-1/stage-2 token IDs, binds exact task bytes and
ordered plans, authenticates the full returned request/output hashes and exact
raw/resolved sampling settings, and deterministically replays atom scores and
episode transitions from journaled model text. It rejects resampling or cleanup
after any started attempt.

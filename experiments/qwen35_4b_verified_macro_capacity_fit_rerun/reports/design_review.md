# Adversarial design review: verified-macro capacity-fit vLLM rerun

Date: 2026-07-10, before any GPU call from this experiment. Initial verdict: **launchable only after
the controls below are implemented and independently rechecked**. The committed harness and tests
encode the listed resolutions.

## 1. A larger token cap alone does not fix scheduler geometry

At `max_num_seqs=64`, a scheduler can admit far more near-50k contexts than a roughly 995k-token KV
cache can hold. Long latency, preemption, or recomputation would then be an infrastructure artifact,
not a measure of reasoning or macro induction.

**Resolution:** freeze capacity-fit concurrency at 19 for 49,152 and 15 for 61,440. After each
engine is built, use its live KV token capacity and block size to conservatively fit every active
sequence at the longest prompt plus the complete registered reserve. Refuse generation on any
failure. Record the full calculation in the receipt-bound preflight.

## 2. Historical cache capacity can drift

The prior 995,328-token figure depends on hardware, vLLM, model allocation, memory utilization, and
other runtime details. Treating it as guaranteed would merely move the hidden assumption.

**Resolution:** use that number only to preregister 19/15. The live engine must independently expose
positive `kv_cache_size_tokens` and block size, report model context exactly 65,536, and pass the
calculation before rows exist. Runtime identity and packages are receipt-bound.

## 3. Context length and cache capacity are distinct limits

A batch may fit total KV capacity while one request exceeds the model context, especially after
adding the forced-close sequence and answer stage.

**Resolution:** preflight every exact tokenized prompt against
`prompt + thinking budget + 2 + 512 <= 65536`, and separately check block-rounded aggregate cache
fit. The longest designed prompt leaves 2,522 context tokens at the 61k rung.

## 4. Reducing concurrency can silently change another engine knob

If CUDA graph capture, async scheduling, prefix caching, dtype, or token batching changes with the
new `max_num_seqs`, any improvement has an ambiguous cause.

**Resolution:** freeze every engine argument other than the registered rung pair. CUDA graph capture
equals the rung's concurrency rather than retaining an invalid 64 ceiling. Probe and K12 receipts at
one rung must match on normalized runtime/engine identity after removing only K-specific sampling.

## 5. Reusing parent rows would manufacture a cheap “fresh” matrix

Old K4 or partial K12 rows came from `max_num_seqs=64` and therefore do not instantiate the new
protocol. Reusing them would mix scheduler regimes and may select favorable survivors.

**Resolution:** use a new external root, reject any root overlap or symlink alias, import no output,
and require new prefixes/receipts. Parent artifacts are diagnostic-only and not required for this
experiment. K4 and K12 have distinct namespaces, batch geometries, roles, and catalog IDs; seed
derivation remains the same frozen protocol and is not the nonreuse boundary.

## 6. The v2 smoke is no longer model-unseen

The direct parent has already submitted these fixed prompt identities. Calling them “fresh” would
overstate the evidence even though the model is stateless and no weights were updated.

**Resolution:** describe the matrix precisely as frozen and previously unscored under an independent
capacity-fit protocol. Freeze prompt identities from preflight-only fields, import no decoded model
output, and make no contamination-free capability-gain claim from this smoke.

## 7. A K4 probe can accidentally become K12 evidence

If a passing probe is copied, pooled, or pointed to as the base arm, the semantic matrix has
different K and compute across arms.

**Resolution:** K4 is `termination_probe_only`; selectable entries must be receipt-valid
`complete_matrix_arm` bundles with exactly K=12 and 12 records. Catalog construction and resolution
reject a probe pointer. Tests explicitly attempt and fail to resolve K4 as selected base.

## 8. Looking at decoded output while choosing the rung is outcome shopping

Parser success, macro use, correctness, or qualitative answer inspection could make 49k versus 61k
a hidden accuracy hyperparameter.

**Resolution:** termination selection reads counts and finish metadata plus token identity only for
the frozen periodic-tail detector. It never reads decoded text or scores. Tests mutate all decoded
strings and correctness fields and require an unchanged decision.

## 9. Periodic loops are still a cap contact, but not unresolved censoring

Long contexts may enter stable token loops. Treating every forced close as unresolved would make a
known model behavior force unbounded escalation; ignoring loops would hide a bad operating point.

**Resolution:** preserve the predecessor's exact token-periodicity classifier and report loops
separately. A rung allows at most 25% periodic-loop contacts and strictly less than 5% nonperiodic
boundary contacts. The selection record explicitly states that token identity—not decoded content—
enters only this detector.

## 10. Arm-by-arm escalation can bias the designed comparison

Carrying a passing base arm from 49k into a 61k designed arm confounds representation with budget
and creates a favorable adaptive mixture.

**Resolution:** any inadequate probe, base arm, or designed arm rejects the entire rung. The next
rung starts with a new K4 probe and, if eligible, both new K12 arms. The first adequate contiguous
rung is the only selectable matrix.

## 11. A hand-edited selection file can erase lower failures

Even content-blind metrics permit outcome shopping if a lower adequate tier is marked inadequate or
omitted so that a later rung can be selected.

**Resolution:** selection must contain the exact contiguous ladder prefix. Before semantic access,
the analyzer verifies every referenced receipt and recomputes every lower and selected termination
dictionary byte-for-value from receipt-bound rows. Any drift fails. Tests cover an edited lower-rung
adequacy history.

## 12. Crashes between preflight, rows, receipt, and catalog can create ambiguous resume state

Without a strict commit protocol, a rerun could silently overwrite a partial call or forget a
completed bundle after an interruption.

**Resolution:** checkpoint preflight in the catalog, atomically write rows/metadata, and write the
receipt last. Reconciliation permits only preflight-only to complete for the same identity. Unknown
files, uncheckpointed artifacts, partial bundles, mutation, disappearance, or unexpected catalog
entries fail closed. Success and terminal-failure finalization are idempotence-tested.

## 13. Alternate artifact roots can defeat mutual exclusion

Putting the lock inside an environment-overridden root would let two runs select different roots and
simultaneously occupy the same GPU or mutate tracked selection files.

**Resolution:** use one fixed nonblocking lock at the experiment's canonical parent directory,
independent of the output override, and hold it from inventory validation through engine closure,
receipt commit, and catalog refresh. GPU-owner coordination remains required because the lock cannot
govern unrelated experiments.

## 14. Semantic analysis can leak before both arms are valid

Parsing or inspecting base decoded text before the designed receipt and both termination checks
succeed exposes outcome information that could influence recovery.

**Resolution:** analysis has three passes: verify the complete history and both selected receipts;
read token rows and recompute both termination gates; only then load task/library content, decode,
and hidden-grade. Tests assert semantic helpers are untouched on missing or inadequate arms.

## 15. A terminal setup failure needs an explicit artifact

If 61k fails and the runner merely stops, future work may interpret the absence of a selection as an
interrupted run and retry selectively.

**Resolution:** a completed rejection at the final rung writes a validated `pass:false` selection
with both tier histories, `selected_thinking_budget:null`, and catalog `selected:null`. Re-running
the finalizer is byte-idempotent.

## 16. A positive designed smoke is not the research claim

The designed library is generator-known and smoke uses no mined, hint, random, or matched-compute
sample-more controls. Treating a smoke pass as verified invention would bypass the repository's main
scientific standard.

**Resolution:** call this an interface/capacity gate only. A pass licenses a separate preregistered
experiment retaining the full mechanism controls and matched-compute baseline. A failure localizes
the frozen interface; neither outcome alone proves learned macro value.

## Review verdict

The implementation now contains the required source hashes, live dual capacity checks, one-engine
phase boundaries, fresh namespace, K4 nonpromotion, runtime parity, strict state machine, receipt-last
storage, checkpoint reconciliation, first-adequate proof, terminal-unselected history, and three-pass
analysis. Proceed only after the full CPU suite and model-free validator pass at the launch commit,
an independent read-only audit issues GO, and the current GPU owner confirms the device is free.

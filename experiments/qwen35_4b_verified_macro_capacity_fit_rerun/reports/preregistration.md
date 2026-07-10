# Preregistration: verified-macro capacity-fit vLLM rerun

Status: **frozen on 2026-07-10 before this experiment's first GPU call**. At freeze time no model
output existed in this experiment's external namespace. `configs/default.yaml`, the committed
runner/analyzer, and receipt validators—not manual judgment—control every branch.

## Question and scope

Can the frozen verified-macro v2 smoke reach an interpretable, nonbinding termination regime when
vLLM scheduler concurrency is fitted to the live KV cache rather than left at 64?

This is an inference-validity follow-up, not a capability-gain experiment. It can establish that a
base/designed smoke comparison is eligible, or localize a remaining termination/interface boundary.
It cannot establish that learned macros beat matched-compute primitive sampling.

The direct parent is `qwen35_4b_verified_macro_long_context_rerun` at source commit
`d22d67636f1694d49100176aaf10c4b7e83beb51`. The parent has already rendered/called the v2 smoke
prompt identities under other scheduler settings, although it had not released a scored semantic
v2 result when this protocol was frozen. Accordingly, these tasks are described as a **frozen,
previously unscored matrix under an independent capacity-fit protocol**, not as fresh or model-unseen.

## Frozen inputs and noninheritance boundary

The following files are byte-identical copies whose exact SHA-256 values are enforced before model
construction:

- `data/tasks.json`: `82fbbd57e26fd392aa8f30ec6f26d370dc08dd78b3279bed6ee2e2174aea5073`
- `data/demonstrations.json`: `1531b2722c5dc64530cbafda3e20a3de8a52ab537e50dc35ee4ec50a9fae06cf`
- `data/libraries.json`: `a2ae3663753a3a0d0c9614a5d7c1d250506c74fd7879e11e99b66f5c1e43f865`
- `src/macro_domain.py`: `3a59b931faf42a6731ad73e31f9e8cdedf44c29423db4d8645b4e50a66ab21a7`
- `src/model_harness.py`: `a43fb0e76f65819e5d1048f965e74c06409da870e77c7ce46f6df247257fa552`
- `src/vllm_runner.py`: `fd9972bdcb3a9e8b9841b45ed8e2849017a6e80b601e924817cdaaa5144b8782`

The base and designed model-facing record-list hashes are respectively
`bd66aa64942f9e57e1fe55ae716c154ea1231480d6163f1811a07828ba364907` and
`c5a6cd00d9600b7a63c8e2c132e202b25da30f30af299afb3735a8f5525d9e86`.
`data/prompt_manifest.json` freezes the ordered record IDs, input-record hashes, rendered-prompt
hashes, and tokenizer counts observed in content-blind predecessor preflights. It contains no output
field. Every new engine must reproduce that manifest exactly.

No predecessor JSONL, metadata, receipt, termination verdict, parse result, or score is imported or
required. Parent artifacts made with `max_num_seqs=64` are permanently diagnostic-only and may not
be pooled, promoted, selected, or used to fill missing samples.

## Model and backend

The sole model is `Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. All inference uses the byte-identical local
`src/vllm_runner.py` in the uv-managed `.venv-vllm`; Hugging Face Transformers inference and mixed
backends are forbidden.

Each GPU invocation constructs one vLLM engine, invokes `generate_vllm_batch` once for one
experiment phase, writes one receipt-complete bundle, closes the engine, and exits. Budgeted
generation may contain a stage-one vLLM call and a stage-two continuation call for forced closures.
Separate phases therefore never share model or KV state. Runtime identity, installed-package
versions, GPU identity, engine arguments, runner hash, model revision, sampling parameters, seed
rules, and termination token IDs are bound into the receipt. Within a rung, the probe and both
matrix arms must have identical comparable protocol identity except for K.

## Capacity geometry

The engine context is 65,536 tokens. Every completion reserves:

`thinking budget + 2 forced-close tokens + 512 answer tokens`.

The ladder and concurrency mapping are frozen:

| Rung | `max_num_seqs` | Longest designed prompt | Reserved total | 16-token rounded context | Concurrent requirement at planning capacity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 49,152 | 19 | 1,060 | 50,726 | 50,736 | 963,984 |
| 61,440 | 15 | 1,060 | 63,014 | 63,024 | 945,360 |

The final column is compared with the predecessor engine's observed 995,328-token cache only to
choose the mapping. It is not accepted as the new engine's capacity. After constructing each new
engine and before writing any row, the preflight must read and bind:

- `vllm_config.cache_config.kv_cache_size_tokens`;
- live cache block size;
- live model `max_model_len`, which must equal 65,536;
- the current arm's exact rendered prompt-token counts;
- logical sequence count and `active=min(records*K, max_num_seqs)`; and
- `active * ceil(max(prompt + reserve)/block_size) * block_size`.

Generation is forbidden unless the live requirement is no greater than the live capacity. The
engine also fixes `gpu_memory_utilization=0.9`, `max_num_batched_tokens=32768`, prefix caching off,
eager mode off, asynchronous scheduling off, bfloat16, tensor parallel size one, and CUDA-graph
capture no greater than that rung's `max_num_seqs`.

## Sampling and experimental units

The 12 smoke tasks comprise six reuse and six no-reuse procedural DSL tasks. The two matrix arms are:

1. `base`: primitive inventory only.
2. `designed_ceiling`: the frozen generator-known recurring composites, callable as neutral aliases.

Both arms use K=12, temperature 0.6, top-p 0.95, top-k 20, run seed 2701, budgeted thinking, and a
512-token answer allowance. Demonstrations, visible I/O, surface limit five, expanded primitive
depth limit five, prompt template, strict parser, and hidden tasks remain identical across arms.

Each rung begins with a base K=4 termination probe. K4 is deliberately too small for semantic
selection and is stored under `smoke_budget_probes`, with receipt role `termination_probe` and tier
mode `termination_probe_only`. Selectable rows live under `smoke_tiers`, require K=12 and receipt
role `complete_matrix_arm`, and use distinct prefixes and batch geometries. Stable seed derivation
is intentionally shared by protocol; the safety boundary is that no K4 byte can fill, augment, or
replace a K12 arm.

## Content-blind termination classifier

For every completion, the classifier uses only registered token counts, finish metadata, and token
IDs for a frozen periodic-tail detector. It does not use decoded strings, parse results, correctness,
macro use, visible score, hidden score, or task selection.

A reasoning-boundary contact is a forced close or `n_thinking_tokens + 1 >= thinking_budget`.
Boundary contacts whose retained thinking tail contains at least 8,192 token IDs are classified as
periodic when some period from 1 through 2,048 matches at least 99% of the tail comparisons. A
boundary contact without such periodicity is unresolved. An answer-limit contact is any truncated
completion, answer-stage length finish, or answer count at least 512.

A batch is adequate exactly when:

- unresolved contact rate is strictly less than 0.05;
- answer-limit contact rate is strictly less than 0.05; and
- periodic-loop contact rate is at most 0.25.

Token identity is used only for periodicity and is explicitly disclosed in every selection record.
Mutating decoded text or correctness metadata must leave this classifier unchanged.

## Registered phase and selection state machine

1. Run a new base K4 probe at 49,152/19.
2. If it is inadequate, preserve it as diagnostic-only and go directly to a new 61,440/15 probe.
3. If the 49k probe is adequate, run a new base K12 arm at 49k. If that arm is inadequate, advance
   to a new 61k probe; do not run or infer the missing designed arm.
4. If base K12 is adequate, run a new designed K12 arm at the same rung. If it is inadequate,
   advance to a new 61k probe.
5. Apply the identical state machine at 61k. The first rung with adequate K4, base K12, and designed
   K12 is the sole selectable rung. Stop immediately.
6. If the 61k probe or either 61k K12 arm is inadequate, stop with `pass:false`,
   `selected_thinking_budget:null`, and no catalog pointer.

Every lower rung must be present as an exact contiguous history and inadequate before a higher rung
can be selected. Completed lower rows remain diagnostic-only. No per-task escalation, arm carryover,
cross-rung pooling, sample backfill, or favorable-rung selection is allowed.

Before semantic analysis, the finalizer verifies every receipt in the selection history and then
recomputes every lower and selected termination metric from its receipt-bound token rows. A modified
adequacy bit or missing arm fails before decoded text is parsed or inspected and before hidden task
data is loaded.

## Semantic smoke gate

Only after a selected rung contains two receipt-valid, termination-adequate K12 arms may the analyzer
decode programs and load hidden examples. Within each task/arm, the deployable selector chooses the
earliest valid candidate among those with maximal visible-example score. The selected index is fixed
before any hidden example is graded.

The smoke gate passes only if:

- parse rate is at least 0.50 in base and in designed;
- designed produces a valid macro-using candidate on at least two distinct reuse tasks; and
- designed reuse oracle hidden coverage is no lower than base reuse oracle hidden coverage.

Oracle coverage is nondeployable and is used here only as an interface ceiling check. A passing
smoke licenses a future separately preregistered matched-compute experiment; it is not itself a
macro capability claim.

## Storage, crash safety, and inspection boundary

Artifacts live by default under
`/workspace/large_artifacts/qwen35_4b_verified_macro_capacity_fit_rerun/scientific_smoke_v1`.
The environment override must be absolute, nonsymlinked, and neither equal to, inside, nor containing
the predecessor root. A fixed lock outside the override namespace prevents two capacity-fit runs
from mutating alternate roots concurrently.

Preflight is checkpointed before generation. Rows and metadata are written atomically; the receipt
is the last commit marker. The tracked catalog inventories all external bytes. On resume, the only
legal catalog transition is the same preflight-only bundle becoming receipt-complete. An unknown
file, uncheckpointed bundle, mutated preflight, partial bundle, deleted completed entry, changed
receipt, or unexpected selected pointer fails closed.

Neither raw rows nor decoded output may be manually inspected to alter this protocol. Operational
errors may be repaired only without changing model-facing records, sampling, thresholds, phase
history, or registered capacity mapping. Any substantive change requires a new experiment.

## Registered interpretations

- Live capacity check fails: infrastructure mismatch; no model conclusion.
- 61k ends termination-inadequate: tested long-context protocol remains setup-inconclusive; no macro
  conclusion.
- Termination clears but semantic smoke fails: frozen free-form macro induction/interface ceiling
  under this protocol; do not run the full mechanism matrix.
- Designed oracle rises but selected performance does not: candidate selection bottleneck.
- Semantic smoke passes: capacity-fit vLLM made the scientific comparison runnable; proceed only in
  a separately frozen experiment with mined, hint, matched-random, designed, and matched-compute
  sample-more controls.

## Reproducibility gate

Before launch, `scripts/run.py --validate` and the complete CPU test suite must pass. GPU launch also
requires coordination confirming that no other process owns the device or fixed experiment lock.

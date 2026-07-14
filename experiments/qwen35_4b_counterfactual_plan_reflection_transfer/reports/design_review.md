# Adversarial Design Review

## Review 1 — 2026-07-14

**Reviewed commit:** `3eae868d182f4a02848f6415d8eaafdb87465336`

**Verdict:** **HOLD**

**Access:** zero tokenizer/model/GPU/benchmark/hidden/protected events; no review edits

The independent adversary reproduced the committed smoke and then tested the proposed
geometry. It found the original corpus impossible: the string family had only 122
eligible signatures for 192 required rows, and an uncaught state-explosion exception
terminated full construction. It also exhaustively checked the visible examples and
found that 14/30 smoke tasks admitted two to nine depth-three plan spellings.

Additional blockers were:

1. correct-versus-shuffled isolates task-specific auxiliary plan supervision, not a
   reflection-specific effect; a matched non-reflective plan-label arm is required;
2. the fixed `READY` seam is controlled branch transfer, not a literal interrupted
   action state, so claims must be scoped accordingly;
3. direct-control, retention, literal-reflection, and candidate-compute contracts were
   missing or contradictory;
4. tokenizer rendering, exact loss masks, LoRA/optimizer/batching parity, target-token
   accounting, and immutable receipts were not frozen;
5. qualification/confirmation power and seed-independent gates were not executable;
6. the conditional J stage lacked independent fit/confirmation/causal data and fixed
   readout/intervention controls, and therefore must not share behavioral evidence.

### Repair checkpoint status

The following defects are repaired in the working design and covered by executable CPU
tests:

- a finite exact-depth catalog is enumerated before allocation;
- globally behavior-equivalent and shallower-equivalent programs are excluded;
- seven visible demonstrations uniquely identify the exact ordered plan;
- full 216/72/144/144 construction succeeds with zero program or behavior collisions
  and complete per-split operation-position coverage;
- state explosions are rejected, not leaked;
- shuffled donor labels occupy separate supervision fields, preserve task truth, and
  are behaviorally wrong for the recipient;
- validation uses fail-closed exceptions rather than optimization-removable asserts;
- the construction entry point consumes the committed config and installs a Python
  audit-hook benchmark read firewall.

The verdict remains HOLD. No tokenizer/model/GPU/training/Jacobian stage may run until
the remaining design contracts are implemented, committed, and independently reviewed.

### Post-review implementation awaiting Review 2

The working repair adds the previously missing non-reflective correct-label arm,
real retention tasks, exact record and mask code, a frozen QLoRA recipe, within-step
token parity, executable paired gates, explicit literal-reflection accounting, and
two-seed confirmation rules. The J stage has been moved to a required separate future
experiment. These changes are not self-authorizing; Review 2 must inspect their exact
committed revision.

## Review 2 — 2026-07-14

**Reviewed commit:** `1cb3c351b1ca14b518abe7cbff02ac67e6134726`

**Verdict:** **HOLD** for training/evaluation; tokenizer implementation statically
safe, but a separate clean authorization signature is required.

The adversary reproduced full 216/72/144/144 plus 48 retention construction, verified
all 36 optimizer groups and 216 shuffled donors, passed the focused model-free tests,
and found the rendering/mask/trainer logic coherent and fail-closed. It confirmed the
J follow-up is genuinely result-separated.

It then demonstrated that the decision functions could pass a one-task, one-family,
one-depth artifact because they did not require the sealed task-ID sets. It also found
that generation metadata, merged trees, stage ancestry, retention promotion,
literal-reflection execution, adapter ON/OFF behavior, and adapter-tree lineage were
not mechanically enforced. These are blocking provenance defects even though the
statistical design itself is now specified.

During review, one overly broad repository search surfaced unrelated tracked run-log
lines containing hidden-output fields despite the benchmark exclusion. The reviewer
made zero tokenizer/model/GPU/benchmark calls and no edits, but this access mistake
means its tokenizer-safe assessment is not accepted as the required clean independent
authorization.

### Review 2 remediation implemented, pending independent review

- exact sealed task IDs, all three families, and both retention depths are now required;
- scoring binds input/label receipts, vLLM metadata, model/revision, runner, environment
  lock, clean Git state, sampling, seed, engine geometry, merged checkpoint, arm, and
  adapter seed;
- merge and vLLM load bind full artifact trees and exact source arm/seed lineage;
- screen, replication, confirmation, and final promotion use hash-bound stage receipts;
- retention has an executable promotion gate;
- literal reflection now has an exact branch constructor and token-matched base-prefix
  scorer;
- every merged adapter must pass a greedy token/logprob ON/OFF gate before scoring.
- runner metadata now hashes the generated JSONL itself, and every scoring path checks
  that binding before accepting rows.

The repaired model-free suite passes 45 focused tests and reconstructs all 624 sealed
depth-three plus retention tasks with no collision. No authorization changes occur
until these repairs are committed and a fresh clean reviewer returns its verdict.

## Review 3 — 2026-07-14

**Reviewed commit:** `492376af67fd03e8b75210b8bb42ebb297fdbeed`

**Verdict:** **PASS_TOKENIZER_ONLY; HOLD full implementation.**

This was the required clean review: it read only the allowlisted experiment/shared
implementation files, used temporary synthetic fixtures, and made zero tokenizer,
model, GPU, benchmark, protected-output, or web calls. Both exact-commit CI workflows
were also green. It independently passed all 45 focused tests, syntax, and full
construction.

The tokenizer path is authorized because it pins the sole permitted model/revision,
checks EOS 248046, reconstructs sealed records, forbids truncation, validates exact
mask/loss boundaries and optimizer-step parity, and exclusively writes its receipt.
This verdict does not authorize any model, GPU, training, evaluation, J-space, or
benchmark event.

The adversary reproduced eight full-implementation blockers:

1. scoring trusts self-issued prompt/label receipts instead of reconstructing exact
   sealed prompt bytes, label bytes, and per-task family/depth mappings;
2. primary scoring does not compare the complete sampling and resolved-sampling
   dictionaries, so unregistered penalties and custom-prompt flags can pass;
3. family and retention-depth gates check sets rather than the exact per-task mapping
   and balanced sealed counts;
4. the literal branch lacks a dedicated sealed reflection-input constructor;
5. consumers do not fully validate stage-receipt schema/cardinality/ancestry, and
   confirmation generation is not mechanically stage-gated;
6. merge/runner lineage omits parts of the source training receipt, PEFT recipe,
   trainer/Git identity, and prerequisite receipt chain;
7. adapter ON/OFF does not reconstruct exact sealed calibration mappings, and runtime
   parity does not prove installed-lock or exact cross-arm environment identity;
8. the runner lacks the live KV-capacity/preemption preflight required by the pinned
   vLLM operating contract.

Review 3 demonstrated false capability, positive-control, and reflection-specific
passes using substituted constant labels plus internally consistent synthetic
receipts. It also demonstrated qualification with family counts 142/1/1 and retention
with depth counts 1/47 and family counts 46/1/1. These are decisive HOLD findings.
They must be remediated and independently reviewed before any full execution.

### Review 3 remediation implemented, pending Review 4

- `src/eval_inputs.py` is now the single canonical builder for action prompts, exact
  oracle labels, literal-reflection prompts, literal-action branches, receipts, and
  task metadata. Scorers reconstruct these bytes and reject self-consistent forged
  receipts; a regression test reproduces the constant-label attack and fails closed.
- Primary, adapter-gate, and literal paths compare every `SamplingConfig` field plus
  the complete resolved sampling dictionary, including penalties, custom-prompt flags,
  logprob settings, and thinking shuffling.
- Capability, positive-control, calibration, and retention gates require the exact
  per-task family/depth mapping. Every compared row must share one runtime-protocol
  hash, closing the 142/1/1 family and 1/47 depth attacks.
- A dedicated sealed literal-reflection constructor is distinct from the action input.
  Literal action prompts and their receipt are deterministically reconstructed from
  the exact reflection generation.
- Stage receipts have an exact schema, prerequisite claim cardinality, both frozen
  seeds, source hashes, issuer Git/script identity, and clean-worktree requirement.
  The runner requires and validates a stage receipt before every non-smoke model load,
  including confirmation generation.
- Training copies its tokenizer and stage receipts, records trainer Git/code, recipe,
  record, parity, and full adapter-tree identities. Merge validates the complete PEFT
  recipe and embeds the source training/stage/tokenizer/adapter-config files in the
  hashed merged tree; the runner revalidates them at load time.
- Runtime validation checks every locked distribution version and exact cross-arm
  package/GPU/Git/engine/cache identity.
- The runner authenticates pinned Qwen hybrid-cache geometry, then requires both the
  rounded live KV-token inequality and conservative `active_sequences * 11` block
  inequality before generation. The complete terms and margins enter metadata.

The model-free suite now passes 55 focused tests plus syntax and full construction.
All model/GPU/training/evaluation/J/benchmark flags remain false. These changes are
not self-authorizing and require a fresh clean adversarial verdict on their committed
revision.

## Review 4 — 2026-07-14

**Reviewed commit:** `542ba82592d96eafcf56cd5e70bfad948b43b65b`

**Verdict:** **HOLD full implementation.**

Both exact-commit CI workflows were green. The clean adversary passed all 55 focused
model-free tests, syntax, full 216/72/144/144 plus 48 construction, and separate
token- and block-overcommit attacks. It confirmed that Review 3 findings 1–4 and 8
are genuinely closed.

Three high-severity false-acceptance paths remain:

1. stage consumers accept prerequisite claims containing arbitrary or nonexistent
   SHA-256 strings, while the authorizer accepts minimal hand-written decision and
   retention JSON without authentic score ancestry or analyzer identity;
2. the committed merged-checkpoint test accepts `weights.safetensors` containing only
   the bytes `weights` plus self-consistent fabricated lineage, so tree integrity does
   not establish scientific origin or arm/seed identity;
3. installed-package validation parses only `name==version` lock entries and silently
   omits the direct-URL `vllm` requirement, allowing a forged vLLM build while checking
   the other 188 distributions.

No model, GPU, training, evaluation, J-space, or benchmark work is authorized. The
next implementation checkpoint must make the exact attacks fail by resolving and
revalidating prerequisite artifacts, retaining and authenticating the actual source
adapter tensors/tree through merge and consumption, validating a real composite
checkpoint inventory, and enforcing the normalized direct-URL vLLM build pin.

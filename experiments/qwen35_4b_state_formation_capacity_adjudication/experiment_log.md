# State-Formation Capacity Adjudication Experiment Log

## 2026-07-13 — experiment created

- Created a fresh successor under `structured_execution_and_compilers`; neither prior result-bearing
  directory will be extended or overwritten.
- Status is `in-progress`. No model load, GPU mechanics call, procedural result corpus, training,
  evaluation, or scientific analysis has run in this directory.
- The triggering uncertainty is the post-result audit of
  `qwen35_4b_state_carry_vs_state_bag_fullrank_delta`: the raw full-rank state miss remains valid, but
  it did not isolate LoRA capacity because shared state-module initialization and dropout streams
  differed across experiments and non-capacity promotion gates failed simultaneously.

## 2026-07-13 — design correction: capacity-only verdict axis

- Removed Bag advantage, answer-gain, query-stratum, edge-cut, swap, and sample-more requirements from
  the capacity verdict. They address utilization or serial mechanism, not whether the representation
  formed.
- Retained the same Carry loop, procedural state targets, state-before-query placement, exact K=1
  path, and joint node+phase+checksum readout.
- Froze fresh procedural splits and three fixed-final 1,500-step seeds 7411–7413. Prior rows and
  checkpoints are context only, never training data or result cells.

## 2026-07-13 — design correction: joint-first with conditional controls

- Rejected a state-only-only adjudication because it would change the original joint objective and
  could not answer whether LoRA rank caused that failure.
- Froze the sequential branch: three LoRA joint runs first; after a valid LoRA formation miss, three
  LoRA state-only controls plus three full-rank joint runs become mandatory; only a full-rank joint
  miss opens the three full-rank state-only controls.
- The maximum is 12 result-bearing runs, but controls and dense runs are prohibited until their
  identity-bound branch receipt exists.
- Joint retains answer/state/fixed-point weights `1.0/0.5/0.05`. State-only omits answer loss rather
  than multiplying it by zero.

## 2026-07-13 — cross-capacity confound controls

- Froze one serialized common loop-state initialization bundle per seed. Every capacity/objective
  arm must reload it strictly and prove bit-identical names, shapes, dtypes, and tensor values before
  training.
- Replaced the divergent PEFT-versus-direct dropout paths with a common custom adaptation-hook
  backend. Both capacities use dropout `0.05` and the same capacity/objective-independent
  per-microbatch dropout schedule.
- Froze separate gradient clipping for shared loop-state parameters and adaptation parameters; dense
  adaptation gradients cannot change the clipping scale applied to shared modules.
- Added intact and adaptation-disabled state evaluation for every checkpoint. A capacity rescue must
  show a registered positive intact-minus-disabled effect.
- Setup-only overfit/readout controls must pass before result training. Failure is a repair stop, not
  a capacity result.

## 2026-07-13 — terminal-metric and conditional-contrast hardening

- Replaced pooled trajectory accuracy with exact terminal joint node+phase+checksum correctness at
  K equal to semantic depth. The `0.40` gate applies separately to every seed×depth cell; validation
  depth 1 is reported but excluded because K=1 bypasses adaptation.
- Initially froze 1,024 rows for each trigger/deep result evaluation split. Validation has 256 rows at
  each depth 1–4; depth extrapolation and joint family+surface shift have 128 rows at each depth 5–12.
  The later adversarial repair below adds the distinct 768-row sealed trained-depth split.
- Reserved seeds 73305 and 73306 as sealed 1,024-row post-trigger depth and joint-shift contrasts.
  They remain unscored until a valid LoRA miss is followed by all Stage-B fixed finals and trigger
  evaluations, including LoRA state-only, and a dedicated identity-bound seal receipt proves no prior
  contrast scoring access. If direct full shape misses any trigger cell, the seal mandates state-only
  and leaves those rows unopened. The LoRA-miss receipt licenses training but cannot license contrast
  evaluation, preventing conditional branch selection from contaminating the rescue comparison.
- Added deterministic-tensor output/gradient parity against pinned PEFT `lora.layer.Linear` before
  the common LoRA hook can be treated as compatible with the parent PEFT recipe. This does not load a
  second Qwen model.
- Named the affirmative conditional result `DIRECT_FULLSHAPE_RECIPE_RESCUE`; it is explicitly not
  mathematical rank identification.
- Replaced the arithmetically impossible 32-row “balanced” control with 48 setup-only rows: exactly
  two per depth 2/3/4 × query kind × training family × training template cell, seed 73991.
- Froze the exact `design-boundary` interface and canonical `reports/design_receipt.json`; it binds
  the intake, scientific contract/review, architecture/runbook/handoff, and config. Source/tests,
  implementation review, and the training lock must be clean at its recorded HEAD and are bound
  separately by downstream artifacts, so a mechanical repair can preserve the scientific boundary
  while invalidating all artifacts produced under the old implementation digest.
- Required tiered sealed-data validation: preauthorization stages may verify compressed identity but
  may not decompress or canonical-reopen contrast payloads.

## 2026-07-13 — adversarial repair: complete sealed-domain replication

- Found that the two original deep sealed splits could not retest a LoRA branch triggered solely by a
  trained-depth validation miss. Added `contrast_validation` at seed 73307 with 768 fresh rows over
  required depths 2–4 (256/depth). The complete sealed matrix now covers trained depth, depth
  extrapolation, and joint shift.
- Retained exactly six authorized capacity×seed contrast-evaluation jobs: each job evaluates all three
  sealed splits in intact and disabled modes. A sealed LoRA intact pass has terminal priority and
  emits `LORA_TRIGGER_MISS_NOT_REPLICATED_ON_SEALED_CONTRAST` regardless of full-rank score.
- Required category-matched replication before rescue: every LoRA `trained`, `depth`, or `joint`
  category that failed on trigger must fail again in its corresponding sealed domain. Additional
  sealed failures are allowed; a missing counterpart emits
  `LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_ON_SEALED_CONTRAST` once full rank passes its
  absolute cells. Full-rank absolute failure still takes priority and mandates state-only.
- Made sealed-evaluation recovery executable rather than advisory. Each same-cell retry requires
  exactly one newly tracked, content-validated `FAILED_ATTEMPT_ARCHIVED` receipt appended to the
  existing access-ledger event; completed evaluations are ineligible for failure archival. Ledger
  revisions now hold a separate stable lock inode across atomic replacement, with the temporary file
  and parent directory fsynced, and initial access rejects an archive that predates its event.
- Hardened optimizer receipts so adaptation and common-state learning rates are logged separately and
  independently recomputed from the registered schedule at every exact step.
- Required post-contrast analysis to reopen the Stage-B-bound full-rank-joint, LoRA-joint, and
  LoRA-state-only trigger evidence and recompute their formation summaries. Stage C separately
  reopens the exact Stage-B LoRA-state-only and full-rank-joint predecessors before classification.
- Expanded the PEFT compatibility gate to both exact FP32/dropout-off/autocast-off and live-like
  bf16-autocast/dropout-0.05 output/A/B-gradient parity, with respective tolerances
  `atol=1e-6`/`rtol=1e-5` and `atol=2e-3`/`rtol=1e-2`, plus a same-device-seed realized-mask protocol.
- Named `ADAPTATION_DISABLED_REVERSAL` for intact formation miss plus disabled formation pass. It is a
  reported adapter-interference diagnostic, never an adaptation-required or rescue result; both LoRA
  and full-rank sealed statuses remain visible.
- Recorded `posttraining_and_adaptation` as a secondary program fit because the experiment adjudicates
  update parameterization, objective interaction, and adapter dependence. Recurrent state formation
  remains the sole primary estimand under `structured_execution_and_compilers`.

## 2026-07-13 — implementation audit and CLI repair

- The first real CPU-smoke invocation found a pre-result false rejection: an empty authorization
  allowlist rejected the required omitted receipt. Implementation status was immediately returned to
  `NO_GO`; no model, result corpus, training, evaluation, or scientific analysis had run.
- Repaired authorization semantics so every preauthorization stage and LoRA-joint path accepts only
  the required omitted receipt, while all conditional paths require their exact canonical branch
  receipt. Added an early rejection for the unregistered state-only contrast cross-product.
- Hardened output construction and operator safety: missing analysis phases, state-only contrast,
  irrelevant phase/evaluation axes, conflicting `--smoke` stage selection, junk authorizations, and
  noncanonical outputs all fail before dispatch.
- The final implementation passes 130/130 experiment tests, including 23/23 static/CLI tests and
  14/14 focused sealed-data/replay tests. An independent exhaustive audit verified all 23 normalized
  CLI cells and their canonical outputs. The real CPU smoke then passed with zero benchmark reads,
  no GPU/model load, and `scientific_evidence: false`.
- These changes harden execution mechanics only; they do not change the preregistered scientific
  design. Downstream source-bound artifacts must use the final post-repair source-contract digest.

## 2026-07-13 — clean-head freeze and setup artifacts

- Froze `reports/design_receipt.json` at clean committed head `86230605` with exact status
  `DESIGN_FROZEN`, zero benchmark reads, design identity
  `d943b909250c2ebd377b8094bb55324a5d5ccf555c7559736526f013d248ac52`, source-contract digest
  `903f19141ef982f0e90ea856edd72d75fb48d2dbd96e81dc166a6c32d4c14116`, and training-lock digest
  `05546fe977583116d6169ea0dfa7b27e1184dd4a2b61d556dfb3f889d5b2b7b1`.
- Generated the seven fresh deterministic splits. The tracked manifest records 12,000 training rows,
  1,024 trigger rows in each domain, 768 sealed trained-depth rows, 1,024 sealed depth rows, 1,024
  sealed joint-shift rows, zero cross-split structural duplicates, zero benchmark reads, and data
  contract `2eb5c134fb0818aeadb1cd9ac0074554cc26b5c010c248316c6e1eeb56f02291`.
- Left all three sealed contrast payloads unopened after generation. Their empty access ledger has
  identity `238c0ace4e4cc929657de28130e0c2775f31faa2f096027e25ab8bd635ae922d`.
- Created and independently reopened the three canonical common-state bundles. Seeds 7411, 7412, and
  7413 have bundle SHA-256 values `2d3d1f11c78d561a4e623e0512326ad20560e10aaa192276611250b1d3b73497`,
  `c236d1f936b570447746a561286da643fe958a705aa26e17bd5fe0bbb7299c9e`, and
  `c4a20778d5ff803faff944bd3735a92869f132982bef2b51a5e5c0ecbbee5448`, respectively; each tracked
  mirror is byte-identical to its external sidecar.
- Live preflight found exactly one idle RTX 6000 Ada, 49,138 MiB total minus 2 MiB in use, no GPU
  process, a compatible 76-package environment, and both required causal-convolution and
  flash-linear-attention kernels. All 130 experiment tests passed again. No model was loaded.

## 2026-07-13 — first live G0 stopped on revision-provenance defect

- The first seed-7411 LoRA G0 loaded only the requested `Qwen/Qwen3.5-4B` snapshot at pinned revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, then stopped inside `_load_base` before wrapper
  construction or any mechanics probe. The training split had passed its permitted pre-G0 validator;
  no sealed contrast payload was opened, no positive control or training/evaluation step ran, and no
  canonical G0 output was created.
- Root cause: Transformers 5.13.0 resolves `config.json` through the exact pinned snapshot and its
  cache utility extracts the correct 40-character commit, but the Qwen3.5 causal-LM wrapper retains a
  derived text config whose runtime `_commit_hash` is `None`. The runner incorrectly treated that
  lossy field as the sole revision proof.
- Preserved the exact setup failure at
  `runs/failures/g0_lora_seed7411_revision_provenance_failure.json`. This is mechanics evidence only,
  never scientific evidence and never branch authorization.
- The fail-closed repair verifies the resolved commit and records the byte count and SHA-256 for
  config, tokenizer assets, safetensors index, and every indexed shard before model construction. A missing runtime
  config hash is accepted only after that complete proof; any non-null mismatch remains fatal. The
  real local snapshot currently resolves all nine files through the pinned commit.
- Because the repair changes the registered source contract, every data, initialization, and setup
  artifact produced under digest `903f19141ef982f0e90ea856edd72d75fb48d2dbd96e81dc166a6c32d4c14116`
  is invalidated and must be regenerated before retry. The frozen scientific design receipt remains
  unchanged.
- Preserved all 19 invalidated files (17,651,037 bytes) under the external `invalidated_setup/`
  archive. The tracked archive receipt reopens every byte count and SHA-256 and has identity
  `e29f2059058b621903c1bbf2933dd56aa0e500a0c73b70ffc8b19f72faa85014`.

## 2026-07-13 — corrected-source setup regenerated

- Recreated CPU smoke, all seven data splits, the contrast-access ledger, and all three initialization
  bundles from committed source contract
  `9fd420f5f29fea2d9144bf50d3b187fc8e50d9acc9cb076656372281029614fb`. CPU smoke again records zero
  benchmark reads, no model load, and `scientific_evidence: false`.
- Every procedural payload reproduced the exact prior content hash; the new source-bound manifest has
  SHA-256 `fa871390b28a9a0119ad77957bc1b403cc7008aa7052a539e6788aa8d871be4c` and data contract
  `a57c05100ff51897dbfdc2e140b1eff6634513eaa873a222e2fd2d7f9b37ad2e`. The new sealed-access ledger
  has no events and identity `14748750079830ea25e7935baf35a4689957b3bfb9db0b7630f643133bbfde8d`.
- Seeds 7411, 7412, and 7413 preserve the exact common tensor-value digests while their new source-bound
  bundle hashes are `fbb40b74d97fd07dd0e6590382f99834d5091e1467df1f3edda814092f386c77`,
  `b05cbafcf9a100dd80309a5b50309649372fe6841e32cdd38d37b15bdcaba6ed`, and
  `cd9923902e25f04ba5aca0ba001f036aa5ac6e8ae6792ac059d78773f22b1bf3`, respectively. Every bundle
  reopened and each tracked receipt is byte-identical to its external sidecar.

## 2026-07-13 — G0 reached final persistence, then path shadowing stopped it

- The corrected seed-7411 retry passed pinned-snapshot proof and every registered in-memory G0 check,
  including PEFT output/gradient parity, K=1 parity, K=4/K=12 call geometry, two-step and live joint
  gradients, optimizer/clipping receipts, timing, finite worst-depth output, checkpoint roundtrip,
  RNG isolation, and VRAM headroom.
- The final `_write_json(output, receipt)` then received the last `StateLoopOutput` tensor container
  rather than the canonical `Path`: the two-step probe had reused the parameter name `output` as a
  local. Python function scoping retained that shadow through the final write. No canonical G0 receipt
  was created, so the in-memory pass is not durable evidence and authorizes neither positive control
  nor result training.
- Preserved the exact failure at
  `runs/failures/g0_lora_seed7411_receipt_path_shadow_failure.json`. The training payload was the only
  result payload opened; no sealed contrast was opened and no positive-control/training/evaluation
  stage ran.
- Renamed the local to `probe_output` and added an AST regression prohibiting any assignment to the
  canonical `output` parameter inside `model_smoke`. Audited all runtime sources: this was the only
  G0/path-parameter shadow. Also detached the PEFT diagnostic before scalar conversion to remove its
  non-semantic autograd warning.
- Archived every setup artifact bound to source contract `9fd420f5…614fb` at
  `large_artifacts/qwen35_4b_state_formation_capacity_adjudication/invalidated_setup/source_9fd420f5f29fea2d9144bf50d3b187fc8e50d9acc9cb076656372281029614fb/`.
  The tracked archive receipt has identity
  `7f974f52e153eb2fe7a3985c8221cfa8f7cd2d8440eb63d608abf03a63651d47` and independently reopens all
  19 files totaling 17,651,037 bytes.
- An independent audit found that all four hand-authored historical receipts had included a trailing
  newline in their claimed self-hash, unlike the runtime's compact canonical-JSON identity. Corrected
  the four identity fields without changing their evidence payloads and added a regression that
  revalidates every tracked `runs/failures/*.json` receipt against the runtime identity function.
- Final independent re-review is `GO`: the full suite passes 135/135, the static contract passes
  27/27, both 19-file archives reopen byte-for-byte, and the post-review source contract is
  `3baa7b532d62bae4d9751dfe4be9c6ce314c11ca4524266f4cebea63289d5c42`.

## 2026-07-13 — post-path-fix setup regenerated

- Recreated CPU smoke, all seven procedural splits, the seal ledger, and all three common-state
  bundles from committed source contract
  `3baa7b532d62bae4d9751dfe4be9c6ce314c11ca4524266f4cebea63289d5c42`. CPU smoke SHA-256 is
  `8858671b077dd609141eb218b6932e9b8d7772877c2308df57c59d3d066ae467`; it records zero benchmark
  reads, no model load, and no scientific evidence.
- The new source-bound data manifest has SHA-256
  `c2338a7aa2dad245683b6d3aebeef704199e9f143490830179263ea1fed57247` and data contract
  `5ff3255bebe18a4a73a3bd9d1db4c153e4727aa23e0e064cfd5ed3d685307968`. Its exact seven-file
  projection reproduces the prior payload metadata at digest
  `3d52d6b31ffbd916cdd86495e0c14f0d11c293f868914e667d10a05acfecfba2`. Independent local reopen
  validated all 15,072 non-contrast rows and decompressed zero sealed rows.
- The new seal ledger has file SHA-256
  `e2922cb1e51f58d378a5bc4fd6d49de391ea10f04fe06ac2c576424d91e2a85a`, canonical identity
  `45f58325fbf7f362265bb6812cfeb23fc974bca5b7c25eea61d229d211446523`, and `events: []`.
- Seeds 7411, 7412, and 7413 have bundle SHA-256 values
  `9ffbb459238ed42cfeef541c0b331b744318a1940a55c5a6a8bca133866774f4`,
  `9d666538e98de6c2660d42221286c89de36fdd786d5223df4b932bf8a6425670`, and
  `6ce43dffdc7c47dc8e8d7370e42a394bcb25d35de063bfa04ab9cea3983d4d1b`. Their receipt identities are
  `136565d109b7b936452d9d91a4a64de565157320aa5b965e49030da05e253256`,
  `361274c81e734c651897f5c8b69380f6dd1f2ef4906aa7c6d8191bb8a148572b`, and
  `85699c6008c83704453acfaff8aee1de68e977eaf0fcb552151c04f206b0273d`; every external sidecar is
  byte-identical to its tracked mirror and all tensor-value digests reproduce exactly.
- Independent audit is `GO`: it independently validated the source contract, manifest geometry,
  compressed-byte hashes, empty-ledger identity, all three bundle files and receipt mirrors, 15,072
  non-contrast rows, and zero sealed row decompressions.

## 2026-07-13 — seed-7411 LoRA G0 passes durably

- The fresh retry wrote canonical receipt `runs/setup/g0_lora_seed7411.json`, file SHA-256
  `8495799beb226644f8c88b18f26510c7c4cfaed4117fb81d54b1cc94d4efac66`, runtime identity
  `e7394bcf48175c8232ac825a4f74f16734452602b9d8657be09ca47e293f3735`, and exact status
  `MODEL_SMOKE_PASS`. The runtime's canonical loader reopens it against source contract
  `3baa7b53…d5c42`, data manifest `c2338a7a…247`, design receipt `d943b909…ac52`, and initialization
  receipt `136565d1…3256`.
- The pinned snapshot proof covers nine exact files at revision `851bf6e8…cd0a`, 9,342,815,919 bytes,
  aggregate file-manifest digest `06486f260c8e9135835c768802e0e947bab0470e4454a36e138da6d10ed1fe12`.
  The target manifest has 62 ordered modules and the LoRA bank has exactly 16,232,448 parameters.
- FP32/dropout-off and bf16/dropout-0.05 PEFT output/A-gradient/B-gradient parity each pass at zero
  observed error. Initial enabled-versus-disabled and both pre/post-update K=1 errors are zero. The
  second state-only step and the live joint probe give finite nonzero gradients to every required
  recurrent group and all 124 LoRA tensors, with zero base-model gradient tensors. All required Adam
  states are complete and finite.
- The ten-step probe is finite at 0.309 seconds/update. K=12 is finite with exactly 682 ordered calls;
  the setup row has zero structural overlap with result data. Destructive checkpoint restoration is
  exact, peak allocated/reserved memory is 11.20/11.36 GiB, and 35.63 GiB remains free after G0.
- This is setup evidence only. The model stage opened only the permitted training payload; the seal
  ledger remains byte-identical at `e2922cb1…a85a`, identity `45f58325…6523`, with `events: []`.
- Independent receipt audit is `GO`: the runtime identity, all lineages, mechanics, environment,
  optimizer, restore, and firewall claims independently reopen with no blocker.

## 2026-07-13 — seed-7411 positive control stops at 0/48

- The fixed 256-update LoRA positive control reached its final gate, then raised
  `tiny state-path overfit failed: 0.0 < 0.95`. The oracle component necessarily passed its 0.99 gate
  because the run continued to the overfit check. No canonical positive-control receipt was created.
- Preserved the exact setup failure at
  `runs/failures/positive_control_lora_seed7411_overfit_failure.json`, identity
  `44397a2e278293bf54fe5d172ac4294c565a2a98ab7c8f4faaeb5ee044e8ec7c`. It binds the G0, source,
  design, data, model, seed, 48 rows, 256 completed updates, and exact exception. It records no result
  payload read, no result training, and no sealed access.
- Independent shape/target tracing found no scorer-index defect. State targets and logits both have
  aligned `[batch, step]` geometry, and terminal joint scoring compares all three heads at the same
  final semantic step. Near chance, exact joint accuracy is 1/256, so zero of 48 is not surprising.
- The leading mechanics defect is an underpowered update path: the positive control performs one
  single-row microbatch per optimizer update instead of honoring the configured accumulation of 16.
  Across 48 rows, the failed recipe exposed each row only five or six times. It also persisted no
  losses, per-head metrics, gradient norms, optimizer state, or failure receipt before raising.

## 2026-07-13 — positive-control accumulation correction approved

- Three independent failure reviews found no target, tensor-shape, terminal-index, recurrence,
  gradient, or scorer defect. The failed setup control made one singleton presentation per optimizer
  update and omitted the globally frozen accumulation of 16. It therefore made 256 presentations,
  only five or six per high-entropy row, while preserving 256 optimizer updates.
- The frozen-boundary ruling permits exactly one narrow correction: the same 48 rows, canonical row
  hash, seed 73991, 256 optimizer updates, state-only objective, learning rate `2e-4`, weight decay
  zero, dropout 0.05, clip thresholds, oracle threshold 0.99, final threshold 0.95, initialization,
  row order, and fixed-final decision now use 16 sequential singleton microbatches per optimizer
  update. Loss is divided by 16; each group is clipped once; the optimizer steps once; no early stop,
  selection, or second retune is allowed.
- The exact 4,096-event schedule gives the first 16 rows 86 exposures, the remaining 32 rows 85, and
  depth totals 1,368/1,368/1,360. Global indices feed the unchanged dropout-seed preimage. Fixed
  diagnostics at updates 0, 1, 16, 64, 128, and 256 score intact and adaptation-disabled modes while
  proving no parameter, train/eval-mode, or CPU/CUDA RNG mutation.
- Any reached exception now writes canonical and byte-identical tracked `SETUP_CONTROL_FAILED`
  receipts before re-raising. The receipts deny training and record zero benchmark, result-payload,
  and sealed-contrast access. Downstream training continues to accept only
  `POSITIVE_CONTROL_PASS`.
- The source-bound suite at the archive boundary passed 154/154 tests: 13 focused positive-control tests, eight durable
  invalidation tests, and 27 static/CLI/provenance tests. Independent code and GPU/runtime audits both
  give `GO`. This changes setup mechanics only; the frozen scientific design is unchanged.

## 2026-07-13 — source-3baa setup archived after control correction

- Before deletion, the complete source-`3baa7b53…d5c42` setup reopened as exactly 20 files totaling
  17,775,495 bytes with file-manifest identity
  `115b9dd5e810cb1bcf88e58a4da366370cbb48c4a1fc1adfcdcb37a5283e1a25` and an empty seal ledger.
- The durable archive is
  `large_artifacts/qwen35_4b_state_formation_capacity_adjudication/invalidated_setup/source_3baa7b532d62bae4d9751dfe4be9c6ce314c11ca4524266f4cebea63289d5c42/`.
  Its tracked mirror is `runs/failures/invalidated_setup_source_3baa7b53.json`, with receipt identity
  `1daa86e02f7d3f3c612c2f1ec01db89e4967b5b9ce7eb19ba75c752fe1e283aa`. Both receipts and every
  archived byte were independently rehashed before current setup files were removed.
- The receipt binds invalidated source `3baa7b53…d5c42` to replacement source
  `fe82f8cfe13d656a588c8f2766fc5519820edd884c9b855dcf7a16a14af547d5` and to the preserved 0/48
  failure identity `44397a2e…e8ec7c`. That failure remains outside the archive and unchanged.
- Current CPU smoke, data manifest, access ledger, initialization bundles, and G0 receipt are absent
  by design. No sealed payload was decompressed during archival; the archived ledger has `events: []`.
  No positive-control pass or result artifact exists.

## 2026-07-13 — post-archive directory cleanup hardened

- The first repository gate after archival stopped because successful deletion left empty
  `runs/cpu_smoke/` and `runs/setup/` directories on the local filesystem. Git cannot transport empty
  directories, so CI would have observed a different tree. The archive itself and both receipts
  remained exact.
- Adversarial follow-up reproduced partial deletion that could not resume and symlinked canonical
  roots that could redirect cleanup. The final helper uses the immutable archive receipt as its
  journal, verifies archive-only recovery before recreating the byte-identical tracked receipt,
  accepts every live source only as exact or absent, resumes unlink/`rmdir`/fsync interruption, and
  rejects unknown or tampered state, receipt/archive mismatch, special entries, and symlinks in every
  canonical root, ancestor, or evidence leaf.
- Twenty-five invalidation/recovery tests exercise those states. The full source-bound suite passes
  171/171, both independent recovery re-audits give `GO`, and a real completed replay leaves the
  20-file archive plus both receipt bytes and mtimes unchanged.
- No setup artifact was created under either intermediate post-archive source. The final source
  contract for regeneration is
  `1d1368cf064689322d9df7f345e67b026cecccc32d3a7b7514b82f253d434b0a`.

## 2026-07-13 — final-source setup regenerated

- Recreated CPU smoke, all seven deterministic procedural splits, the contrast-access ledger, and
  all three common-state bundles under final source contract
  `1d1368cf064689322d9df7f345e67b026cecccc32d3a7b7514b82f253d434b0a`. The CPU receipt has SHA-256
  `56032f75f8c3be863b43d1d6fc46a27199c8162961bfbe49b146d41e0ac7ad43`, records
  `CPU_SMOKE_PASS`, zero benchmark reads, no model load, and no scientific evidence.
- The manifest has SHA-256 `85286a9549b0e74111eaa0c1254f04f68cad4d75159abc8ba59c169637280cd9`
  and data contract `891ad7848ba23429ed5c7d66f2d6f4f85e8cc6560044b56f51f218bff957e9c8`.
  All seven compressed payload hashes reproduce the archived source-3baa data exactly. Validation
  reopened all 15,072 non-contrast rows, found zero cross-split structural duplicates, and
  decompressed zero sealed rows.
- The contrast ledger has file SHA-256
  `0c03a0d8e0b665415bf956856296a2750ce1fe7facd2335d6a711f96f5aac261`, identity
  `b122d49054cd37474784a8b4ec3bcc0026e745321718dee799615e3051983c14`, and `events: []`.
- Seeds 7411, 7412, and 7413 have bundle SHA-256 values
  `64f6ac313ee46335c4e8869ac3333136b4fe1739c8c59954b81f971fb96a2971`,
  `ef662c4a3f184b5d18a88142362a9803b51627e2bb59b4d0180afb331f5820ae`, and
  `29f35ec72113039aed8c278a9166c5a70e0c9c575b88d3c077d62dacc3ce8f9a`; receipt identities are
  `bbb21fef784beb1bf9dc50de27ff41624a85904c39191f2288e94cb7a8ad10ae`,
  `a5d1a3fdc574ac4968709c65c49b59e859b03ab4226d847dde40fdbdc98ce3f7`, and
  `42f9f66b848bcadd6e320db5421d2dfe39572793511b2b38151f6163a92e19e7`.
  The canonical loader independently reopened each bundle, every tracked receipt is byte-identical
  to its external sidecar, and the three tensor-value digests exactly match the archived source-3baa
  bundles.
- No model-bearing stage ran during regeneration. No result training is licensed, and the sealed
  ledger remained unchanged throughout initialization validation.

## 2026-07-13 — final-source seed-7411 LoRA G0 passes

- The canonical receipt `runs/setup/g0_lora_seed7411.json` has file SHA-256
  `8c381fc7e883e384debef6ce64e55dd57aacab6f30e4a17dd6ef82d4db56aa01`, identity
  `928e756fa6aea30104f0ed8a71c03620ed8663198a4811f6fcf9b6fda53f820c`, and exact status
  `MODEL_SMOKE_PASS`. It reopens against final source `1d1368cf…434b0a`, data manifest
  `85286a95…0cd9`, design identity `d943b909…ac52`, seed-7411 initialization identity
  `bbb21fef…0ae`, the pinned training lock, and only `Qwen/Qwen3.5-4B` revision `851bf6e8…cd0a` on
  the Transformers backend. No branch authorization was supplied or required.
- The pinned snapshot proof covers the same nine exact model/tokenizer files at the registered
  revision. The target manifest contains 62 ordered modules and the LoRA bank contains 16,232,448
  trainable adaptation parameters, with every output factor still exactly zero at construction.
- Pinned-PEFT output, A-gradient, and B-gradient comparisons pass with zero observed error in both
  FP32/dropout-off and bf16-autocast/dropout-0.05 regimes. Enabled-versus-disabled output and both
  pre/post-optimizer K=1 errors are zero, with zero K=1 adaptation calls.
- Both state-only optimizer probes and the live joint probe give finite nonzero gradients to every
  required recurrent group and all 124 LoRA tensors while producing zero base-model gradient
  tensors. Every required Adam state is complete and finite with no registered exemption.
- The ten-step path is finite at 0.311 seconds per step. The structurally disjoint setup-only K=12
  row is finite with exactly 682 ordered calls across 11 identical target cycles. Destructive
  checkpoint restoration returns both adaptation and common-state digests exactly and changes logits
  by zero. Peak allocated/reserved memory is 11.20/11.36 GiB, leaving 35.63 GiB free.
- The empty contrast ledger remains byte-identical at file SHA-256 `0c03a0d8…ac261`, identity
  `b122d490…3c14`, with `events: []`. G0 opened only the permitted training payload and created setup
  evidence, not scientific evidence. Independent receipt audit recomputed every lineage, all nine
  pinned snapshot files, parameter geometry, mechanics gate, and ledger invariant and gives `GO`.

## 2026-07-13 — corrected seed-7411 LoRA positive control passes 48/48

- The canonical receipt `runs/setup/positive_control_lora_seed7411.json` has file SHA-256
  `f71ee195b3e326abe59daca019b266932990f17adff20ceba8b414f5ac8c88ec`, identity
  `8db4595e1473d2f26d1d99d2ac1bb9be45947720d676bf43cc697cb90b402df7`, exact status
  `POSITIVE_CONTROL_PASS`, and exact G0 lineage identity `928e756f…820c`. It binds final source,
  data, design, initialization, model, revision, backend, and training lock with no branch receipt.
- The oracle-coded readout scores 1.0. The setup corpus is the frozen 48-row factorial grid with two
  rows in each depth 2/3/4 × query kind × family × template cell, canonical row digest
  `581dadcb…ef41`, and zero structural overlap with every result split.
- The corrected path completes exactly 256 optimizer updates, 16 sequential singleton microbatches
  per update, and 4,096 total presentations. There are 256 clips and optimizer steps per parameter
  group, 257 zero-gradient calls, no early stop, and no checkpoint selection. The first 16 rows have
  86 exposures, the remaining 32 have 85, and depth exposures are exactly 1,368/1,368/1,360.
- Row order, dropout schedule, and optimizer-event digests are respectively `b3ece16c…e6fa`,
  `8dc49101…98c`, and `ae8322bc…b6b`. Dropout probes at presentations 1, 2,048, and 4,096 have the
  registered call geometry. Optimizer probes at updates 1, 16, 64, 128, and 256 show finite positive
  group norms, constant learning rate `2e-4`, zero base trainables, finite nonzero gradients for
  every required recurrent and LoRA tensor, and the registered answer-only aggregate exemption.
- Fixed intact/disabled diagnostics at updates 0, 1, 16, 64, 128, and 256 leave parameters, model
  mode, and CPU/CUDA random streams unchanged. Intact terminal joint accuracy progresses from 0.00
  to 0.00, 0.0625, 0.75, 1.00, and 1.00. The fixed final is 48/48, exceeding the 0.95 gate; the
  adaptation-disabled fixed final remains 0/48.
- Adam state is complete and finite for every required parameter. Adaptation input/output and common
  state move by L2 norms 9.046, 8.905, and 3.425, and the final trainable-value digest differs from
  initialization. The receipt authorizes training but is setup evidence only: benchmark reads are
  zero, result/sealed payload lists are empty, and the ledger remains byte-identical at
  `0c03a0d8…ac261`, identity `b122d490…3c14`, with `events: []`. Independent audit regenerated all
  48 rows and all 4,096 schedule events, recomputed the full row/dropout digests, reopened every
  lineage and diagnostic invariant, and gives `GO`.

## Current authorization

Seeds 7412 and 7413 LoRA G0 and their identical positive controls are now authorized. No result
training is authorized until all three per-seed setup gates pass canonically.

## 2026-07-13 — seed-7412 LoRA G0 stops on aggregation-scalar precision

- Seed 7412 passed the pinned-snapshot load, zero-function and K=1 checks, K=4 call geometry, both
  state-only optimizer probes, a finite live-joint forward/backward, and the registered joint dropout
  schedule. It then stopped before joint clipping or stepping at the frozen all-groups reachability
  gate. No canonical `runs/setup/g0_lora_seed7412.json` was created.
- All 124 LoRA tensors, four initializer tensors, the step encoder, all eight sufficiency tensors,
  and the damping scalar had finite nonzero gradients. The frozen base remained gradient-free.
  `aggregate_logit.grad` existed and was finite but its norm was exactly zero, so the joint all-groups
  gate correctly failed even though the state-only-oriented helper excludes that scalar from its
  `all_required_tensors_finite_nonzero` summary.
- Preserved the failure at
  `runs/failures/g0_lora_seed7412_live_joint_gradient_reachability_failure.json`, identity
  `ce3406f8fa788c08421687d5d6a0843a2eb7035fd254d5086123d895b2bb634c`. It binds the final-source
  contract, design, data, model/revision/backend, seed-7412 initialization, reached and unreached
  checks, train-only access, zero benchmark/sealed access, and denies every downstream authorization.
  This is setup-mechanics evidence, not a LoRA or state-formation result.
- Root-cause review found that the FP32 `aggregate_logit` is sigmoid-transformed and cast to BF16
  before broadcasting through the registered last-state/mean-state convex mix. Two otherwise matched
  seed-7411 G0 runs produced scalar gradients whose de-sigmoided magnitudes are exactly `2^-12` and
  `2^-11`, while seed 7412 produced zero with an allocated gradient tensor. This is consistent with a
  one-example BF16 projection/reduction quantization defect and argues against simple graph
  disconnection or magnitude underflow; the regenerated live gate remains the falsification test.
- An unchanged retry is prohibited. The only reviewed repair keeps the existing BF16 mean arithmetic,
  forms the scalar convex mix in FP32, and casts its completed result once back to model dtype. It
  preserves K=1, parameters, initialization, row, masks, objective, schedule, thresholds, and both
  capacity arms; the nonzero joint-scalar gate remains strict. Automatic G0 failure persistence must
  also be added before retry.
- Any repair changes the source digest. After repair tests and independent review pass, every current
  source-`1d1368cf…434b0a` setup artifact must be archived, setup regenerated, and G0/control replayed
  from seed 7411. Seed 7413 and all result-bearing stages remain prohibited.

## Current authorization

No model-bearing stage is authorized until the aggregation-precision and G0-failure-persistence
repair is reviewed and the source-bound setup is regenerated. No result training is authorized.

## 2026-07-13 — aggregation precision and G0 persistence repair passes review

- Implemented the reviewed narrow numerical repair under source contract
  `d4269bf34f7c80affcc8c1e8a33fee9afddcd912d1bd9dead223e520ee108b36`. The registered recurrence
  states and their mean retain the live BF16 dtype. Only the FP32 sigmoid scalar's convex mix of the
  already-computed last and mean states runs in FP32 with autocast disabled, followed by one cast back
  to model dtype. K=1, state-only optimizer exemption, parameters, RNG, calls, rows, objectives,
  schedules, thresholds, and capacity symmetry are unchanged.
- A deterministic width-256 BF16 cancellation case gives the legacy scalar gradient exactly zero and
  the production helper's gradient `0.0450000092`, matching the analytic FP32 derivative. The same
  check passes directly on the registered RTX 6000 Ada. Random width-2,560 CUDA probing confirms the
  intended forward effect is at BF16 rounding scale and common to both capacities. The frozen live G0
  gate remains strictly nonzero; a repeated zero after regeneration is a hard stop, not permission to
  add rows or weaken the gate.
- Refactored G0 into a thin persistence guard and instrumented attempt. Every reached failure binds
  only the lineage actually validated, retains structured gradient/dropout progress, denies all
  downstream authorization, and re-raises. One serialization is written into two independently
  fsynced staging files; the source-qualified mirror is installed first and the canonical failure
  second without replacement. Their final inodes are independent, and existing files, broken
  symlinks, symlinked ancestors, and partial-link states fail closed.
- The setup invalidation helper accepts canonical G0 pass or failure receipts. Reached lineage must
  be exact; null lineage is allowed only at its corresponding early stage and may not contradict the
  completed-check list. A failed G0 requires its identical nonsymlink mirror and cannot coexist with
  a positive control. The historical seed-7412 manual receipt remains tracked-only and unchanged.
- Source-contract version 6 includes the new tests. The full suite passes 201/201: 38 invalidation/
  recovery, 14 G0 persistence/authorization, three aggregation precision, and 27 static/CLI/
  provenance tests. Independent numerical, failure-persistence, and real-artifact archive audits give
  `GO`.
- A read-only rehearsal validates the real 21-file source-`1d1368cf…434b0a` setup totaling
  18,288,790 bytes and trigger identity `ce3406f8…b634c` against replacement source `d4269bf3…8b36`.
  Nothing has yet been moved. The next action is to publish this repair, archive that exact setup,
  regenerate all source-bound artifacts, and replay from seed 7411.

## Current authorization

Archival of source-`1d1368cf…434b0a` setup is authorized after this repair is published and green in
CI. No model-bearing stage or result training is authorized until replacement setup is regenerated.

## 2026-07-13 — source-1d setup archived under the repaired source

- Commit `799fb98b` passed both repository-validation and site-publish workflows before mutation.
- Archived the exact 21-file source-`1d1368cf…434b0a` setup (18,288,790 bytes; files identity
  `d05f410c…9d88`) under replacement source `d4269bf3…8b36`, using the preserved seed-7412 G0
  failure identity `ce3406f8…b634c` as the external trigger.
- The durable and tracked archive receipts are byte-identical at SHA-256 `e899bc46…9484`, receipt
  identity `13cdcaec…2050b`. Canonical CPU-smoke, generated-data, initialization, G0, and control
  paths were deleted only after archive verification. The generated-data directory retains only its
  tracked `.gitignore`; both tracked setup directories are absent; all source-1d initialization
  payloads now live only in the immutable archive.
- Replayed the archival command as a read-only idempotency check; it reopened the complete archive,
  trigger, tracked receipt, and cleanup postconditions without rewriting the archive.

## Current authorization

Replacement-source CPU smoke, deterministic data/empty-ledger generation, and all three shared
initialization bundles are authorized. No G0 or model-bearing stage is authorized until that entire
setup is regenerated and strictly reopened under source `d4269bf3…8b36`. No result training is
authorized.

## 2026-07-13 — replacement-source setup regenerated

- CPU smoke passed under source `d4269bf3…8b36` without loading a model; receipt SHA-256 is
  `1d5a57c9…6fdb`.
- Regenerated all seven deterministic splits with zero cross-split structural duplicates and zero
  benchmark reads. Manifest SHA-256 is `e935c31a…1e57`, data contract is `8e95991b…d5b`, and the
  access-ledger identity is `eaf7ba23…b3cb` with `events: []`. The strict reopen checked every
  compressed payload but decompressed no sealed contrast split.
- Regenerated and canonically reopened all three shared initialization bundles. Bundle SHA-256 /
  receipt identity: seed 7411 `539e8e6c…5dc26` / `a79d0f9b…07f8a`; seed 7412
  `bf916363…15e9d` / `28e7d6db…36d86`; seed 7413 `59699025…2593c` / `c4e8657b…40ffa`.
  Every sidecar is byte-identical to its tracked mirror, and every tensor-value digest exactly equals
  the corresponding source-1d bundle.

## Current authorization

Seed-7411 LoRA G0 is authorized as the first model-bearing replay under source `d4269bf3…8b36`.
Later seeds, positive controls, and all result training remain blocked until their exact predecessor
setup gates pass and are committed.

## 2026-07-13 — replacement-source seed-7411 LoRA G0 passes

- Canonical receipt SHA-256 `5127c305…ecedd`, identity `185835ee…3216`, under source
  `d4269bf3…8b36` and manifest `e935c31a…1e57`.
- The repaired `aggregate_logit` has a present, finite, nonzero live-joint gradient norm
  `5.328678525984287e-5`. Every one of 124 LoRA tensors and every other required recurrent group is
  finite/nonzero; the frozen base has no gradient tensors.
- Both K=1 parity probes are exact with zero adaptation calls, the K=12 setup path is finite with the
  exact 682 adaptation calls, and destructive checkpoint restoration returns exact tensor digests
  and zero logit error. The setup receipt records train-only access, zero benchmark/sealed access,
  and authorizes only the seed-matched positive control.

## Current authorization

The seed-7411 LoRA positive control is authorized. Seeds 7412/7413 G0 and all result training remain
blocked until this setup-control replay passes and is committed.

## 2026-07-13 — replacement-source seed-7411 positive control passes

- Canonical receipt SHA-256 `9dcd28ee…14850`, identity `6a1394d9…cefa`, with exact lineage to G0
  identity `185835ee…3216` and manifest `e935c31a…1e57`.
- Oracle readout accuracy is 1.0. The fixed-final intact score is 48/48 exact terminal triples and all
  144 trajectory steps; adaptation-disabled is 0/48 exact terminal triples. The path therefore still
  exercises the LoRA update after the FP32 aggregation repair.
- The control completed exactly 256 optimizer updates, accumulation 16, and 4,096 singleton
  presentations. It opened no result payload, benchmark, or sealed contrast and remains setup-only
  evidence.

## Current authorization

Seed-7412 LoRA G0 is authorized under source `d4269bf3…8b36`. Seed 7413, later positive controls,
and all result training remain blocked until their exact predecessor setup gates pass and are
committed. The unchanged nonzero aggregation-gradient gate is mandatory.

## 2026-07-13 — replacement-source seed-7412 LoRA G0 validates the precision repair

- Canonical receipt SHA-256 `3ef95869…aa2ca`, identity `737a8b39…0a89f`, under source
  `d4269bf3…8b36` and manifest `e935c31a…1e57`.
- The historically exact-zero `aggregate_logit` gradient is now present, finite, and nonzero at
  `6.673135794699192e-5`. All other live-joint trainable groups remain finite/nonzero, the frozen base
  has no gradients, K=1 is exact, K=12 is finite with 682 calls, and checkpoint roundtrip error is
  zero.
- The unchanged gate passed without a tolerance or exemption. This supports the preregistered BF16
  projection/reduction explanation and resolves the mechanics question; it is not scientific evidence
  about LoRA capacity.

## Current authorization

The seed-7412 LoRA positive control is authorized. Seed 7413 and all result training remain blocked
until the seed-7412 control passes and is committed.

## 2026-07-13 — replacement-source seed-7412 positive control passes; result boundary stays closed

- Canonical receipt SHA-256 `c76da222…a254`, identity `02a329d9…669a`, with exact lineage to G0
  identity `737a8b39…0a89f` and manifest `e935c31a…1e57`.
- Oracle readout accuracy is 1.0. The fixed-final intact score is 48/48 exact terminal triples;
  adaptation-disabled is 0/48. The run completed exactly 256 optimizer updates, accumulation 16,
  and 4,096 singleton presentations, with parameter changes in the LoRA and common-state groups.
- The receipt records zero benchmark, result-payload, and sealed-contrast access and remains
  setup-only evidence under source `d4269bf3…8b36`.
- A read-only pre-result authorization re-audit then found that generic receipt loading does not
  re-enforce every producer-written model/backend, access, scientific-evidence, and downstream
  authorization field. It also found stage-ordering firewalls that exist in the runbook but are not
  all enforced at the execution boundary. No result was run.
- The seven fail-open classes are preserved under the still-current source contract as a
  nonauthorizing audit-failure receipt at identity `18892121…3a13`. It is the canonical trigger for
  archiving source-d426 setup after the repaired source is published.

## Current authorization

No further model-bearing stage or result training is authorized. Preserve this source-d426 setup,
repair and adversarially test the authorization consumers, archive every invalidated source-d426
artifact, regenerate source-bound setup, and replay its gates in order before scientific execution.

## 2026-07-13 — source-v8 published; source-d426 setup durably archived

- The first source-v7 archive preflight stopped before mutation because successful legacy positive-
  control receipts omit the later explicit result-evaluation deny bit. Source v8 accepts only that
  legacy-safe omission or exact boolean `false`, rejects `true`, `null`, and integer substitutes, and
  removes the redundant unignored per-source lock while retaining the shared `runs/run.lock`.
- Source-v8 reviewed implementation `f9364c36…b9d873` and full contract `7991d46a…b1cc88` pass
  357/357 tests and the exact machine GO. Commit `ee729def` was pushed to `main`; both Validate
  Repository and Publish Research Site completed successfully before archival resumed.
- The registered transaction preserved 23 source-d426 files totaling 18,927,960 bytes. The archive
  files identity is `1538f2f2…ec3ed0`, tracked receipt file SHA-256 is `9aa04d35…efc1a1`, and receipt
  identity is `e7a71362…818b77`; it binds replacement source `7991d46a…b1cc88` and trigger identity
  `18892121…3a13`.
- Independent verification recomputed every payload hash and both receipt identities, matched exact
  archive and 23-leaf zero-quarantine membership, confirmed empty canonical CPU/setup directories,
  only `.gitignore` under generated data, and no live initialization bundle. An immediate idempotent
  rerun revalidated the same evidence without further mutation.

## Current authorization

Only archive-checkpoint validation, commit, push, and workflow verification are authorized. Do not
regenerate CPU smoke, data, initialization, or any model-bearing setup until that checkpoint is green.
No result training is authorized.

## 2026-07-13 — source-v9 makes retained setup directories clone-stable

- Repository validation correctly rejected the archive's empty retained `runs/cpu_smoke` and
  `runs/setup` directories because Git cannot represent them and CI would see a different tree.
- Source v9 adds one tracked empty `.gitkeep` per directory. The archiver requires an empty,
  inode-distinct stable regular file; excludes it from archive evidence; permits no other residue;
  preserves it through cleanup; and revalidates it at the terminal postcondition. Missing, nonempty,
  hardlinked, symlinked, and extra entries fail closed.
- Reviewed implementation `1c73fbf9…d5ad3b` and full source contract `5629a3a4…99e236` have exact
  machine GO. No setup was regenerated and no model or sealed payload was opened during this
  structural repair.

## Current authorization

Only combined source-v9/archive-checkpoint validation, commit, push, and workflow verification are
authorized. Regeneration and every model-bearing or result stage remain blocked until both workflows
are green.

## 2026-07-13 — source-v9 non-model setup regenerated

- Combined source-v9/archive commit `9932a560` passed both Validate Repository and Publish Research
  Site before regeneration began. The box exposed one idle RTX 6000 Ada with 49,140 MiB total; the
  pinned environment and required training extensions passed preflight.
- CPU smoke passed without loading a model at SHA-256 `1655354b…45a38`, source
  `5629a3a4…99e236`, and zero benchmark reads.
- All seven deterministic splits regenerated with zero structural duplicates. Manifest SHA-256 is
  `957013ad…a4517`, data-contract identity is `3677a0a3…ae761`, and the sealed-access ledger remains
  `events: []` at identity `eb8028bf…9c84dc`. Post-generation inspection read only manifest and
  compressed-byte metadata, not sealed rows.
- All three shared initialization bundles reopened through the canonical loader, have byte-identical
  external sidecars and tracked mirrors, and exactly reproduce their archived tensor-value digests.
  Bundle SHA-256 / receipt identity: seed 7411 `5ed9d5c6…0e1b8` / `74ddebb1…31413`; seed 7412
  `15366ea6…dcb2c` / `01dd7e7c…1ee4b`; seed 7413 `bda608a2…bf8b4` / `a7ab7d5d…e912f`.

## Current authorization

Only non-model setup validation, commit, push, and workflow verification are authorized. LoRA G0,
positive controls, result training, evaluation, and analysis remain blocked until both workflows are
green.

## 2026-07-13 — source-v10 seed-7411 LoRA setup pair passes

- Non-model setup commit `33abfe33` passed both required workflows before the model-bearing command.
- Seed-7411 G0 passed at file SHA-256 `efde2db9…420f9`, receipt identity `0fe46a0c…7a383`. It binds
  the pinned Qwen revision, source v10, manifest `0b1cca35…7422a`, and seed-7411 initialization
  `e202efb8…3e0d8`.
- The exact 62-target, 16,232,448-parameter LoRA setup passed both PEFT parity regimes with zero
  observed output/A-gradient/B-gradient error. K=1 and destructive checkpoint restoration have zero
  logit error. The live joint probe gives finite nonzero gradients to all 124 LoRA tensors and every
  required common-state tensor, no base tensor, and aggregation-scalar norm `2.2352195e-5`; K=12 is
  finite. Peak allocated memory was 11.197 GiB.
- The seed-matched positive control passed at file SHA-256 `04b7f995…a34f6`, identity
  `18d02610…ebe89`: oracle accuracy 1.0; fixed-final intact 48/48; disabled 0/48; exactly 256 updates,
  accumulation 16, and 4,096 presentations; adaptation and common-state parameters both changed.
- Both receipts record zero benchmark and sealed-contrast access and are setup-only evidence. The
  control's seed-specific training authorization does not satisfy the three-seed setup barrier.

## Current authorization

Only seed-7411 setup-pair validation, commit, push, and workflow verification are authorized. Seed
7412 G0/control may begin only after both workflows are green. Result training, evaluation, and
analysis remain blocked.

## 2026-07-13 — source-v10 seed-7412 LoRA setup pair passes

- Seed-7411 setup commit `d0642d4a` passed both required workflows before seed 7412 began.
- Seed-7412 G0 passed at file SHA-256 `e2ef4951…fc18f`, receipt identity `4af55cc3…e30de`, with exact
  lineage to initialization bundle `ab0b70c1…9169b` and the unchanged source/manifest.
- Both PEFT parity regimes, K=1 before/after, and destructive checkpoint roundtrip have zero observed
  error. All 124 LoRA tensors and every required common-state group have finite nonzero live-joint
  gradients, the base has none, aggregation-scalar norm is `1.0440836e-4`, and K=12 is finite.
- The seed-matched positive control passed at file SHA-256 `1097d31b…c41e5`, receipt identity
  `ae4fdb5f…e461b`: oracle accuracy 1.0; fixed-final intact 48/48; disabled 0/48; exactly 256 updates,
  accumulation 16, and 4,096 presentations; parameter values changed.
- Both receipts record zero benchmark and sealed-contrast access and remain setup-only evidence. The
  complete setup barrier still lacks seed 7413.

## Current authorization

Only seed-7412 setup-pair validation, commit, push, and workflow verification are authorized. Seed
7413 G0/control may begin only after both workflows are green. Result training, evaluation, and
analysis remain blocked.

## 2026-07-13 — source-v10 seed-7413 LoRA setup pair passes; setup matrix complete

- Seed-7412 setup commit `41d587ec` passed both required workflows before seed 7413 began.
- Seed-7413 G0 passed at file SHA-256 `b1a42d2e…5b9256`, receipt identity `1909508f…b47e43`, with
  exact lineage to initialization bundle `7dca25ea…a7c5c` and the unchanged source/manifest.
- Both PEFT parity regimes, K=1 before/after, and destructive checkpoint roundtrip have zero observed
  error. All 124 LoRA tensors and every required common-state group have finite nonzero live-joint
  gradients, the base has none, aggregation-scalar norm is `1.5991002e-3`, and K=12 is finite.
- The seed-matched positive control passed at file SHA-256 `5ea3a552…fa01c7`, receipt identity
  `7ed7332a…4f85cc`: oracle accuracy 1.0; fixed-final intact 48/48; disabled 0/48; exactly 256 updates,
  accumulation 16, and 4,096 presentations; parameter values changed.
- All three source-v10 LoRA G0/control pairs now pass. Every setup receipt records zero benchmark and
  sealed-contrast access and remains setup-only evidence; no scientific training has started.

## Current authorization

Only final setup-matrix validation, commit, push, and workflow verification are authorized. After
both workflows pass, run the three Stage-A LoRA joint cells in frozen seed order. Evaluation,
analysis, and all conditional branches remain blocked until their exact upstream barriers authorize
them.

## 2026-07-14 — Stage-A seed-7411 LoRA joint training completes

- Complete setup commit `5b7c22fe` passed both required workflows before the first scientific cell
  began. The source-v11 digest remained exactly `5a8ed26d…6666`, `main` was aligned, and the sole RTX
  6000 Ada was idle with 48.6 GiB free.
- The source-v11 repaired launch boundary durably progressed from its authorized attempt through
  terminal `COMPLETE`; there was no retry or partial-recovery path. Exactly 1,500 fixed updates ran in
  9,217.945800065994 seconds. Interim scientific metric values were not inspected during execution.
- The production classifier reopens the cell as `COMPLETE` with no errors. Run SHA-256 is
  `19cac1e2…7aa60`, run identity `6db22c2f…f81f1`, checkpoint-metadata SHA-256
  `0c68e7a5…14966`, and checkpoint identity `174f3c69…cf9e`.
- Adaptation-state SHA-256 is `9af35963…9e71f`; loop-state SHA-256 is `333f8c44…50a5`; optimizer
  journal SHA-256 is `45664bad…e5a3`; metrics SHA-256 is `2b97d188…8ade`. Each of the four tracked
  mirrors is byte-identical to but inode-distinct from its external counterpart.
- All 1,500 optimizer rows report finite gradients and zero base trainable parameters. The cell binds
  setup barrier `eb1fa85a…bc77`, seed-7411 G0/control, source v11, and the exact initialization.
  It opened only train, no benchmark or sealed contrast. It does not authorize evaluation until all
  three Stage-A training cells are complete.

## Current authorization

Only seed-7411 training-cell validation, commit, push, and workflow verification are authorized.
After both workflows pass, run Stage-A seed 7412. Evaluation, analysis, and conditional branches
remain blocked on the complete Stage-A barrier.

## 2026-07-14 — Stage-A seed-7412 LoRA joint training completes

- Seed-7411 training commit `72d526d8` passed both required workflows before the second scientific
  cell began. Source v11 remained exactly `5a8ed26d…6666`, local and remote `main` were identical,
  and the sole RTX 6000 Ada was idle.
- The authorized seed-7412 attempt reached terminal `COMPLETE` without retry or recovery. Exactly
  1,500 fixed updates ran in 9,202.148542404175 seconds. Interim scientific metric values were not
  inspected during execution.
- The production classifier reopens the cell as `COMPLETE` with no errors. Run SHA-256 is
  `0bb1680e…1e05`, run identity `f7d049c6…d0ca`, checkpoint-metadata SHA-256
  `1b588d9f…e9fe`, and checkpoint identity `90750967…0324`.
- Adaptation-state SHA-256 is `cba9fb80…8c74`; loop-state SHA-256 is `03f4b811…8076`; optimizer
  journal SHA-256 is `32109aa7…948a`; metrics SHA-256 is `ce8e7d32…348a`. Each of the four tracked
  mirrors is byte-identical to but inode-distinct from its external counterpart.
- All 1,500 optimizer rows report finite gradients and zero base trainable parameters. The cell binds
  setup barrier `eb1fa85a…bc77`, seed-7412 G0/control, source v11, and the exact initialization. It
  opened only train, no benchmark or sealed contrast. It does not authorize evaluation until all
  three Stage-A training cells are complete. The realized-artifact suite passes 363/363.

## Current authorization

Only seed-7412 training-cell validation, commit, push, and workflow verification are authorized.
After both workflows pass, run Stage-A seed 7413. Evaluation, analysis, and conditional branches
remain blocked on the complete Stage-A barrier.

## 2026-07-14 — Stage-A seed-7413 LoRA joint training completes; barrier complete

- Seed-7412 training commit `07547f4e` passed both required workflows before the final Stage-A cell
  began. Source v11 remained exactly `5a8ed26d…6666`, local and remote `main` were identical, seeds
  7411/7412 classified `COMPLETE`, seed 7413 classified `ABSENT`, and the sole GPU was idle.
- The authorized seed-7413 attempt reached terminal `COMPLETE` without retry or recovery. Exactly
  1,500 fixed updates ran in 9,527.480490207672 seconds. Interim scientific metric values were not
  inspected during execution.
- The production classifier reopens the cell as `COMPLETE` with no errors. Run SHA-256 is
  `8bbdb3b3…8b59`, run identity `07f91939…362b`, checkpoint-metadata SHA-256
  `25936f83…fbfd`, and checkpoint identity `d65d5bb3…e68d`.
- Adaptation-state SHA-256 is `2ed8167f…9a65`; loop-state SHA-256 is `97d59a7f…b298`; optimizer
  journal SHA-256 is `44f9eaa2…fc9c`; metrics SHA-256 is `6afe4d38…cd38`. Each of the four tracked
  mirrors is byte-identical to but inode-distinct from its external counterpart.
- All 1,500 optimizer rows report finite gradients and zero base trainable parameters. The cell binds
  setup barrier `eb1fa85a…bc77`, seed-7413 G0/control, source v11, and the exact initialization. It
  opened only train, no benchmark or sealed contrast. The realized-artifact suite passes 363/363.
- All three Stage-A cells classify `COMPLETE` with no errors. Their complete training-barrier identity
  is `31d7f6f3…6451`.

## Current authorization

Only final Stage-A training validation, commit, push, and workflow verification are authorized.
After both workflows pass, evaluate the three LoRA joint checkpoints on the trigger set in frozen
seed order, then analyze `lora_joint`. Conditional branches remain blocked on that analysis receipt.

## 2026-07-14 — seed-7411 trigger evaluation completes; analyzer path defect found pre-result

- Complete Stage-A commit `664d6574` passed both required workflows before evaluation began. The
  reached-training barrier authorized the exact seed-7411 trigger invocation on source v11.
- Evaluation completed in 978.507162809372 seconds with exactly 3,072 intact and 3,072
  adaptation-disabled rows. Summary identity is `fd7a516e…9264`; intact rows SHA-256 is
  `59e55adc…f6c`; disabled rows SHA-256 is `65323163…745`. The receipt binds checkpoint
  `174f3c69…cf9e`, reached-training barrier `6d506285…589e`, and source `5a8ed26d…6666`.
- The evaluator opened only `validation`, `depth_extrapolation`, and `joint_holdout`; it records zero
  benchmark reads, no sealed-contrast payload, and no contrast-ledger event. Scientific values were
  not inspected.
- The independent production `_load_evaluation` reopen stopped before consuming result rows because
  `_canonical_expected_path(ROOT / config["paths"]["large_artifacts_dir"])` rejects the registered
  `../../large_artifacts/...` value as non-lexical-canonical. The CLI analyzer necessarily reaches
  the same path. This is a deterministic mechanics defect, not a LoRA result.
- Changing source v11 would invalidate the source-bound setup/training/evaluation graph. Preserve it
  unchanged. Freeze a separate, self-contained analysis-recovery experiment that consumes the exact
  v11 receipts and changes only the internal registered-path resolution boundary before opening any
  scientific values.

## Current authorization

Only seed-7411 evaluation validation, documentation, commit, push, and workflow verification are
authorized. After both workflows pass, freeze and adversarially review the separate analysis-recovery
consumer. Remaining trigger evaluations may resume only after that consumer is mechanically ready;
no result value, LoRA verdict, sealed contrast, or conditional training arm is yet authorized.

## 2026-07-14 — recovery published; seed-7412 trigger completes; operator visibility deviation

- The separate `qwen35_4b_state_formation_analysis_recovery` consumer passed 10/10 focused tests and
  its non-result smoke at file SHA-256 `02f6f927…43f7`, receipt identity `30353be5…2f8f`, and
  recovery source contract `6ab26016…e94b`. It changes only the exact registered external prefix and
  clean descendants before calling the unchanged v11 canonical repository/no-symlink validator.
- Recovery commit `a6360cc1` and its fully enriched date follow-up `e35e071e` passed both repository
  workflows. The producer source remains exact v11; no result value was opened by the recovery smoke.
- The fixed seed-7412 LoRA joint trigger command then completed once, without retry, in
  1,008.6487309932709 seconds. Summary file SHA-256 is `a18ba761…f4e2`, receipt identity
  `03eaf96b…597d`, intact rows SHA-256 `dd3f9424…cc4f`, and disabled rows SHA-256
  `743f821a…c085`; both modes contain exactly 3,072 rows. The receipt reopens checkpoint
  `90750967…0324`, has exact K=1 bypass, opens only the three trigger payloads, and records zero
  benchmark/sealed-contrast access and no branch authorization.
- During post-run integrity validation, an intended metadata-only `jq` projection included the whole
  `modes` object. That object contains per-split scientific metrics, so their values were printed
  before seed 7413. This was an unauthorized operator-visibility deviation, not a model, data,
  receipt, or analyzer defect. It is documented separately and cannot be undone.
- No metric was classified, pooled, compared to the gate, or used to choose an action. Seed 7413,
  the analysis code, seed order, fixed finals, thresholds, and all conditional branches were frozen
  before exposure. To prevent adaptive stopping, seed 7413 remains compulsory and must run unchanged
  after this exact evaluation/deviation record is published and both workflows pass.

## Current authorization

Only seed-7412 evaluation/deviation documentation, full validation, commit, push, and workflow
verification are authorized. After both workflows pass, run the fixed seed-7413 LoRA joint trigger
evaluation. Aggregate analysis and every conditional branch remain blocked on the complete exact
three-seed evaluation matrix. The terminal report must disclose imperfect operator blinding.

## 2026-07-14 — seed-7413 trigger completes; exact Stage-A evaluation matrix complete

- Seed-7412 evaluation/deviation commit `27166420` passed both required workflows before the fixed
  seed-7413 command began. Source v11 remained exactly `5a8ed26d…6666`, the seed/checkpoint/runner
  were unchanged, and the sole GPU was idle.
- The authorized seed-7413 trigger evaluation completed once, without retry, in
  986.8842830657959 seconds. Summary file SHA-256 is `411af149…80a`, receipt identity
  `b35b9e14…da70`, intact rows SHA-256 `d62c43a9…b30c`, and disabled rows SHA-256
  `683156e9…6cb`; both modes contain exactly 3,072 rows.
- The receipt strictly reopens checkpoint metadata `25936f83…fbfd`, checkpoint identity
  `d65d5bb3…e68d`, target-cell receipt `07f91939…362b`, training barrier `6d506285…589e`, exact
  source v11/config, and the fixed evaluation axes. It reports exact K=1 bypass, opens only
  `validation`, `depth_extrapolation`, and `joint_holdout`, and records zero benchmark or
  sealed-contrast access and no downstream authorization.
- The corrected post-run projection selected metadata and row-lineage fields only; no seed-7413
  scientific metric was printed or inspected. The prior seed-7412 visibility deviation remains
  disclosed and does not change the frozen analysis or branch rule.

## Current authorization

Only seed-7413 evaluation documentation, validation, commit, push, and workflow verification are
authorized. After both workflows are green, run the already-published recovery consumer exactly as
`--phase lora_joint`. Its original v11 receipt alone classifies Stage A and authorizes the next
registered branch; no conditional training, sealed contrast, or source edit is authorized before it.

## 2026-07-14 — recovered immutable analysis mandates Stage B capacity controls

- Seed-7413 evaluation commit `b326f6cd` passed Validate Repository and Publish Research Site before
  analysis. Local `main` and `origin/main` matched, and the producer/recovery frozen contracts were
  unchanged.
- The recovery consumer ran exactly once as `--phase lora_joint`. It changed only
  `_canonical_expected_path`, delegated all scientific work to immutable producer v11
  `analysis.py`, and emitted `RECOVERED_V11_ANALYSIS_COMPLETE` at sidecar identity
  `d068482a…f40e`.
- The producer output has file SHA-256 `cb9fee75…818a` and receipt identity `b973bc01…a862`. Status
  and verdict are `LORA_JOINT_MISS_CONTROLS_REQUIRED`; next stage is exactly
  `run_lora_state_only_and_fullrank_joint`.
- Zero of 57 required seed×split×depth cells passed the 0.40 terminal joint-state threshold. The
  maximum intact required accuracy was 0.0234375; per-seed maxima were 0.0234375, 0.015625, and
  0.015625. Trained, depth-extrapolation, and joint-shift categories all miss. Adaptation contrast is
  `ADAPTATION_CONTRAST_UNCERTAIN`, with neither a consistent positive intact effect nor a disabled
  reversal.
- The result establishes a valid failure of the registered LoRA joint recipe, not a rank-causal
  conclusion. No sealed contrast was opened. The previously disclosed seed-7412 visibility
  deviation did not change the frozen analysis, and imperfect operator blinding remains a terminal
  limitation.

## Current authorization

Only analysis publication, validation, commit, push, and workflow verification are authorized.
After both workflows are green, Stage B must run all three LoRA state-only controls and all three
direct-full-shape joint arms under the exact analysis authorization. Full-rank G0/positive-control
setup precedes result training; sealed contrast remains prohibited.

## 2026-07-14 — first Stage-B G0 exposes downstream copy of the path defect

- Analysis commit `719c8e5a` passed both workflows before the exact source-v11 full-rank seed-7411
  G0 command began.
- The attempt failed at `branch_authorization` before its first completed check or model load. The
  branch consumer recomputes LoRA-miss evidence and reaches v11's same nonlexical registered
  external prefix. Canonical/mirror failure file SHA-256 is `47305826…2c71`, identity
  `070c23af…aa24`; both are byte-identical and inode-distinct.
- The receipt records no model load, training/evaluation, benchmark, sealed contrast, scientific
  evidence, or downstream authorization. Producer v11 refuses overwrite until the pair is archived.
- A separate branch-recovery experiment passed 14/14 tests and exact downstream smoke at identity
  `d1135ea2…49b5`, then archived the failure at identity `ff478d40…0ec3` without removing either
  producer path. Its checkpoint must be published and green before verified retirement.

## Current authorization

Only recovery/archive documentation, validation, commit, push, and workflow verification are
authorized. After green CI, retire the exact producer failure pair through the frozen recovery,
publish that retirement, and require both workflows green before retrying full-rank seed-7411 G0.

## 2026-07-14 — branch failure archived and exact producer paths retired

- Recovery/archive commit `bdedabf4…b6b2` passed both workflows.
- The retirement transaction verified that commit contains the exact archive copy, archive receipt,
  producer canonical, and producer mirror. It then removed only the two producer paths while holding
  their exact inodes; the recovery archive remains unchanged at `47305826…2c71`.
- Retirement receipt SHA-256 is `6e4c8ee3…53ad`, identity `c9abdc59…eae7`, with zero model,
  training/evaluation, benchmark, or sealed access.

## Current authorization

Only retirement validation, documentation, commit, push, and workflow verification are authorized.
After both workflows are green, retry full-rank seed-7411 G0 through the frozen branch-recovery
wrapper. Positive control and result-bearing cells remain blocked on that passing G0 checkpoint.

## 2026-07-13 — source-v9 seed-7411 G0 stops before model load; cache proof repaired

- Non-model setup commit `ff4a8b9b` passed both repository workflows before the exact seed-7411 LoRA
  G0 command began on an otherwise idle RTX 6000 Ada.
- G0 stopped at `model_setup` before model load, wrapper construction, mechanics, training, or
  evaluation. The general stable reader refused `chat_template.jinja` because Hugging Face's standard
  cache snapshot is a symlink into `blobs/`. The canonical failure and source-qualified mirror are
  byte-identical at file SHA-256 `39ec9625…46ec7` and identity `30af333c…9cfe9`; they record only
  train validation, zero benchmark/sealed access, and no downstream authorization.
- Source v10 keeps general repository/artifact reads no-symlink. Its dedicated cache proof requires
  exact snapshot basename/revision and only canonical `../../blobs/<content-id>` targets, hashes an
  inode-distinct stable blob while revalidating the held snapshot/link, and rejects traversal,
  non-content targets, hardlinks, mixed roots/revisions, malformed indexes, and missing shards.
- The real pinned cache proof passes all nine files totaling 9,342,815,919 bytes at exact revision
  `851bf6e…cd0a`, file-set identity `06486f26…d1fe12`, without loading the model. The authorization
  context also now preserves body `StableArtifactError` diagnostics instead of mislabeling them as an
  aliased implementation review.
- Reviewed implementation `a5a494b7…6f1c4a` and full source contract `979a9012…f394b7` pass 360/360
  tests and exact machine GO.

## Current authorization

Only source-v10 validation, commit, push, and workflow verification are authorized. After both
workflows pass, archive every source-v9 setup artifact with the source-qualified G0 failure as trigger,
publish and validate that archive checkpoint, then regenerate from CPU smoke. No model-bearing retry
or result stage is authorized before those gates.

## 2026-07-13 — source-v9 setup durably archived under source v10

- Source-v10 commit `3756ce29` passed both repository workflows before archival began.
- The registered transaction preserved 20 source-v9 files totaling 17,655,138 bytes. Files identity
  is `7360b00d…1f2650`, tracked receipt file SHA-256 is `086d35af…be14e`, and receipt identity is
  `8d5fe94d…33ad5c`; it binds replacement source `979a9012…f394b7` and G0 trigger identity
  `30af333c…9cfe9`.
- Independent verification recomputed every payload/receipt hash, matched exact archive and 20-leaf
  zero-quarantine membership, confirmed only structural sentinels in canonical CPU/setup directories,
  only `.gitignore` under generated data, no live initialization bundle, and the retained byte-exact
  source-qualified G0 failure mirror. Immediate idempotent replay changed nothing.

## Current authorization

Only archive-checkpoint validation, commit, push, and workflow verification are authorized. Do not
regenerate source-v10 CPU/data/initialization or run a model until both workflows are green. No result
stage is authorized.

## 2026-07-13 — source-v10 non-model setup regenerated

- Archive commit `9c1fadde` passed both Validate Repository and Publish Research Site before
  regeneration began.
- CPU smoke passed without loading a model at file SHA-256 `ebfb68fe…17bc5`, source
  `979a9012…f394b7`, and zero benchmark reads.
- All seven deterministic splits regenerated with zero cross-split structural duplicates. Manifest
  SHA-256 is `0b1cca35…7422a`, data-contract identity is `5fba6c3c…c252`, and the sealed-access ledger
  remains `events: []` at identity `d0b9eda7…17e04`. Inspection read only receipt, manifest, and
  compressed-byte metadata; no sealed row was decompressed.
- All three shared initialization bundles reopened through the canonical loader, have byte-identical
  but inode-distinct external sidecars and tracked mirrors, and exactly reproduce their archived
  source-v9 tensor-value digests. Bundle SHA-256 / sidecar receipt identity: seed 7411
  `e202efb8…3e0d8` / `9f6923d4…147abd`; seed 7412 `ab0b70c1…9169b` / `cf90d157…806e9`; seed 7413
  `7dca25ea…a7c5c` / `791aa7ec…6201`.

## Current authorization

Only non-model setup validation, commit, push, and workflow verification are authorized. LoRA G0,
positive controls, result training, evaluation, and analysis remain blocked until both workflows are
green.

## 2026-07-13 — first Stage-A launch stops PREPARED; source-v11 repairs publication

- Final source-v10 setup commit `555cf0b5` passed both required workflows before the exact seed-7411
  Stage-A command began.
- The launch stopped before model load, train-payload access, optimizer construction, or training.
  Its durable journal reached only `PREPARED`; the canonical external directory existed but had no
  entries, and the tracked result directory was absent. This is not scientific or LoRA evidence.
- Root cause: `atomic_write_json` passed both a repository-relative full destination and its relative
  parent to the no-clobber publisher. Descriptor traversal therefore looked for the parent path a
  second time beneath itself and raised `FileNotFoundError`; the wrapper mislabeled it as an
  overwrite refusal.
- The source-v10 journal is preserved byte-exact at SHA-256
  `209d4c1f9183359535a07fe5bef3f535ebc13e2f00d4ddcf10c289501b1912df`, journal identity
  `ef72ba4e…f51c9`, and attempt authorization `d5c14f38…254ff`. Failure receipt identity is
  `6b23f95d…b8c2`; it records zero benchmark/sealed/result-payload access and no model or training.
- Source v11 normalizes no-clobber JSON destinations, publishes a leaf relative to its held parent,
  preserves the stable-I/O cause, and rebinds validated training/checkpoint/evaluation outputs to
  canonical absolute paths. It recovers only an exactly empty PREPARED output directory and rejects
  any nonempty markerless collision.
- Three new regressions reproduce the relative-path bug, exact mkdir-only crash, and unsafe nonempty
  state. The full suite passes 363/363. Reviewed implementation is `7d6cd93f…d278`; full source
  contract is `5a8ed26d…6666`. No model, GPU, benchmark, result payload, or sealed split was used in
  the repair.

## Current authorization

Only source-v11/failure-evidence validation, commit, push, and workflow verification are authorized.
After both workflows are green, archive all source-v10 setup with the preserved launch failure as
trigger, publish that checkpoint, retire the stale PREPARED paths, and replay setup from zero under
v11. No result-bearing stage is authorized.

## 2026-07-13 — source-v10 setup durably archived under source v11

- Source-v11 repair/catalog commit `94ec3780` passed both required workflows before archival began.
- The registered transaction preserved 25 source-v10 files totaling 19,566,021 bytes: all generated
  data and the empty sealed-access ledger, CPU receipt, all three initialization bundle/sidecar pairs,
  all three tracked initialization mirrors, and all three LoRA G0/control pairs.
- Files identity is `f767eb80…d4b91`; tracked receipt file SHA-256 is `ab583fc1…e217`; receipt identity
  is `252be000…5d6a3`. It binds replacement source `5a8ed26d…6666` and launch-failure trigger identity
  `6b23f95d…b8c2`.
- Independent postconditions leave only `.gitignore` under generated data and the two structural
  `.gitkeep` sentinels under CPU/setup. An immediate idempotent rerun reproduced the same 25-file
  receipt without mutation. No model, GPU, benchmark, result payload, or sealed split was opened.

## Current authorization

Only archive-checkpoint validation, commit, push, and workflow verification are authorized. After
both workflows are green, retire the canonical empty Stage-A output and stale PREPARED journal only
after revalidating their published failure/journal copies, then regenerate source-v11 setup. No model
or result-bearing stage is authorized meanwhile.

## 2026-07-13 — stale PREPARED attempt names retired after durable publication

- Archive checkpoint `24733d34` passed both required workflows before retirement.
- The canonical journal still matched its published source-v10 copy byte-for-byte at SHA-256
  `209d4c1f…12df`, with one event whose head was exactly `PREPARED`. The external result directory
  still had zero entries, and the tracked result directory was absent.
- The source-v10 failure receipt and source-v10 setup archive receipt were rehashed to identities
  `6b23f95d…b8c2` and `252be000…5d6a3`. Only then were the stale canonical journal and empty external
  directory retired. The published journal copy remains immutable under tracked failures.
- The retirement receipt records both retired paths, their preconditions, the green archive commit,
  zero benchmark/sealed/result/model/training access, and no downstream authorization.

## Current authorization

Only retirement-checkpoint validation, commit, push, and workflow verification are authorized. After
both workflows are green, regenerate source-v11 CPU/data/empty-ledger/initialization setup and publish
that checkpoint before any model-bearing gate. No result-bearing stage is authorized.

## 2026-07-13 — source-v11 non-model setup regenerated and strictly reopened

- Retirement commit `aa85086f` passed both required workflows before regeneration began.
- CPU smoke passed without loading a model at file SHA-256 `d46b32bd…0192`. The deterministic data
  manifest has file SHA-256 `d104a9c0…a22c`, data-contract identity `43363814…6669`, zero structural
  duplicates, and zero benchmark reads. The contrast ledger remains empty at receipt identity
  `01b2724b…41f3f`.
- The seven compressed payload hashes and row counts reproduce the frozen procedural substrate,
  including 12,000 train rows, 1,024 validation rows, and the three still-sealed contrast payloads.
  Validation inspected only manifest/compressed-byte metadata; no sealed row was decompressed.
- All three initialization bundles reopened through the canonical loader. Each external sidecar is
  byte-identical to but inode-distinct from its tracked mirror, and each 15-tensor value digest exactly
  reproduces source v10: seed 7411 `5338572e…af31`, seed 7412 `30aac3e3…e14`, seed 7413
  `04054449…efeb`. Bundle SHA-256 / sidecar identity: seed 7411 `a03486a1…c77` /
  `fc997954…c6b`; seed 7412 `3ae97463…021f5` / `688825b1…b26b`; seed 7413
  `1167ebeb…8b2f2` / `f7b95a1c…45ddc`.
- Regeneration used source contract `5a8ed26d…6666`, loaded no model, read no benchmark, and created
  no result evidence.

## Current authorization

Only source-v11 non-model setup validation, commit, push, and workflow verification are authorized.
After both workflows are green, replay seed-7411 LoRA G0 and its positive control. Later setup and
all result-bearing stages remain blocked by their frozen ordered barriers.

## 2026-07-13 — source-v11 seed-7411 LoRA setup pair passes

- Non-model setup commit `bcd57181` passed both required workflows before seed 7411 began.
- LoRA G0 passed at file SHA-256 `9292794f…1264d`, receipt identity `da06c198…94d0f`, with exact
  lineage to initialization bundle `a03486a1…c77`, manifest `d104a9c0…a22c`, and source
  `5a8ed26d…6666`.
- Both PEFT parity regimes, K=1 before/after, zero function, and destructive checkpoint restoration
  have zero observed error. All 124 LoRA tensors and every required common-state group have finite
  nonzero live-joint gradients, the base has none, the aggregation-scalar norm is
  `6.167606625240296e-5`, and the K=12 path is finite. The ten-step probe measured
  `0.2844161748886108` seconds/step with 11.197 GiB peak allocation and 35.629 GiB free after G0.
- The seed-matched positive control passed at file SHA-256 `ee1de715…c98d8`, receipt identity
  `c830e65b…0c259`: oracle accuracy 1.0; fixed-final intact 48/48; disabled 0/48; exactly 256 updates,
  accumulation 16, and 4,096 presentations; parameter values changed.
- Both receipts record zero benchmark and sealed-contrast access and remain setup evidence rather
  than scientific results. The complete setup barrier still lacks seeds 7412 and 7413.

## Current authorization

Only seed-7411 setup-pair validation, commit, push, and workflow verification are authorized. Seed
7412 G0/control may begin only after both workflows are green. Result training, evaluation, and
analysis remain blocked on the complete three-seed setup barrier.

## 2026-07-14 — recovered full-rank G0 passes; handoff repair frozen

- The branch-recovery retirement checkpoint `e69c1960…c79` passed both workflows before retry.
- Full-rank seed-7411 G0 passed at file SHA-256 `cdc90cd…c68f`, identity
  `e1f1c906…f89dc`: K=1 parity before/after optimizer and checkpoint restoration have zero error;
  all 62 direct-full-shape delta tensors have finite nonzero live-joint gradients and complete finite
  optimizer state; K=12 is finite; and about 22.1 GiB remains free.
- The G0 authorizes only its positive control and records zero training/evaluation, benchmark,
  contrast, or scientific evidence. First-recovery STARTED/COMPLETE file hashes are
  `f0c78055…3c87` / `755ad561…f941`.
- The positive-control handoff stopped before wrapper STARTED publication because the frozen first
  recovery tests the former failure pathname rather than its exact bytes/status. No producer or
  model work occurred.
- The additive byte/status-aware handoff suite passes 8/8. Its no-model smoke passes all six controls
  at file SHA-256 `0e0409de…6e5b`, identity `d6fecf0a…de4c`, under source contract
  `4d2ffde3…8acd`.

## 2026-07-14 — full-rank seed-7411 setup pair passes

- Handoff commit `627254f1…d892` passed both workflows before the positive control.
- Producer control file SHA-256 is `a0d17e2e…7a16`, identity `6708a4d4…9649`; handoff STARTED/
  COMPLETE file hashes are `b1b91cb2…ff2b` / `e0887536…13f1`, with COMPLETE identity
  `af0dd15c…2262`.
- Oracle and intact fixed-final joint state are 48/48; adaptation-disabled joint state is 0/48 after
  exactly 256 optimizer updates, accumulation 16, and 4,096 presentations. The 892,272,640 full-
  rank parameters and shared-state parameters both changed; early stop and checkpoint selection are
  false.
- G0/control are setup-only evidence and record zero result-payload, benchmark, contrast, or
  scientific-evidence access.

## Current authorization

Only seed-7411 full-rank setup documentation, validation, commit, push, and workflow verification are
authorized. After both workflows are green, run the seed-7412 full-rank G0 and positive control
through the handoff wrapper. Result training remains blocked on the complete setup matrix.

## 2026-07-14 — full-rank seed-7412 setup pair passes

- Seed-7411 setup commit `c507488c…3c06` passed both workflows before seed 7412 began.
- G0 file SHA-256 / identity are `10bf22fc…1d18` / `62ecb79e…951dc`; handoff COMPLETE identity is
  `d61d6441…9246`. K=1 and checkpoint-roundtrip error are zero, all 62 deltas have complete finite
  optimizer state, joint gradients are finite/nonzero, K=12 is finite, and 22.1 GiB remains free.
- Control file SHA-256 / identity are `1cbbd823…8510` / `6575e1d2…4554`; handoff COMPLETE identity
  is `6334d72f…c14a`. Oracle/intact are 48/48 and disabled is 0/48 after exact 256 updates,
  accumulation 16, and 4,096 presentations; full-rank/shared-state parameters changed.
- Both receipts remain setup-only evidence with zero result-payload, benchmark, contrast, or
  scientific-evidence access.

## Current authorization

Only seed-7412 setup documentation, validation, commit, push, and workflow verification are
authorized. After both workflows are green, run the seed-7413 full-rank G0 and positive control.
Result training remains blocked on the complete setup matrix.

## 2026-07-14 — full-rank seed-7413 pair passes; setup matrix complete

- Seed-7412 setup commit `92473a53…8530` passed both workflows before seed 7413 began.
- G0 file SHA-256 / identity are `021a8444…d635` / `4d2316d3…2ff0`; handoff COMPLETE identity is
  `fbfb282f…5aef`. K=1/round-trip error are zero, all 62 deltas have complete finite optimizer state,
  joint gradients are finite/nonzero, K=12 is finite, and 22.1 GiB remains free.
- Control file SHA-256 / identity are `8a4af0d6…fde8` / `976f28ef…14df`; handoff COMPLETE identity
  is `ccff1520…58c9`. Oracle/intact are 48/48 and disabled is 0/48 after exact 256 updates,
  accumulation 16, and 4,096 presentations; full-rank/shared-state parameters changed.
- All three full-rank G0/control pairs now pass and remain setup-only, with zero result-payload,
  benchmark, contrast, or scientific-evidence access.

## Current authorization

Only complete setup-matrix documentation, validation, commit, push, and workflow verification are
authorized. After both workflows are green, run the mandatory three LoRA state-only and three full-
rank joint Stage-B training cells under their exact producer barriers. Evaluation remains blocked on
the complete Stage-B training matrix.

## 2026-07-14 — source-v11 seed-7412 LoRA setup pair passes

- Seed-7411 setup commit `5536f785` passed both required workflows before seed 7412 began. The next
  remote commit touched only another experiment and generated indexes; source contract
  `5a8ed26d…6666` was reverified after fast-forward.
- Seed-7412 G0 passed at file SHA-256 `c19a6944…6b82d0`, receipt identity `e2229ad2…da380`, with
  exact lineage to initialization bundle `3ae97463…021f5` and the unchanged manifest/source.
- Both PEFT parity regimes, K=1 before/after, zero function, and destructive checkpoint restoration
  have zero observed error. All 124 LoRA tensors and every required common-state group have finite
  nonzero live-joint gradients, the base has none, the aggregation-scalar norm is
  `4.465342499315739e-5`, and the K=12 path is finite. The ten-step probe measured
  `0.2639180898666382` seconds/step with 11.197 GiB peak allocation and 35.629 GiB free after G0.
- The seed-matched positive control passed at file SHA-256 `1a8d263b…a27a37`, receipt identity
  `fdb1a16e…cd72a3`: oracle accuracy 1.0; fixed-final intact 48/48; disabled 0/48; exactly 256
  updates, accumulation 16, and 4,096 presentations; parameter values changed.
- Both receipts record zero benchmark and sealed-contrast access and remain setup evidence rather
  than scientific results. The complete setup barrier still lacks seed 7413.

## Current authorization

Only seed-7412 setup-pair validation, commit, push, and workflow verification are authorized. Seed
7413 G0/control may begin only after both workflows are green. Result training, evaluation, and
analysis remain blocked on the complete three-seed setup barrier.

## 2026-07-14 — source-v11 seed-7413 LoRA setup pair passes; matrix complete

- Seed-7412 setup commit `47e160bf` passed both required workflows before seed 7413 began.
- Seed-7413 G0 passed at file SHA-256 `87faadfa…aca4d0`, receipt identity `1fa2b054…eec8d5`, with
  exact lineage to initialization bundle `1167ebeb…8b2f2` and the unchanged manifest/source.
- Both PEFT parity regimes, K=1 before/after, zero function, and destructive checkpoint restoration
  have zero observed error. All 124 LoRA tensors and every required common-state group have finite
  nonzero live-joint gradients, the base has none, the aggregation-scalar norm is
  `1.5512113459408283e-3`, and the K=12 path is finite. The ten-step probe measured
  `0.281365180015564` seconds/step with 11.197 GiB peak allocation and 35.629 GiB free after G0.
- The seed-matched positive control passed at file SHA-256 `e2e70edf…fc5b57`, receipt identity
  `8d74b90d…d689b7c`: oracle accuracy 1.0; fixed-final intact 48/48; disabled 0/48; exactly 256
  updates, accumulation 16, and 4,096 presentations; parameter values changed.
- All three source-v11 LoRA G0/control pairs now pass. Every setup receipt records zero benchmark and
  sealed-contrast access and remains setup evidence rather than a scientific result.

## Current authorization

Only final setup-matrix validation, commit, push, and workflow verification are authorized. After
both workflows pass, run the three Stage-A LoRA joint cells in frozen seed order. Evaluation,
analysis, and all conditional branches remain blocked until their exact upstream barriers authorize
them.

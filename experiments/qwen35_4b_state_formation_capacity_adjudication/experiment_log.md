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

## Current authorization

Execution is retracted pending adversarial review of a fail-closed positive-control correction and
source-bound invalidation. No positive control retry, later-seed G0, or result training is authorized.

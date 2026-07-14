# Qwen3.5-4B Deep-Advantage MOPD Experiment Log

## 2026-07-12 — intake and pre-output design

- Created a new result-bearing directory rather than modifying the completed
  two-teacher experiment.
- Selected the deep-only branch because deep independently passed both
  same-prefix audit contrasts in both predecessor blocks; MOPD remained
  untested only because quick was separately mandatory.
- Reused the exact immutable 40/60 soup by SHA-256 instead of constructing a
  numerically new starting checkpoint. New training artifacts have a separate
  external root.
- Froze two new route blocks, the unchanged strict deep-over-quick-and-student
  rule, five-update locality, four 60-deep/20-soup rounds, three primary seeds,
  and unconditional final comparisons against sources, router, controls, and
  sample-more.
- Added two direct mechanism controls: deep targets on one-to-one matched
  non-deep-selected states, and quick targets on the exact selected states.
  The off-policy continuation control and parameter soups remain.
- Copied the parent harness and procedural gym, then adapted quotas, route gate,
  target-cache inventory, locality mixture, controls, and confirmation without
  generating task-model output.
- Passed 50 isolated tests and all 14 family selftests; verified the exact
  quick, deep, and immutable-soup file hashes. The smoke receipt contains no
  task-model generation.
- Committed the complete frozen design at `1ef1f5ad`, pushed it to shared
  `main`, and wrote `runs/preregistration_receipt.json` with byte hashes for all
  frozen files before any Qwen load.

## 2026-07-12 — pinned-model and installation preflight

- Passed all four pinned-runtime semantic probes and a finite Transformers
  training forward pass; vLLM resolved the registered full/piecewise graph
  geometry.
- Revalidated exact quick, deep, and soup checkpoint hashes and merge receipts.
  On the eight fixed canary prompts, every installed checkpoint differed from
  base; quick/deep differed on 8/8, soup/quick on 8/8, and soup/deep on 7/8.
- The installation gate authorizes the fresh two-block route qualification.
  No route evidence or training output exists yet.

## 2026-07-13 — fresh route qualification passes

- Generated two disjoint 192-state soup blocks and 9,216 continuations: three
  policies × four selection + four audit branches. All state, replay, model,
  engine, and branch-hash checks passed.
- Deep routed on 28/26 states. Independent audit deep-minus-soup was
  +0.16499/+0.12205 by block (pooled +0.14209, one-sided lower bound
  +0.12297); deep-minus-quick was +0.20003/+0.14203 (pooled +0.16910, lower
  bound +0.15337). The frozen deep gate passed.
- Quick also passed diagnostically on 29/18 routes in this new replication,
  unlike its predecessor reversal. The locked treatment remains deep-only;
  the result is evidence for, not permission to improvise, the later
  cross-fitted two-teacher design.
- Downstream authorization is exactly `locality_pilot`. No update has run.

## 2026-07-13 — five-update exact-logit locality passes

- The first two online candidate batches supplied 55/60 required deep routes,
  so the frozen runner opened a third batch rather than weakening the quota.
  Three batches yielded 90 deep routes among 576 failed states.
- The final round manifest contains exactly 60 deep capability units, 20 soup
  anchors, and 60 disjoint non-advantage controls. Control matching used 57
  exact cells and three family/kind fallbacks. Its SHA-256 is
  `f4a1eb0848804ddef874ece0afef89a1ea39d84f2717b971575f2ff5f021e0e8`.
- Cached all three policies for 140 samples and 35,147 active positions. The
  cache SHA-256 is
  `20c21a5bb8e8b8058d0b7377929b37fd5e2eca8e55598d61b6ac827503761e76`.
- The pilot completed all five updates with the exact 15-deep/5-soup mixture.
  Training mean corrected top-50 loss was `0.05242`; held-probe loss improved
  `0.04773→0.02947` and overlap improved `0.84840→0.85163`.
- Exact batch-of-one probes measured centered non-target drift `0.02760`,
  relative entropy drop `0.03112`, and target loss `0.01293→0.01170`. Every
  frozen check passed. The probe is one midpoint token per consumed unit, so
  the authorization is literal and deliberately not described as global
  token-position invariance.
- Downstream authorization is exactly `four_round_mopd`. No capability result,
  control comparison, or benchmark event exists yet.

## 2026-07-13 — seed-42 integration round-1 cache recovery

- Full-dose round 0 passed: 20/20 updates, exact 60-deep/20-soup geometry,
  mean corrected loss `0.05669`, probe loss `0.08318→0.05112`, and
  non-decreasing overlap. Its merged receipt is
  `8432e6391ce1f4ce328938163984f490bc424325832c891a58a5dbf35cb06920`.
- Round 1 used three fresh candidate batches. It found 81 deep routes among
  576 failed states and froze the exact 60-deep/20-anchor/60-control quotas
  (56 controls exact-cell, four family/kind).
- Target-cache construction stopped before loading any policy because one
  matched route-control episode tokenized to `3,203 > 3,072`. This was an
  implementation omission, not a registered scientific stop: the fixed
  training length existed, but completion-preserving prompt fitting did not.
- Recovery keeps the frozen `3,072` budget and every completion/target
  position, deterministically left-truncates only oldest prompt tokens, records
  the exact cut in cache/training receipts, and still fails if a completion
  leaves no causal prompt token. Regression coverage reproduces the exact
  `3,203→3,072` case. Existing states, branch ledgers, routes, quotas, and
  model outputs are reused unchanged.

## 2026-07-13 — live integration integrity audit

- Round 1 subsequently passed its frozen gate: 20/20 updates, mean corrected
  loss `0.04901`, probe loss `0.03915→0.02020`, and overlap
  `0.84121→0.84604`. Exactly 60 deep and 20 anchors were consumed with zero
  prompt truncation. The sole 131-token cut remains a cache-only matched
  route-control sample.
- An independent math audit confirmed the teacher-top-50 corrected reverse-KL
  value, gradient, causal indexing, masks, quotas, and no-hint construction.
  It also exposed prospective fail-open edges before they affected a primary
  update.
- Cache creation now fails before model load if a capability or anchor prefix
  would be shortened, and every trainer independently rejects any shortened
  sample selected by its arm. Per-role truncation counts are receipted. The
  known round-1 route control therefore requires a deterministic full-prefix
  replacement from the already-frozen candidate pool before that control may
  run; the primary manifest/cache remain immutable.
- Cache creation, resume, and training now bind stage, frozen config hash,
  top-k, and exact quick/deep/soup paths plus model-config and merge-receipt
  hashes. Existing round-0/1 caches pass the strengthened validator.
- Fixed a pre-control implementation error: off-policy pressure probes had
  taken the first eight lexicographic units (5/3 and 1/7 capability/anchor in
  actual rounds 0/1). They now use and receipt the registered deterministic
  6-capability/2-anchor geometry. The frozen initial-objective-loss scaling
  definition is unchanged.
- Full-round probe entropy contracted `10.28%` in round 0 and `12.33%` in
  round 1. This is not a registered full-round stop—the 10% entropy ceiling is
  specific to locality—so no post-hoc gate was added. It is preserved as a
  collapse-risk warning for confirmation and final interpretation.
- All 58 experiment tests pass after the safeguards. Seed 42 remains in
  progress; no capability or control comparison exists yet.

## 2026-07-13 — deterministic control-only full-prefix overlay

- Implemented the required route-control recovery without changing the frozen
  primary manifest or target cache. The control path replays the original
  matcher from all 15 hashed candidate artifacts, applies only the registered
  `3,072`-token full-prefix eligibility condition, and fails if the filter
  changes any nonoffending match.
- On the actual round-1 evidence, 490/495 non-deep candidates are eligible and
  the replay changes exactly one of 60 pairs. The cut
  `episode-256a1dbfee96673bcc5a8066` is replaced at the same cache index by the
  first legal exact-cell candidate,
  `episode-2e451ff7c44b165288e7c8f4` (`2,907` tokens, no cut). The other 59
  controls and the 56-exact-cell/4-family-kind geometry are unchanged.
- Only the future `non_advantage_route` arm can consume the derived cache.
  Wrong-teacher, primary, and off-policy paths retain their original inputs.
  The overlay copies 139 samples byte-semantically and scores only the one new
  sample under the same quick/deep/soup target policies; transitive receipts
  bind the source cache, manifest, candidate files, tokenizer, and replacement.
- A no-GPU end-to-end dry run against the real round-1 inputs produced the
  expected sample at cache index 138 and passed both provenance validators.
  Nine focused rematch tests and all 79 current experiment tests pass. Actual
  policy scoring and control training have not run, so no control result exists.

## 2026-07-13 — seed-42 four-round integration passes

- Rounds 2 and 3 completed the frozen seed-42 trajectory. Each of all four
  rounds used three fresh candidate batches, selected exactly 60 deep units and
  20 soup anchors, consumed them once over 20 updates, and passed finite-loss
  plus non-decreasing-overlap gates.
- Deep-route supply by round was 90/81/78/83. Mean corrected losses were
  `0.05669`/`0.04901`/`0.04855`/`0.05404`; probe losses were
  `0.08318→0.05112`, `0.03915→0.02020`, `0.03476→0.01893`, and
  `0.04873→0.02793`. The terminal integration receipt records four completed
  rounds and gate pass; the round-3 merge receipt is
  `88512a57ebb190f0392118a30258eee5fb3bc58d5d34ae04e384afc8842f9122`.
- Probe entropy drops were `10.28%`/`12.33%`/`8.90%`/`11.42%`. The locality
  ceiling is not a registered full-round gate, so no post-hoc stop was added;
  the contractions remain an explicit collapse-risk warning.
- Hardened the later control/confirmation chain without altering frozen
  science: one canonical route-control matcher, atomic score-last external raw
  artifacts, raw-to-score semantic replay, exact full confirmation geometry,
  and raw hashes retained through benchmark execution. An independent audit
  found no remaining blocker; 116 experiment tests and repository CI pass.

## 2026-07-13 — NF4 versus bf16 diagnostic warns against trainer inference

- Ran the preregistered interpretation-only diagnostic after seed 42 and code
  stabilization. It validated all four fixed 6-deep/2-soup probes, exact LoRA
  attachment/replay, 7,970 natural target positions, and unchanged artifacts.
- NF4 objective gain averaged `+0.02191`, but the explicit bf16 merges averaged
  `-0.000224`. Only 15/32 unit gains agreed in sign (`46.88%`), Pearson gain
  correlation was `-0.152`, and mean midpoint update cosine was `0.407`.
- Endpoint top-1 agreement was 31/32, illustrating the trap: endpoint outputs
  can look close while the measured update direction is not. The diagnostic
  has no scientific gate and no downstream authorization. It does not change
  seed 42's registered pass; it makes sealed vLLM procedural confirmation the
  only acceptable capability verdict.

## 2026-07-13 — seed-43 round-1 pass and byte-preserving recovery

- Seed 43 passed round 0 with 20/20 updates, mean corrected loss `0.05638`,
  probe loss `0.08318→0.04706`, and non-decreasing overlap. Round 1 reached
  the exact 60-deep quota after two fresh batches (52 quick diagnostics, 272
  abstentions) and froze 60 deep, 20 soup anchors, and 60 matched controls
  (56 exact-cell, four family/kind).
- After the second batch's student-state receipt was durably complete, the
  first quick-branch subprocess stopped before model construction because a
  concurrent confirmation hardening temporarily removed the legacy
  `_engine_protocol` symbol imported by `branch_states.py`. No branch draw
  began and no state was regenerated. The exact compatibility function was
  restored, direct acquisition-import regressions were added, and the same
  integration command reused every authenticated batch-0/batch-1 byte before
  starting the untouched quick branch.
- Round 1 then passed: 20/20 updates, zero prefix truncation, mean corrected
  loss `0.05172`, probe loss `0.05210→0.02899`, and overlap
  `0.83395→0.83775`. Its merge receipt is
  `2160ecd97c4ee2eda7a29f09048802b953e99d7f026420a653d22c8c009db449`.
  The seed-43 integration receipt remains deliberately in progress at two of
  four rounds; this infrastructure interruption is not a scientific stop and
  no capability conclusion exists yet.

## 2026-07-13 — parameter-control outputs now fail closed

- A pre-control transition audit found that the soup25/50/75 merge receipts
  bound their source adapters and output safetensors, but resume and independent
  authorization did not rehash those output bytes. Corruption could therefore
  weaken a sealed comparator without invalidating the controls receipt.
- Weighted merges now receipt an exhaustive, sorted recursive inventory of
  every regular inference artifact (weights, model configuration, tokenizer,
  and nested load assets), rejecting symlinked or non-regular paths. Both the
  controls runner and the independent authorization path require exact current
  names and hashes plus the frozen model, revision, adapters, mixture weight,
  and merge semantics.
- Regression coverage fails on mutated, missing, extra, and symlinked weight,
  configuration, tokenizer, and nested artifacts, including both resume and
  independent-audit call paths. All 199 experiment tests pass in the pinned
  training environment. No parameter-control output existed when this was
  fixed, so no result or frozen scientific choice changed.

## 2026-07-13 — seed-43 round-2 pass

- Round 2 required all three frozen candidate batches. The first two were only
  four examples short of quota; the final supply was 82 deep, 76 quick, and
  418 abstentions over 576 failed states. The assembler consumed exactly 60
  deep units and 20 soup anchors once each, with 60 matched controls (57
  exact-cell, two family/kind, and one kind/level).
- The all-policy cache bound 140 samples and 35,266 active target positions to
  the exact quick, deep, and soup models. Capability, anchor, and route-control
  prompt truncation were all zero.
- Training completed 20/20 updates with mean corrected top-k loss `0.05588`.
  The held probe improved from `0.04249→0.02281` loss and
  `0.83777→0.84112` top-k overlap, so the frozen round gate passed. Probe
  entropy contracted about `10.66%`; as in prior full rounds, this is retained
  as a collapse-risk warning, not promoted into a post-hoc stop.
- The round-2 bf16 merge receipt is
  `08136cf7e1a8b4b46b8ac6ffae3422c9f38737263fe1d86ab20adf698b0156b6`.
  The tracked seed-43 receipt now records three completed rounds and remains
  deliberately in progress while round 3 runs; no capability conclusion exists.
- Several long generation subprocesses span Git commits because unrelated
  confirmation/control hardening was checkpointed while the GPU remained live.
  The exact vLLM runner hash, package lock, model configuration, engine geometry,
  state/branch hashes, and token ledgers were unchanged and all registered
  engine checks passed. The commit-marker churn is provenance metadata, not a
  generation-protocol change.

## 2026-07-13 — seed-43 four-round integration passes

- Round 3 reached the exact 60-deep quota after two candidate batches, with 43
  quick diagnostics and 281 abstentions. It froze 60 deep, 20 soup anchors,
  and 60 controls (55 exact-cell and five family/kind), with zero prompt
  truncation across every role.
- Training completed 20/20 consume-once updates with mean corrected loss
  `0.05130`. Held-probe loss improved `0.07297→0.04417` and top-50 overlap
  improved `0.83025→0.83288`; the frozen round gate passed. Probe entropy
  contracted `13.11%`, retained as the same non-gating collapse-risk warning.
- Seed 43 is terminal at four completed rounds with gate pass. Deep-route
  supply was 90/60/82/60 and mean losses were
  `0.05638`/`0.05172`/`0.05588`/`0.05130`. The terminal merge receipt is
  `4af497550de22d9bbafdd9de97dd95eabeb6b16b6fa9a7516bf78c4c719d6ecf`.
- The independent manifest→cache→trainer→adapter→merge audit passes across all
  four rounds; the tracked integration receipt hash is
  `885419b1fd8bafb4d16aa56369d91168e0dc0aca92416cc7a4361eeaad2eba11`.
  This supports replication of route supply and optimizer stability only. It
  is not evidence of deployed capability gain; seed 44 and sealed comparison
  remain mandatory.

## 2026-07-14 — seed-44 four-round integration passes

- Seed 44 completed all four frozen integration rounds. Deep-route supply was
  `90`/`82`/`69`/`65`; rounds 0 and 1 required three candidate batches and
  rounds 2 and 3 required two. Every round froze 60 deep capability units, 20
  soup anchors, and 60 disjoint controls. Control matching was
  `57/3`, `56/3/1`, `57/3`, and `54/6` across exact-cell/family-kind/
  kind-level tiers respectively.
- All four rounds completed 20/20 consume-once updates with mean corrected
  losses `0.05674`/`0.05561`/`0.04284`/`0.05147`. Held-probe loss improved in
  every round: `0.08318→0.04540`, `0.05051→0.02258`,
  `0.06825→0.04500`, and `0.05601→0.02952`. Top-50 overlap also increased in
  every round.
- Probe entropy contracted `11.05%`/`9.53%`/`10.62%`/`7.55%`. The registered
  full-round gate still passes, but the round-0 and round-2 contractions retain
  the material non-gating collapse-risk warning; no post-hoc stop is added.
- The terminal merge receipt is
  `33ae673db2abda3bfee69f311f5d5d5b8e1bda29fb2c1b286b3adbe514d4ba00`.
  The independent manifest→cache→trainer→adapter→merge audit passes across
  all four rounds; the tracked integration receipt hash is
  `ff329ebccdc888689b6d6c985a558e66a2385aaa701babf8724d61444428bf1f`.
- Three optimizer seeds now establish safe completion of the registered update
  and replicated fresh strict-deep route supply after round 0. They do not
  establish capability gain, causal routing, superiority to any comparison,
  retention, transfer, or composability; sealed confirmation remains
  mandatory.

## 2026-07-14 — confirmation model-byte boundary hardened before controls

- A fail-closed pre-control review found two related gaps: semantic controls
  authorization did not reauthenticate the frozen quick/deep/soup source
  checkpoints, and the prior confirmation path could recompute and adopt model
  bytes after authorization when it created global `ADMISSION`. No controls
  authorization, admission, or confirmation output existed when found.
- `model_provenance.py` now authenticates the committed source receipt at
  ancestor commit `37dc74ef`, the exact seven-file model root, the exhaustive
  weight/inference inventory, and only the frozen source or local tokenizer
  profile. Symlinked leaves/directories/ancestors, nested extras, non-regular
  files, and mutated/missing/extra config, generation, chat-template,
  tokenizer, receipt, or weight artifacts fail closed.
- The semantic controls authorizer now seals one canonical 13-arm map: three
  sources, three primary seeds, three trained controls, three parameter soups,
  and the exact soup alias under sample-best-of-8 decoding. It recomputes this
  map immediately before no-clobber publication. Confirmation requires exact
  arm-map equality, rehashes all arms immediately before and after global
  `ADMISSION`, and rehashes each selected arm before `STARTED` and after
  generation. Admission is built only from the authorized map.
- The independent integration/control/benchmark audits and benchmark runner
  now use the same model authenticator. Future LoRA merges record the exhaustive
  inference inventory; legacy source and completed primary receipts remain
  admissible only because every byte matches a fixed canonical profile.
- All 209 experiment tests pass, including explicit mutation transitions after
  authorization and after admission. This is evidence-pipeline hardening only;
  it changes no treatment, gate, or capability conclusion.

## 2026-07-14 — non-advantage-route control arm completes (interim)

- The four-round full-prefix non-advantage-route arm completed 20/20
  consume-once updates in every round. Mean corrected losses were
  `0.05393`/`0.05036`/`0.04990`/`0.04619`; every frozen full-round gate passed.
- Held-probe loss improved `0.06090→0.03392`, `0.04644→0.02432`,
  `0.04979→0.02781`, and `0.03674→0.02730`. Top-50 overlap increased in all
  four rounds. Round-0 probe entropy contracted `10.47%`, retained as a
  non-gating collapse-risk warning; later contractions were `7.79%`, `16.34%`,
  and `4.33%`.
- The full-prefix overlay reproduced the original matched mapping byte-for-byte
  in rounds 0, 2, and 3. Round 1 deterministically replaced the sole original
  state that violated zero-truncation eligibility; no training role was
  truncated. The terminal merge receipt is
  `99e4d3258f450173204466bd4a2b4f1dfadfc54d706008e6fc3944a5f7bd57f5`.
- This is an explicit in-progress checkpoint while wrong-teacher, off-policy
  SFT, and parameter controls run. The aggregate tracked controls receipt does
  not yet exist, so this arm is not admitted to confirmation and supports no
  causal or capability conclusion.

## 2026-07-14 — control-runtime recovery before off-policy merge

- All four wrong-teacher training rounds and merges completed. Off-policy SFT
  round 0 then completed 20/20 consume-once updates and passed its frozen round
  gate, but the parent orchestrator stopped before merge: the documented
  `python3` entrypoint ran under the system interpreter while strengthened
  canonical off-policy replay lazily required `transformers` from `.venv`.
- This was an orchestration-environment failure, not a scientific stop. No
  off-policy merge or later round had started, and the complete round-0 adapter
  receipt was preserved for semantic replay rather than retrained.
- Every model stage now re-execs under the pinned `.venv` before importing
  dependency-bearing validators. The guard preserves the exact arguments,
  accepts the normal virtual-environment Python symlink, and fails before stage
  work if the runtime is absent. The README documents this boundary and two
  regression tests cover re-exec and no-re-exec paths.
- All 211 experiment tests pass after the fix. Resumption must first replay and
  validate the existing round-0 receipt under `.venv`; this recovery changes no
  treatment, seed, target, update, or gate.

## 2026-07-14 — all matched controls complete and independently audit

- Resumption replayed the existing off-policy round-0 adapter under `.venv`,
  validated it, and proceeded directly to merge without retraining. The other
  three off-policy rounds then completed normally.
- All 12 trained-control rounds completed 20/20 consume-once updates and passed
  their frozen gates. Non-advantage MOPD mean corrected losses were
  `0.05393`/`0.05036`/`0.04990`/`0.04619`; wrong-teacher losses were
  `0.07040`/`0.06537`/`0.06047`/`0.06949`. Probe loss improved and top-50
  overlap non-decreased in every MOPD control round.
- Off-policy SFT mean CEs were
  `0.10926`/`0.11021`/`0.09851`/`0.09942`; all registered update gates passed
  and probe loss improved in every round. The frozen off-policy gate contains
  no top-k-overlap or CE threshold, so no MOPD criterion was applied post hoc.
- The 25%/50%/75% deep parameter soups each applied 128/128 nonzero modules and
  recorded exhaustive inference-file inventories. Their receipt hashes are
  `aa42d8e67ee87f8cfd937404bf7c43daa5064483185e2fde72c20d3cb2f43d0d`,
  `1599289b9b83932b6d0a5553daa7907ca2b95a22139140f22e5020f0e6b5280a`,
  and `95b607b4e644f2236b1f01498101dac5a274e008ba1c1fae26e0133234feeb1b`.
- The tracked controls receipt passes and hashes to
  `103ef4cc0b24d7c10666b6f0adfcd4dfae4720415c7fbbc76b681ab79162640b`.
  A separate authorization-path audit replayed every canonical ledger and
  reauthenticated all model bytes. This authorizes requesting sealed
  comparison; it is not a capability or causal-routing result.

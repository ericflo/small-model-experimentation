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

## 2026-07-14 — semantic controls authorization sealed

- The no-clobber semantic authorizer independently replayed all prerequisite
  ledgers, reauthenticated every model byte, and sealed the exact 13-arm
  confirmation map at
  `709694b7d770b5cbb09afe8b932bba3891ab4fea39c54c625fc84c5da973072d`.
- The complete control-code inventory hashed to
  `0dff95e5f12847ca36435cad6bcbe752e97a67215968cebb68d52c5cd47efe54`
  both immediately before and after publication. The authorization receipt
  hashes to
  `f4a5456844adeafd39e2e4f2a8036ed9fff2c78830b2eab9d4a7bfa1300d2278`.
- The only licensed downstream action is sealed confirmation evaluation.
  Global `ADMISSION.json` and every confirmation score remain absent at this
  boundary, so this receipt supports no capability, superiority, retention,
  or causal-routing claim.

## 2026-07-14 — sealed confirmation admitted; first arm started

- The live confirmation entrypoint reproduced the no-clobber semantic
  authorization exactly, then independently reauthenticated the full 13-arm
  model map before and after atomic global admission publication.
- `runs/confirmation/ADMISSION.json` hashes to
  `18c019e92fb6b7f7caed0b0f916b958d528b36b9a30607c2890e6b9385d0125d`.
  It binds block seeds `98700`/`98800`, the exact authorized arm map, evaluator
  inventory, configuration, and authorization receipt.
- Block-0 deep evaluation entered `STARTED`; its external journal hashes to
  `c1eb2878bf936d95b62885309257aa1c98e39bbfa9096f3e6a6cf01d3a2a32f1`.
  No generated bundle or score exists at this in-progress checkpoint, so it
  changes no capability, comparison, retention, or routing conclusion.

## 2026-07-14 — attempt-1 deep transaction quarantined before score

- Block-0 deep completed 4,032 atom rows, 384 episode rows through 18 turns,
  and 19 contiguous inference-call bundles. Semantic finalization then failed
  closed before `scores.json` with `confirmation call journal output lacks a
  scored runner field`; the terminal `QUARANTINED` marker hashes to
  `7c93540429dc340c1b62d4cce9e5d2aa82fa3439e615d5ffb3ae4d5300e33580`.
- A key-presence-only audit found 6,879 journal outputs. The 1,392 outputs that
  carried `retained_thinking_token_ids` satisfied the strict schema; the other
  5,487 lacked only that field. `VLLMRunner._ordinary_output` omitted it, while
  `harness._slim` intentionally defaulted the absent optional evidence to
  `[]`. Thus the generated/scored projection and strict journal schema had
  diverged; model outputs were not malformed.
- No prompt, gold, generated text, item score, aggregate score, or performance
  comparison was inspected. The entire transaction, original authorization,
  and admission are archived as attempt 1 and will not be reused. The failure
  receipt hashes to
  `2e645322ead3fbbdf58760849fe17def81fd12b62cdfa4b6c58808e24612ed41`.
- Recovery is limited to making the ordinary runner output satisfy the already
  registered strict schema without changing text or scoring, adding a direct
  regression, issuing fresh no-clobber authorization/admission, and rerunning
  both sealed blocks from an empty current tree. This is evidence-pipeline
  failure, not capability evidence or a scientific stop.

## 2026-07-14 — ordinary-output journal schema repaired

- The strict call-journal validator remains unchanged. The ordinary runner
  path now emits `retained_thinking_token_ids: []`, exactly matching the
  scoring harness's pre-existing projection for a path that never constructs a
  forced-close continuation. The corrected runner hashes to
  `1e065b9c3718e4d2353dc3215928b936ed3fdaeb0d1e04ce96f215d4c9331054`.
- The change does not alter model calls, sampled IDs, decoded text, prompt
  bytes, task seeds, token budgets, answer/action extraction, scores, engine
  geometry, or model artifacts. A direct regression invokes the naturally
  closed budget path and requires both retained and injected evidence fields.
- All 212 experiment tests pass, including the full call-journal semantic
  replay suite. Attempt 1 remains terminal and no archived byte will be reused.
  A fresh no-clobber authorization/admission, bound to this corrected source
  inventory, is required before any GPU rerun.

## 2026-07-14 — frozen recovery-review placement corrected

- The first fresh authorization request stopped immediately, before model
  hashing or output publication, because the post-quarantine review had been
  appended to preregistered `reports/design_review.md`. The frozen-design audit
  correctly rejected its changed hash; no authorization file was created.
- The original design review is restored byte-for-byte at its registered hash
  `cdeaad5b7bb363d0d650af6080d2d0be6c41d0873328e810965d017da271dd5b`.
  The operational analysis now lives in the additive, non-frozen
  `reports/confirmation_recovery_review.md`.
- Repository checks pass after the correction. This was a pre-authorization
  evidence-pipeline stop and changes neither design nor result.

## 2026-07-14 — corrected runtime receives fresh authorization

- Fresh no-clobber semantic authorization passed after independently replaying
  every integration/control ledger and reauthenticating all 13 arm models. The
  arm-map hash remains
  `709694b7d770b5cbb09afe8b932bba3891ab4fea39c54c625fc84c5da973072d`;
  model selection and decoding are unchanged from attempt 1.
- The corrected 56-file control inventory hashes to
  `690a5b5e345f7f4070f731eb69f7a6d9adf7f8148d4b4c07e8234cb8916250ef`
  both before and after authorization and explicitly binds runner hash
  `1e065b9c3718e4d2353dc3215928b936ed3fdaeb0d1e04ce96f215d4c9331054`.
  The new authorization receipt hashes to
  `2b9b86aa76bfb87169a2c70313f967f20c13a09e62fbab25069120e29f0ef9f1`.
- Fresh admission, raw output, and scores remain absent. This authorizes only a
  new empty-tree sealed campaign and contributes no performance evidence.

## 2026-07-14 — attempt-2 confirmation admitted from an empty live tree

- The live confirmation entrypoint reproduced the corrected no-clobber
  authorization exactly, independently reauthenticated all 13 arm models, and
  atomically published fresh `runs/confirmation/ADMISSION.json` at hash
  `6424b68d01420154d10b7a999332eb4b9d44fca3cad63cf7266f098b9d9c990e`.
- At this checkpoint, no external confirmation file, evaluator process, GPU
  allocation, generated row, or score existed. The parent was still performing
  its post-admission arm-byte recheck. This is in-progress provenance evidence
  only and changes no scientific conclusion.
- The post-admission check then passed and block-0 deep entered `STARTED`; the
  external journal hashes to
  `ca44441a2784a79d314b7c359dde6f136617d6bf68301b9dd0741fe3325b8d38`.
  No generated bundle or score existed when this start marker was recorded.

## 2026-07-14 — attempt-2 block-0 deep completes and validates

- Block-0 deep generated 4,032 atom rows and 384 episode rows through all 18
  turns. The generated journal hashes to
  `adc322ff06f1c585653467c7f8b745e6bc89aad5103dbd31e29312cbad8ac09c`;
  atom and episode raw files hash to
  `80fe4abcb8f09fff268f514e4b9fc2cfb3718d566b537988a2fab50d46339754`
  and `2e5b0b467e9a543d468fc7759675c022b6c4b29cbd5cdf6e8109335003a8c150`.
- The unchanged strict call-journal validator passed, directly resolving the
  attempt-1 evidence-pipeline question. `COMPLETE.json` hashes to
  `d029202d1f11626bdf76ef6f6607b5e7033fc58584d852a83cc30da48dc6663a`
  and binds score hash
  `f6d7c0271aa2d7acdc04e1e62e54dfb79d7a49f244fe332e898b4b813937d1e6`.
- The parent accepted registered aggregates `0.580068` (`n=3,072`) on the deep
  stratum and `0.810029` (`n=1,344`) on the quick stratum, with 4,261,790
  sampled tokens. This is a single source-arm anchor, not treatment evidence;
  block-0 non-advantage evaluation started next with external journal hash
  `b5b1a6c6b19d95630c066ba0767fbaa62f8627836a03deb8384ab71003c7286a`.

## 2026-07-14 — block-0 non-advantage control completes

- The non-advantage control completed the same 4,032-atom/384-episode sealed
  geometry and passed strict journal validation. `GENERATED.json` and
  `COMPLETE.json` hash to
  `42b5c16414a6d3eb1c013990ea47cae02c132c3b983908023bccf107298eb947`
  and `c614b53218d0c46e89e9ebc185441b7bea6af5c6b364d8a8eda55acef79e0f3d`;
  the score hashes to
  `d3e7b57fb437c3ca4194b74f8e7f4fa63ab067d50e62188de8e07d6f6d7ba7d8`.
- Registered means are `0.565173` deep and `0.790951` quick, respectively
  `−0.014895` and `−0.019078` versus the block-0 deep source anchor, with
  4,468,793 sampled tokens. This is a one-block control/source comparison, not
  treatment evidence or a causal advantage-routing result. Off-policy SFT
  evaluation started next with external journal hash
  `fbd64d2cd877e2e84a811337541e1d60de63a3d680ad8fad67fcf785e9db4e2e`.

## 2026-07-14 — block-0 off-policy SFT control completes

- Off-policy SFT completed the sealed geometry and strict journal validation.
  `GENERATED.json` and `COMPLETE.json` hash to
  `f865801c755fb2fc1ba2ff81d1dd4f780d432b99116b69986ecf06d535a8f8ae`
  and `b23d72356ab3cedf7794462d6bc6d403705eaa1b964eda4893ec263e5b5195de`;
  the score hashes to
  `44e8e830e0a05c5d903b0638fcc756ea19203b80044b26ce70948b5d6c5ae002`.
- Registered means are `0.577559` deep and `0.797585` quick, `−0.002509` and
  `−0.012444` versus the deep source anchor, using 4,342,736 sampled tokens.
  It is `+0.012386` deep and `+0.006634` quick versus non-advantage, but remains
  one control on one block; treatment and replication conclusions are sealed.
  Primary seed 42 started next with external journal hash
  `fc7fd5c948893412f06d181a647a45ff9b183b5f833d291089b046096acc9ee4`.

## 2026-07-14 — block-0 primary seed 42 completes below source

- Primary seed 42 completed the sealed geometry and strict validation.
  `GENERATED.json` and `COMPLETE.json` hash to
  `a3ecc145f38c4413e27bd1f3b56bc868836129069dd647b30b6bdafd96159caf`
  and `104601605d2ad0b5c12f41947242989c2cfdd11d24dedd954abc0134cf955274`;
  the score hashes to
  `585f99d71141a70de1db609b827b767218068645d31db0f7396324152c232841`.
- Registered means are `0.577005` deep and `0.795225` quick, `−0.003063` and
  `−0.014804` versus the deep source anchor, using 4,422,572 sampled tokens.
  It is also `−0.000554` deep and `−0.002360` quick versus off-policy. This is
  an unfavorable first treatment block, not a terminal conclusion: the frozen
  decision requires all treatment seeds, comparators, and both blocks. Primary
  seed 43 started next with external journal hash
  `81047c1f3329eb71fd76dfa47a4a2c6dd87627c2bcc227761567b155d400d3ef`.

## 2026-07-14 — block-0 primary seed 43 repeats negative deep sign

- Primary seed 43 completed sealed generation and strict validation.
  `GENERATED.json` and `COMPLETE.json` hash to
  `32c006cfa1e970cf845b2b1d8af23d606a314f1ee09aeba49a1466a0721a2d95`
  and `44c7f6c239d00fb512f3c474ca20f711a7d85f39b3fd7f1607729a97caaa36bb`;
  the score hashes to
  `9ebf64444ce068a14de943d2c3c28c34e2dc567a23197e7307c93a029d42111e`.
- Registered means are `0.576097` deep and `0.804167` quick, `−0.003971` and
  `−0.005862` versus the source, using 4,408,848 sampled tokens. The deep sign
  repeats seed 42's unfavorable result; seed 44 and block 1 remain required.
  Primary seed 44 started next with external journal hash
  `a607b9b93f7c629aff92083f308c53cb0058e353f00a4ee7f2603bcd4e4082e2`.

## 2026-07-14 — all three block-0 treatment seeds are below deep

- Primary seed 44 completed sealed generation and strict validation.
  `GENERATED.json` and `COMPLETE.json` hash to
  `49ca7c8cfb9597b2b8ec8b047a82ce9de321f11cf323d695ff7d068d44b0c1af`
  and `2cb75d45ba30974e3457212001377804a951cb422ad2f62951326e752b8b9dc9`;
  the score hashes to
  `cf3021207260a30357df7dc52d74c1b75fb81a315a04d22fb8424f03dfff4a45`.
- Registered means are `0.572010` deep and `0.805422` quick, `−0.008058` and
  `−0.004607` versus the source, using 4,393,570 sampled tokens. Seeds 42, 43,
  and 44 all have negative block-0 deep signs, so the frozen success condition
  is already unreachable. The preregistered campaign nevertheless requires all
  arms and both blocks for the terminal control ranking and receipt.
- The block-0 quick source evaluation started next with external journal hash
  `0ab077980fc97ee8cc5481575a0059896f686735539903ea6634cb15f59b154d`.

## 2026-07-14 — block-0 deep source also dominates quick source

- The quick source completed sealed generation and strict validation.
  `GENERATED.json` and `COMPLETE.json` hash to
  `8b826d1202add12808efd9951f964598b2f01bfe2775ae23b2abb6fe64621185`
  and `a51a444f0d467dfa6005fda65ddb80c62acfdb7e75acdd3c64cde3c488a2fb80`;
  the score hashes to
  `089bbdb35d9749cece3fe23bead1f3da25149c5409994d2ca1c64de4364ab08b`.
- Registered means are `0.529284` deep and `0.775684` quick, `−0.050784` and
  `−0.034344` versus the deep source, using 4,773,735 sampled tokens. The deep
  source is therefore the better source on both block-0 strata. All three MOPD
  seeds lie between the two sources but fail to improve the better source.
- The frozen soup comparator started next with external journal hash
  `b2ca6c3aeb2348510d78c99ba7efefe417684a14e5ae946e5a11316fbedb699d`.

## 2026-07-14 — block-0 MOPD does not dominate its soup initialization

- The immutable 40/60 soup completed sealed generation and strict validation.
  `GENERATED.json` and `COMPLETE.json` hash to
  `243d7fb8d00b0ccb430811cdb9017650395139dcdf0504eb630716adc1728da8`
  and `742f388e187847f1e74654a855cd3529cc6145cf392f8aa676dc63b8942251ca`;
  the score hashes to
  `57dce804f32f7e0c2c0c18b23cb170075698eb9216587a34ce8818cbc7eb090c`.
- Registered means are `0.571744` deep and `0.811128` quick, `−0.008324` and
  `+0.001099` versus the deep source, using 4,392,180 sampled tokens. Primary
  seeds 42/43/44 are only `+0.005261`/`+0.004353`/`+0.000266` above soup on
  deep and are `−0.015903`/`−0.006961`/`−0.005706` below it on quick. The
  treatment does not dominate its own initialization on this block.
- Soup25 started next with external journal hash
  `4bd39562939e2d3b726ce5221e22370fed9f2accdb6ddee8b702a9c558a8c2b6`.

## 2026-07-14 — block-0 soup25 is dominated by the initialization

- Soup25 completed sealed generation and strict validation. `GENERATED.json`
  and `COMPLETE.json` hash to
  `777084b5a15287b83302d20a6b40ba828c4a72f9276056e3eebf6d1ca04b065a`
  and `f98a8d5e6581253c68b2e900f23cb53d1e740aff27ef9181ef6729fe842e0a28`;
  the score hashes to
  `574617cd1646ec8d00f7cb3d6861d3c89a6a892ecb454d2c79d08d2069bf3487`.
- Registered means are `0.543311` deep and `0.801212` quick, `−0.036757` and
  `−0.008817` versus the deep source, using 4,693,690 sampled tokens. Soup25
  is also `−0.028434` deep and `−0.009916` quick versus the immutable 40/60
  initialization, so it is dominated on both strata.
- Soup50 started next with external journal hash
  `b55a6675fea796006dc99e44bf7b24f42b7344f6ceae11d163c568d260fd544d`.

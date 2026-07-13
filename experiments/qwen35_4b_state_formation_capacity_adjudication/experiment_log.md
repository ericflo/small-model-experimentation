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

## Current authorization

Documentation, adversarial review, CPU tests, and implementation review are authorized. Model-bearing
execution remains unauthorized until the design receipt is frozen and all setup prerequisites in
`docs/gpu_runbook.md` pass. Runtime-bound hashes are intentionally absent because their artifacts have
not been created.

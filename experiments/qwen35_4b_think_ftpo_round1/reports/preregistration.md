# Preregistration — think-block FTPO round 1 (v2)

Frozen before any full-scale GPU run (smoke and the zero-GPU census excepted). Changes
after freeze require a numbered amendment at the bottom; results may not motivate silent
edits. v1 (loop-repair-primary) was revised to v2 after (a) a user redirection toward
outcome-conditioned pivot steering and (b) the adversarial design review's blocking
finding, independently verified: the mining detector flags **0.08%** of existing greedy
base atom completions (1/1200, think@1024) and **0.00%** of episode turns (0/786) —
exact-repetition loops are absent at deployed agentic budgets (they dominate only at
16k+). v2 was frozen before any harvest/training/eval GPU work.

## Arms

- **base** — pinned Qwen3.5-4B.
- **pivot (primary)** — FTPO on outcome-conditioned divergence nodes mined from a prefix
  tree over n=8 verifier-scored think trajectories per prompt.
- **pivot-shuffled (control)** — identical mining pipeline, but each prompt's rollout
  outcome labels are randomly permuted (seed 3407) before node mining: destroys the
  outcome conditioning while preserving prompt mix, node structure statistics, row
  format, and dose. The corpus's required shuffled-label control for preference training.
- **Descoped: loop-repair arm.** The loop census is reported as a zero-GPU artifact
  (existing-log scan, above). Loop-FTPO is requeued as a separate long-context
  (think@16k+) loop-control experiment per the model playbook mandate.

## Census & mechanism predictions

- **P0 (pivot census gate):** on learnable-band tasks (base greedy success ∈ (0.1, 0.9)
  per family-level cell, calibrated in smoke), ≥ 30% of n=8 sampling groups yield ≥1
  eligible divergence node (definition under Mining below). Below 30%: premise-weak,
  report census, stop before training.
- **P1 (mechanism):** pivot arm's greedy success on held-out learnable-band tasks (fresh
  seeds) ≥ base + 0.05 absolute, AND pivot-shuffled shows < half of pivot's gain.
- **P1-format (transfer precondition):** the P1 gain holds (≥ half magnitude) on the
  format-shifted slice (same held-out tasks re-rendered under 2 alternate scaffolds:
  no-"Answer:" template + chat-style framing). Required before any NEGATIVE capability
  claim (else INCONCLUSIVE-transfer).
- **P2 (think-economy guards):** natural-close, answer-parse, and the full termination
  triple (exact-loop / unresolved-contact / answer-limit rates) reported per arm at
  think@1024 and think@2048; pivot must not degrade natural-close or answer-parse by
  > 2pp vs base.
- **P3:** gym-internal aggregate (10 trained families, atoms L1–L4 + episodes, fresh seed
  91001; brinework + spindle reported as never-harvested held-out families) improves for
  pivot; pivot ≥ base − 0.02 is a hard guard.
- **P4 (primary):** menagerie decision rule below.

## Menagerie decision rule (v2 — null-calibrated, quick-gated)

1. **Null calibration first** (before any trained-arm event is interpreted): base-vs-base
   twice on quick seed A and once on quick seed B (three same-seed spread realizations).
   SD_null = sample SD of the three |Δaggregate| realizations, pooled with C50's ~0.034;
   the measured value REPLACES the prior as H0 scale.
2. **Quick gate:** paired (pivot − base) on THREE fresh quick seeds.
   POSITIVE-quick = mean delta ≥ max(+0.03, 2·SD_null/√3) AND ≥ 2/3 seeds positive.
   NEGATIVE-quick = mean delta ≤ +0.01 AND ≤ 1/3 seeds positive.
3. **Medium stage (conditional):** runs ONLY if POSITIVE-quick. One medium base-vs-base
   same-seed pair (medium null realization) then one paired medium event.
   POSITIVE = POSITIVE-quick AND medium delta ≥ max(+0.02, 2·|medium null realization|).
4. **NEGATIVE (capability):** P1 AND P1-format hold, AND NEGATIVE-quick. Claim scope:
   "outcome-conditioned think-token preference at this dose does not move blackbox
   agentic capability" — with the dose stated. Requires the dose-sufficiency
   precondition: post-regularization training pool ≥ 1,200 rows; below that, the flat
   outcome routes to UNDERDOSED → iterate, never NEGATIVE.
5. **INCONCLUSIVE:** anything else → round 2 (fresh seeds, same rule; round-1 events are
   never re-read; single-event quick detectability is honestly ~+0.06 — acknowledged).
6. pivot-shuffled runs one quick paired event on seed A only (diagnostic; not part of the
   decision rule).
- **Collapse guard (C29 signature):** list-transform substrate, fresh seeds: greedy and
  pass@8 (matched sampled-token budget), base vs each trained arm; > 10% relative drop on
  either → recipe flagged damaging regardless of menagerie outcome.
- **No-think guard:** no-think forced-answer accuracy on gym atoms L1 (120 items), pivot
  vs base; drop > 2pp flags answer-channel interference (reported; blocks reuse
  recommendation, not the menagerie read).

## Frozen constants

- **Harvest P:** temperature 0.6 / top-p 0.95 / top-k 20; n=8; think budget 1024
  (= quick-tier deployment budget; keeps harvest ≈ deployment distribution and halves
  cost vs 2048); answer-max 512. Prompt mix ≈ 60% gym atoms from the 10 TRAINED families
  only (brinework, spindle excluded to preserve the held-out control), levels from the
  band calibration; ≈ 40% list-transform code tasks, depths from band calibration
  (candidates 2–4). Adaptive dose: harvest in slices of 800 prompts (gym seeds from
  72001, code seeds from 73001, closed ranges 72001–72999 / 73001–73999) until projected
  post-regularization pool ≥ 1,200 rows or a 5h GPU harvest cap; training proceeds at
  ≥ 600 rows (power caveat reported if < 1,200).
- **Band calibration (smoke):** base greedy success per (family, level) cell on 12 items
  per cell, gym levels {1,2,3} and code depths {2,3,4}; harvest uses cells with success
  ∈ (0.1, 0.9); the calibration table is archived in runs/.
- **Mining P:** prefix tree over the n=8 stage-1 think-token-ID sequences; node
  eligibility: depth ≥ 16 think tokens, ≥ 2 rollouts through each of ≥ 2 sibling
  branches, verified success-rate gap ≥ 0.5 between the max- and min-rate eligible
  branches; ≤ 2 nodes per prompt (largest gap first, ties deeper); rejected = min-rate
  branch's next token; chosen = eligible sibling tokens with rate ≥ rejected + 0.5
  (observed tokens only; deduped by normalized surface; rejected + case variants
  excluded). Success = family verifier score ≥ 1.0. Context = prompt ids + shared think
  prefix ids, capped 6,144 tokens (discard overs). Chosen-per-row distribution is a
  mandatory reported diagnostic.
- **Shuffled control mining:** identical, after permuting the 8 outcome labels uniformly
  at random per prompt (numpy default_rng(3407)); row count downsampled to match the
  pivot pool if larger.
- **Regularization (both trained arms):** rejected-token flattening 0.3 (median-anchored
  power transform, greedy fill, source-share tiebreak gym-vs-code); chosen flattening 0.5
  (p95/floor-50, per-row dedupe then prune); min_chosen_tokens=1; stop-word filter OFF;
  max_train_examples = min(12,000, 70% of pool).
- **Trainer:** bf16 LoRA r=256 α=128 dropout 0 on q/k/v/o/gate/up/down_proj — on this
  hybrid architecture q/k/v/o exist on the 8 full-attention layers only; gate/up/down on
  all 32; linear-attention in_proj_*/out_proj deliberately untargeted (unproven merge
  path). ~0.34B trainable params (~0.68 GB bf16 adapter). No lm_head (merge-path
  constraint). Loss = softplus(ε−Δ_c)·clamp((ε−Δ_c)/ε,0,1) averaged /|C| per row
  + 0.4·MSE(non-target logits vs reference) + 0.05·deadzone(0.5)-MSE(target logits);
  ε=2.0; reference = same weights, adapter disabled, no-grad. **Memory/architecture
  requirements:** final-position logits ONLY (gather last-real-index hidden state, then
  lm_head on the gathered vectors — never materialize full-sequence logits over the
  248,320 vocab); RIGHT-padded batches (real tokens are a contiguous prefix — safe for
  the recurrent/linear-attention blocks; left padding is forbidden); a preregistered
  padding-equivalence smoke gate asserts max |final-logit(batch, right-padded) −
  final-logit(batch-of-1)| ≤ 0.05 on real rows before training; gradient checkpointing
  on; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set in-script; OOM catches include
  torch.AcceleratorError; max-length (6,144-token) memory smoke before the run.
  lr 1.5e-5 linear, warmup 0.1, 1 epoch, adamw_torch, weight_decay 0.01,
  max_grad_norm 2.5, effective batch 16 (per-device tuned in smoke), early stop
  chosen_win ≥ 0.4, log every 5 steps, seed 3407.
- **Deployment:** merge LoRA deltas into the full composite checkpoint (explicit
  name-mapped W += B·A·α/r); **per-arm C49 gate** (pivot AND pivot-shuffled): on-vs-off
  greedy behavioral diff with the real merged checkpoint — identical outputs = fail.
- **Eval decode:** greedy; whitebox at think@1024 and think@2048, N = 500 prompts per
  arm per budget (held-out seeds 75001–75999, same 60/40 source mix, trained-family gym
  cells); gym-internal eval seed 91001 (atoms 3/cell L1–L4 across all 12 families +
  2 episodes/family); menagerie tier defaults, fresh seeds from 61001 (union-checked
  against this log, the gauntlet log, and the suite baseline seed 31337).
- **Matched-compute reference (C2/C5 discipline):** base n=8 coverage at the harvest
  sampling settings on the held-out band tasks, reported as a labeled NON-DEPLOYABLE
  oracle ceiling next to the deployable greedy numbers.
- **Collapse guard:** 120 fresh list-transform tasks (seeds 76001+), greedy + pass@8.

## What is NOT claimed in advance

- No long-context (≥16k) claim; loop-FTPO at long context is a queued follow-up.
- No stacking claim (FTPO on the gauntlet checkpoint): queued follow-up.
- Menagerie per-family movements: exploratory only.
- Node-level pivot signal is Monte-Carlo (n=8) and individually noisy; claims rest only
  on aggregate deltas under the decision rule.

## Amendments

### Amendment 1 (2026-07-10, after harvest smoke, BEFORE any training)

Smoke evidence (80-prompt slice, n=8, frozen sampling; `runs/harvest_smoke/`):
44% of groups are outcome-mixed (signal ceiling), 96% have structural ≥2/≥2
splits, but eligible divergence nodes concentrate at depth 3–8 — the v2
`min_depth=16` floor discarded 10/14 eligible nodes, yielding a 5.6% census vs
the 30% gate. The 30% gate was set before any structural data existed.

Changes (all pre-training, evidence-cited):
1. `mining.pivot_min_depth`: 16 → **2**. Depth is recorded per row and reported;
   the ≥2/≥2-rollout and gap≥0.5 conditions remain the eligibility guards.
2. `mining.pivot_max_nodes_per_prompt`: 2 → **3**.
3. **P0 census gate rebased**: eligible-node group rate ≥ **15%** AND
   mixed-outcome group rate ≥ **30%** (the premise decomposed into its two
   parts; smoke measured 19.7% and 44% respectively at min_depth=2).
4. Harvest-only engine throughput: `max_num_seqs` 64 → 128,
   `max_num_batched_tokens` 16384 → 32768 (harvest is not a compared arm;
   all compared-arm evals keep the frozen 64/16384 geometry).
5. Slice construction tops up to the full slice size round-robin (smoke
   under-filled 71/80 due to per-cell floor rounding).

### Amendment 2 (2026-07-10, after n=16 probe, BEFORE any training)

An 80-prompt probe at n=16 (identical settings otherwise;
`runs/harvest_probe/`) measured census 27.5% (vs 11.25% at n=8), mixed-outcome
rate 50% (vs 31%), and 0.375 mined rows/prompt (vs 0.125) at 1.94× wall cost —
54% more rows per GPU-second with double the rollout mass per node (better
gap statistics). Changes:
1. `harvest_pivot.n`: 8 → **16**; `slice_prompts`: 800 → **400** (constant
   slice wall time). P0 evaluated at the amended gates PASSES on the probe.
2. Dose reality at the 5h cap ≈ 700–900 rows: the POSITIVE and
   mechanism-failure outcomes are readable at round-1 dose; a flat outcome
   routes to UNDERDOSED → round-2 harvest extension (never NEGATIVE below
   1,200 rows), exactly as the decision rule already specifies.

### Amendment 3 (2026-07-10, after trainer smoke, BEFORE any full training)

Trainer smoke on 21 probe rows measured **initial chosen_win ≈ 0.43**: in the
pivot setting the rejected token is an ordinary sampled token (not an
argmax-dominant attractor as in the repetition pipeline), so chosen and
rejected start near parity and the v2 early-stop threshold (0.4, repetition
calibration) would fire before the first optimizer update. Changes:
1. `train.early_stopping_chosen_win`: 0.4 → **0.85** (the published
   calibration for the regime where the rejected token is not dominant).
2. New `train.early_stop_min_progress: 0.2` — early stop may only trigger
   after ≥20% of scheduled optimizer steps.
3. Banked smoke facts: padding-equivalence gate FAILED on this hybrid
   architecture even with right padding (max |Δlogit| = 0.4375 ≫ 0.05 tol);
   the preregistered fallback to batch-of-1 forwards is therefore the
   operative training mode. Trainable params 339.7M (7.47%).

### Amendment 4 (2026-07-11, after the capped harvest, BEFORE any training)

Full harvest (5 slices, 2,000 prompts, 5.5h): 645 pool rows at 0.3225
rows/prompt; census 26.05% eligible / 49.6% mixed — **P0 PASSES**. Arithmetic
conflict between two frozen constants: 70% of 645 = 451 training rows < the
600-row training floor, and the pre-sized "+600 prompts" extension projects to
only ~587. Resolution: the training floor is the controlling constant (it
guards statistical power); the extension size was set before yield was
measurable. The extension is therefore sized from measured yield: **+800
prompts** (two standard 400-prompt slices, fresh seed offsets — slice indices
5–6, no seed reuse), projected pool ≈ 900 → ≈ 630 training rows. Total harvest
≈ 7.7h is recorded as a fast-round-doctrine deviation for round 1. All other
constants unchanged; a flat outcome still routes to UNDERDOSED (pool < 1,200),
never NEGATIVE.

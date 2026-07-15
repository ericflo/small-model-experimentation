# Preregistration: Fresh-Surface Budget-Commit Universal Curriculum

Frozen before any model event. Every seed, bar, quota, and rule below is fixed; a
failed gate is a preserved result, never permission to change this contract inside
this experiment directory.

## Frozen identities

- Experiment: `qwen35_4b_universal_fresh_surface_budget_commit_target_match`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent (baseline and warm start): the authenticated explicit composite
  `replay_after_close` from `qwen35_4b_universal_on_policy_prefix_repair_token_match`;
  merged weights sha256 `7ab4c419f70135d3fe058dba6e79e3a9a61c6661d43e6acb9662f331efe36e2e`,
  adapter weights sha256 `bb59d3bd9273ae3bb3dffe54e983590dada69e6e1bdba571009ffedbba05154d`.
  Runtime LoRA is forbidden; every evaluated arm is an explicit merged composite.
- Arm labels: active control `replay_repeat`; candidates `designed_fresh` (arm D) and
  `budget_commit` (arm B); parent evaluation label `replay_after_close_parent`.
- Seeds: construction/slot-match/training/local/aggregate =
  `77116 / 55117 / 51 / 88013 / 78143`. None may change after its event; 78,143 stays
  sealed until local promotion.

## Frozen treatment corpora

- Generator: `scripts/gen_fresh_curriculum.py` — the thirteen predecessor lesson
  constructors rendered over six fresh surface pools (greek, elements, animals,
  ordinals, gems, digraphs), fresh separators, fresh record attributes
  (heft/shine/reach/pulse), fresh routing capabilities (audio/ledger/cipher/relay/vault),
  plus the new `u_budget` lesson: ordered scan under a hard check allowance, stop at
  the first hit, mandatory `BUDGET` commit on exhaustion, and — on every exhaustion
  task — a satisfier planted immediately past the cutoff so violating the allowance
  yields a parseable wrong answer.
- Arm D: 160 rows at the frozen designed160 quotas
  (`induct=16,execute=12,select=10,trace=12,verify=12,count=6,repair=18,optimize=14,abstain=14,state=16,order=10,probe=10,route=10`).
- Arm B: a deterministic largest-remainder 120-row subset of arm D
  (`induct=12,execute=9,select=7,trace=9,verify=9,count=5,repair=13,optimize=11,abstain=11,state=12,order=8,probe=7,route=7`)
  plus 40 budget lessons from the same construction seed; the arms differ by exactly
  that 40-row substitution.
- Frozen hashes: arm D `e599f1563f6e4fe68aa43ce64bbe450c264170fb07dc17dba9ea74e694f284d5`;
  arm B `ecece8e294f0b1c34a086705b05773d6004c6a9a239283aa26ee1c1bbad39800`;
  replay pool `sft_blend.jsonl` `25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2`
  (byte-identical to every predecessor's copy).
- Audits: executable truth for every row; induction depth/identifiability/dead-end
  audits unchanged; budget outcome balance within 14–30 hits of 40; banned-vocabulary
  scan proves zero occurrences of the predecessors' multi-character alphabetic
  surface tokens (colors, syllables, multi-character romans), predecessor record
  attributes, predecessor routing capabilities, gym replay family names, and public
  benchmark family names. Scope, stated exactly: single-character letters, roman
  I/V/X, and two-digit numerals are exempt because they occur as arithmetic values
  and English fragments, not as surface items; the word "table" is exempt as shared
  template wording ("in table order") present in every predecessor corpus. None of
  the exempt tokens can distinguish treatment from control, which share the entire
  replay mass.

## Frozen exposure match

- Streams: 1,280 shared replay rows (stratified by family and kind, deterministic
  largest-remainder + sha256 ranking under namespace seed 55,117), position-aligned
  and byte-identical across all three arm files, plus one 240-row variable block per
  arm at identical slot positions: control = 240 replay rows; arm D = 160 designed
  rows + 80 replay fillers; arm B = 160 arm-B rows + 80 replay fillers.
- The three variable blocks must match EXACTLY (limit 0) on all three axes measured
  by the training encoder: forward tokens, nonzero loss-bearing targets, and absolute
  loss mass (in fifths). Filler, control, and core sets are pairwise disjoint. The
  joint selection is one MILP (HiGHS, zero relative gap); infeasibility is the
  terminal outcome `STOP_EXPOSURE_MATCH_INFEASIBLE` — no relaxation, no cardinality
  change, no axis dropping.
- Targets are never modified, duplicated, or truncated to force the match; zero
  encoder-skipped rows are required in every arm.

## Frozen training event

One training stage per arm, control first, each behind a clean pushed green
checkpoint: 1 epoch over 1,520 rows, batch 1, gradient accumulation 8 (190 optimizer
steps), LR 1e-5, cosine schedule, warmup ratio 0.03, LoRA rank 32 alpha 64 dropout
0.05 on the seven projection modules, think/close loss weights 0.2/0.2, answer weight
1.0, max length 4,096, training seed 51, warm start from the parent adapter continued
in place. The trainer log must show exactly 1,520 examples and 0 skipped. Train loss
is never capability evidence.

## Frozen local gate

- Tasks: fresh seed 88,013, 8 tasks per each of the 13 skills (104 rows) generated by
  the ORIGINAL-surface generator (`scripts/gen_curriculum.py`, byte-identical to the
  predecessors). Training renders only fresh surfaces, so the gate is itself a
  surface-transfer test. A frozen overlap receipt proves zero canonical user-message
  overlap against all three training streams, both fresh corpora, the replay pool,
  and the regenerated 26-row task sets of local seeds 88,000–88,012.
- One evaluation event: the parent composite and the three newly merged arm
  composites, sequential vLLM engine runs on the pinned geometry (greedy, natural
  thinking, 1,024-token cap, max-model-len 4,096, gpu-mem 0.90, max-num-seqs 16,
  max-num-batched-tokens 8,192, CUDA graph sizes 1/2/4/8/16), input the oracle-free
  frozen task file.
- Absolute bars per candidate: parsed ≥ 96/104; correct ≥ 68/104; cap contacts ≤ 8;
  feasible-route abstentions ≤ 4; `u_execute`, `u_induct`, `u_probe` correct ≥ 4/8
  each. Relative bars per candidate: total correct strictly greater than BOTH
  `replay_after_close_parent` and `replay_repeat`, and correct on the 24
  execute+induct+probe rows strictly greater than BOTH. Ties fail.
- Promotion selects at most one candidate among those passing every bar: higher total
  correct, then higher 24-row target subtotal, then fewer cap contacts, then
  `budget_commit`. If no candidate passes, aggregate seed 78,143 remains sealed
  permanently and the experiment closes as a local negative.

## Frozen conditional aggregate pilot

- One aggregate-only event through the trusted gateway: tier quick, think budget
  1,024, fresh sealed seed 78,143, canonical `qwen_vllm` backend, four explicit
  merged models on the same seed: `base` (the reserialized base composite), `parent`
  (`replay_after_close`), `replay_repeat`, and the promoted candidate. All arms must
  see an identical benchmark source inventory hash. The seed is recorded in a ledger
  and never reused.
- Promotion gates, all required: candidate aggregate strictly greater than base
  aggregate; every one of the ten public family scores strictly greater than base's
  corresponding family score; candidate aggregate strictly greater than
  `replay_repeat` aggregate; candidate aggregate strictly greater than parent
  aggregate.
- Deviation from the residual template, declared prospectively: the every-family
  strictness is measured against BASE (the goal object: an installed model that
  improves every reported family over the uninstalled model), while parent and replay
  must be beaten on aggregate. Demanding strict family wins against the parent would
  make family-level ties at a shared ceiling fatal and tests a different claim.
- A pass is a pilot only. A universal claim additionally requires, in successor
  experiments: independent fresh quick seeds, the medium tier, paired uncertainty,
  and a same-backend matched-compute sample-more baseline.

## Mandatory checkpoint order

1. Model-free construction (corpora, exposure match, local gate design, design
   receipt, adversarial design review) — committed, pushed, green.
2. `train-control`, then 3. `train-designed`, then 4. `train-budget` — each stage
   requires the previous stage's receipt committed at a clean pushed green `main`,
   plus `reports/compute_review.md` verdict `PASS_CONTROL_TRAINING` before stage 2.
5. `merge-arms` — requires `reports/local_design_review.md` verdict
   `PASS_CONTROL_MERGE`.
6. `local` — one event; then 7. conditional `benchmark` only on promotion.

Every receipt binds runner inputs, artifact hashes, git state, and environment; a
dirty preflight aborts; no stage may re-run its model event.

## Interpretation limits

The package-level causal unit is the whole arm (content plus exposure); per-lesson
attribution requires successors. The local suite is a mechanism screen, not the
claim. Benchmark firewall: `benchmarks/` content is never read; only the gateway's
aggregate and public per-family scores are consumed.

Prospective power statements, frozen before any event:

- The every-family-strictly-above-base pilot gate is one-sided conservative and has
  very low power at the quick tier: family scores move in ~1/8 steps, base scores
  exactly zero on several families, and no historical arm — including deltas 4–8×
  this trial's plausible effect — has ever passed it. A pilot PASS is therefore
  extremely strong evidence; a pilot FAIL is the expected outcome even under the
  hypothesis and must be recorded as "not confirmed at quick-tier granularity",
  never as evidence against the surface-general designed dose or against
  universality. The family-level claim, if the aggregate wins hold, moves to
  higher-power tiers in a result-separated successor.
- The surface-generality question (arm D's headline) is read prospectively from the
  local receipt's accuracy comparisons (total correct and the 24-row
  execute+induct+probe subtotal versus parent and replay), whether or not promotion
  passes: promotion additionally demands termination behavior (parse/caps bars)
  that the mid-density predecessor never exhibited, which is arm B's hypothesis.
  Promotion bars stay exactly as frozen; this paragraph only fixes how a
  non-promoted outcome may be described.
- A base family score of 1.0 at seed 78,143 would make the every-family gate
  structurally unpassable; if observed, the event is still recorded as frozen and
  the structural impossibility is noted alongside the outcome.
- The absolute-loss-mass axis is matched as an encoder-measured sum; the trainer
  normalizes loss per row, so matched total mass does not linearly control gradient
  magnitude. Row counts, update counts, and the other two axes are matched
  simultaneously, which bounds this caveat.

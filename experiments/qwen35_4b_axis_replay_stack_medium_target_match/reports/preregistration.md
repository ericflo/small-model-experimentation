# Preregistration: Axis-on-Replay Stack with Medium Pilot

Frozen before any model event. A failed gate is a preserved result, never
permission to change this contract inside this directory.

## Frozen identities

- Experiment: `qwen35_4b_axis_replay_stack_medium_target_match`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent (baseline and warm start): the `replay_repeat` arm of
  `qwen35_4b_goal_gap_axis_curriculum_target_match` — merged weights
  `3df45004fcf42519ce28cdcfedcbb39b0907662f8ecfb8a87b13b416087d0072`, tree
  `4c4f3561efbcafe1b9f777f4bd21bf4949ff89177f77946d0fa0f88cafafacd7`, warm-start
  adapter weights `20be87b5c7a7969d006b2825d3937b10fd0627ea2358af02879451039a07cd36`,
  adapter config `bf5ade0b3328489d5ba676aa497e311d9883f70908cf56bda69f73882e232bac`.
  Runtime LoRA forbidden; every evaluated arm is an explicit merged composite.
- Arms: control `replay_squared`; candidate `axis_on_replay`; parent evaluation
  label `replay_parent`.
- Seeds: inherited corpus construction 77,117 (byte-identical); fresh
  slot-match/training/gate/aggregate = `55119 / 53 / 88015 / 78145`. None may
  change after its event; 78,145 stays sealed until local promotion.

## Frozen treatment corpus (inherited)

`data/sft_axis160.jsonl`, byte-identical to the goal-gap axis experiment's
frozen corpus, sha256
`e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e` — 40 rows
each of `u_tracefix` / `u_explore` / `u_hygiene` / `u_protocol`, with the same
executable-truth, uniqueness, decoy, and banned-vocabulary audits already
adversarially verified (200/200 answers independently re-derived). Replay pool
`sft_blend.jsonl` `25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2`.

## Frozen exposure match

Same geometry as the predecessor: 1,280 shared replay rows (namespace seed
55,119) position-aligned across both arm files, plus one 240-row variable block
per arm (candidate = 160 treatment + 80 fillers; control = 240 replay), EXACT
(limit 0) on forward tokens, nonzero loss-bearing targets, and absolute loss
mass; MILP infeasibility is `STOP_EXPOSURE_MATCH_INFEASIBLE`; zero
encoder-skipped rows; trainer bytes bound to the receipt's encoder hash.

## Frozen training events

Control first, then candidate, each behind a clean pushed green checkpoint: one
epoch over 1,520 rows, 190 updates, LR 1e-5, rank 32 alpha 64, think/close
weights 0.2/0.2, training seed 53, warm start from the parent adapter continued
in place. Train loss is never capability evidence.

## Frozen local gate

One event at seed 88,015 over a frozen oracle-free 144-row input (40 axis
holdout at 10 per kind; 104 retention at 8 per original skill), three
composites (`replay_parent`, `replay_squared`, `axis_on_replay`), the pinned
engine geometry, and the predecessor's promotion logic verbatim with the new
labels: axis-holdout total strictly above BOTH controls; strict kind wins on at
least 3 of 4 (ties fail the kind; a 10/10 control is recorded as "not
detectable"); retention non-inferiority (correct ≥ each control − 5; caps ≤
each control + 3; parsed ≥ each control − 3); route abstentions ≤ 4. No
absolute per-kind floors. No promotion permanently seals seed 78,145.

## Frozen conditional pilot — MEDIUM tier

One aggregate-only event through the trusted gateway: tier `medium`, think
budget 1,024, sealed seed 78,145, canonical backend, four weight-recomputed
composites on the same seed: `base`
(`b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db`), the
parent, `replay_squared`, and the candidate; clean pushed `main` with the
promotion receipt committed; one-seed ledger; identical source inventory.

Pilot gates, all required: candidate aggregate strictly greater than base,
strictly greater than `replay_squared`, and strictly greater than the parent.

The goal gate — every public family strictly above base — is recorded and
reported from the same event. Medium-tier context: this gate has been passed 8
times in 92 historical medium events (versus once in 65 at quick), so a pass is
demanding but not foreordained to fail; a FAIL is still the majority outcome
under the hypothesis and must be recorded as "not confirmed at this event",
never as evidence against the mechanism. Secondary frozen readings: (a) the
replay-compounding measurement is `replay_squared` minus parent on aggregate,
reportable regardless of promotion; (b) the stack-interference reading compares
the candidate's axis-holdout per-kind counts against the predecessor's
(28/40 baseline); (c) family-union reading: whether the candidate holds BOTH
warren-type and rites-type flips in one model.

## Mandatory checkpoint order

1. Model-free construction + design review — committed, pushed, green.
2. `train-control`; 3. `train-candidate` (requires `PASS_CONTROL_TRAINING`);
4. `merge-arms` (requires `PASS_CONTROL_MERGE`); 5. `local`;
6. conditional `benchmark` only on promotion.

## Interpretation limits

Single-seed pilot; no claims minted. The package-level causal unit is the whole
arm. Benchmark firewall: contents never read; only gateway aggregates and
public per-family scores consumed.

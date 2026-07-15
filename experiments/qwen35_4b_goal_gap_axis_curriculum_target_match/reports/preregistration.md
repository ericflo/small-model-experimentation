# Preregistration: Goal-Gap Axis Curriculum

Frozen before any model event. A failed gate is a preserved result, never
permission to change this contract inside this experiment directory.

## Frozen identities

- Experiment: `qwen35_4b_goal_gap_axis_curriculum_target_match`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent (baseline and warm start): the authenticated `designed_fresh` arm of
  `qwen35_4b_universal_fresh_surface_budget_commit_target_match` — merged weights
  `0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979`, tree
  `93433aa2d5f3f0d6d4540126579c09feee1d8502df702c1563bae28eb7f60255`, warm-start
  adapter weights `36f41095c2d628e4706694e7d64d16aba815870a1d3660af0e24b14dc0e6b442`.
  Runtime LoRA is forbidden; every evaluated arm is an explicit merged composite.
- Arms: active control `replay_repeat`; candidate `axis_curriculum`; parent
  evaluation label `designed_fresh_parent`.
- Seeds: construction/slot-match/training/gate/aggregate =
  `77117 / 55118 / 52 / 88014 / 78144`. None may change after its event; 78,144
  stays sealed until local promotion.

## Frozen treatment corpus

`data/sft_axis160.jsonl`, sha256
`e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e`: 40 rows each
of four designed axis kinds built ONLY from public axis descriptions —

- `u_tracefix`: four invented executable formalisms (gauges, line, pile, chain);
  exactly one faulty instruction; the corrected instruction is provably the
  unique single-step repair (exhaustive enumeration over the instruction
  grammar); answer = `STEP <k>: <corrected instruction>`.
- `u_explore`: fully described one-way graph; the answer route is the unique
  shortest path and the stated budget equals its length; think targets use
  compact frontier notation.
- `u_hygiene`: records with embedded adversarial directives carrying a
  format-matched decoy value; the decoy never equals the true answer, so obeying
  any directive is parseable-but-wrong; 30 injected / 10 clean rows.
- `u_protocol`: documented two-flag-plus-tally procedures; closing-rule branches
  trained in balance (both / a-only / neither ≥ 2 rows each; observed 15/14/11).

Banned-vocabulary audit: zero occurrences of public benchmark family names, gym
family names and their flavor nouns (kiln, loom, burrow, chamber), every
predecessor surface pool (original six and fresh six), and every predecessor
attribute/capability set. Replay pool `sft_blend.jsonl`
`25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2`,
byte-identical to every predecessor's copy.

## Frozen exposure match

1,280 shared replay rows (family+kind stratified, deterministic sha256 rank,
namespace seed 55,118), position-aligned and byte-identical across both arm
files, plus one 240-row variable block per arm: control = 240 replay rows;
candidate = 160 treatment rows + 80 replay fillers. The two blocks must match
EXACTLY (limit 0) on forward tokens, nonzero loss-bearing targets, and absolute
loss mass in fifths, solved as one MILP; infeasibility is
`STOP_EXPOSURE_MATCH_INFEASIBLE`. Targets are never modified, duplicated, or
truncated; zero encoder-skipped rows required.

## Frozen training events

Control first, then candidate, each behind a clean pushed green checkpoint: one
epoch over 1,520 rows, batch 1, accumulation 8 (190 updates), LR 1e-5, cosine,
warmup 0.03, rank 32 alpha 64 dropout 0.05, think/close weights 0.2/0.2, answer
weight 1.0, max length 4,096, training seed 52, warm start from the parent
adapter continued in place, trainer bytes bound to the exposure receipt's
encoder hash. Train loss is never capability evidence.

## Frozen local gate

One evaluation event at seed 88,014: the parent composite and both newly merged
arms, sequential vLLM engine runs (greedy, natural thinking, 1,024-token cap,
max-model-len 4,096, gpu-mem 0.90, 16 sequences, 8,192 batched tokens, CUDA
graphs 1/2/4/8/16) over a frozen oracle-free 144-row input:

- Instrument A (installability): 40 axis-holdout tasks, 10 per axis kind,
  generated at the unseen gate seed.
- Instrument B (retention): 104 tasks, 8 per each of the 13 original skills.

Promotion — `axis_curriculum` promotes iff ALL hold:

1. Axis total: candidate correct on the 40 holdout rows STRICTLY greater than
   the parent's total AND the replay control's total.
2. Axis breadth: candidate strictly beats BOTH controls on at least 3 of the 4
   axis kinds (a kind counts only if candidate correct > max(parent, replay)
   for that kind; ties fail the kind).
3. Retention non-inferiority on the 104 retention rows: correct ≥ parent − 5
   and ≥ replay − 5; cap contacts ≤ parent + 3 and ≤ replay + 3; parsed ≥
   parent − 3 and ≥ replay − 3.
4. Feasible-route abstentions ≤ 4 of the 8 retention `u_route` rows.

There are NO absolute per-kind floors: the predecessor proved an induction-kind
floor is structurally unpassable for this lineage, and none of this gate's
conditions can be satisfied or failed by a wall the treatment does not touch.
If the candidate does not promote, aggregate seed 78,144 is permanently sealed
and the experiment closes as a local result.

## Frozen conditional aggregate pilot

One aggregate-only event through the trusted gateway: tier quick, think budget
1,024, sealed seed 78,144, canonical `qwen_vllm` backend, four explicit merged
models on the same seed: `base` (`b654e033...16db` weights), the parent, the
replay control, and the candidate — all weight-recomputed and bound before the
seed is consumed; clean pushed `main` with the promotion receipt committed at
HEAD; one-seed ledger; identical benchmark source inventory across arms.

Promotion gates, all required: candidate aggregate strictly greater than base,
strictly greater than the replay control, and strictly greater than the parent.

The goal gate — every public family strictly above base — is recorded and
reported from the same event. Frozen power statement: at quick-tier granularity
(~1/8-step family scores, base at exactly zero on several families) this gate
has failed for every historical arm including deltas far larger than this
trial's plausible effect; a PASS is extremely strong evidence, a FAIL is the
expected outcome even under the hypothesis and must be recorded as "not
confirmed at quick-tier granularity", never as evidence against the mechanism.
Family-level confirmation, if the aggregate wins hold, moves to the medium tier
in a result-separated successor together with independent seeds and a
same-backend matched-compute sample-more baseline.

Prospective reading of a non-promoted outcome: the axis-holdout per-kind counts
in the local receipt are the mechanism reading (which axes installed, which did
not), reportable as such with promotion bars unchanged. Frozen refinements to
that reading, none of which alter any bar:

- Drift versus content: the retention cap/parsed bands versus the PARENT are
  tighter than the empirically measured drift of pure replay continuation in
  the direct predecessor (5 caps / 4 parses of movement). If the candidate
  fails a retention band versus the parent while the replay control's own
  numbers drift comparably or further in the same direction, the failure is
  read as continuation drift, not content cost — the promotion still fails,
  but the record must say which.
- Control ceiling: a kind where either control scores 10/10 cannot register a
  strict win by construction; such kinds are recorded as "not detectable at
  this instrument", not as installation failures.
- Band edges: any retention comparison landing within ±5 of a band edge is
  reported together with the graded discordance count between the arms.
- Instrument scope: only `u_tracefix` varies formalism within its axis; the
  other three holdout kinds render the same surface as their training rows, so
  a per-kind win there evidences installation, not surface-generality. The
  tracefix uniqueness audit is bounded by the generator's visible instruction
  grammar (add amounts 1–3); two of forty training narrations overstate
  uniqueness under an unbounded amount reading, and the frozen holdout was
  verified unambiguous under amounts 1–9 plus novel items.

## Mandatory checkpoint order

1. Model-free construction (corpus, exposure match, gate design, design receipt,
   adversarial design review) — committed, pushed, green.
2. `train-control`; 3. `train-candidate` (requires `PASS_CONTROL_TRAINING` in
   `reports/compute_review.md` before stage 2); 4. `merge-arms` (requires
   `PASS_CONTROL_MERGE` in `reports/local_design_review.md`); 5. `local`;
   6. conditional `benchmark` only on promotion.

Every receipt binds runner inputs, artifact hashes, git state, and environment;
no stage may re-run its model event.

## Interpretation limits

The package-level causal unit is the whole arm; per-axis attribution inside the
candidate uses the preregistered per-kind holdout counts. The local suite is a
mechanism screen, not the claim. The absolute-loss-mass axis carries the
per-row-normalization caveat recorded by the predecessor. Benchmark firewall:
`benchmarks/` content is never read; only the gateway's aggregate and public
per-family scores are consumed.

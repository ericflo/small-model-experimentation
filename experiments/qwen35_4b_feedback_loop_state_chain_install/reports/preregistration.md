# Preregistration: Feedback-Loop + State-Chain Install

Frozen before any model event. A failed gate is a preserved result, never
permission to change this contract inside this experiment directory.

## Frozen identities

- Experiment: `qwen35_4b_feedback_loop_state_chain_install`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent (baseline and adapter base): the authenticated `hygiene_explore`
  composite — tree `9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971`,
  merge receipt committed in its home cell. FRESH rank-32/alpha-64 adapters
  trained from it via the trainer's `--model-path`; NO warm start. Runtime
  LoRA is forbidden; every evaluated arm is an explicit merged composite.
- Arms: active control `replay_ctl` (trained first); candidate
  `feedloop_state`; parent evaluation label `hygiene_explore_parent`.
- Seeds: construction/namespace/training/axis-gate/retention-screens/
  aggregate = `77130 / 55127 / 61 / 88026 / 88027+88028+88030 / 78151`.
  88029 collided with a predecessor's run_seed and was substituted by the
  preregistered next-free-integer rule; none may change after its event;
  78,151 stays sealed until local promotion.

## Frozen mechanism argument (why the trace-repair kill rule does not bind)

The closed tracefix axis taught single-turn repair CONTENT and died by kill
rule. This dose teaches the EPISODE PROTOCOL the line has never trained:
using rerun evidence across an act→observe→revise loop (`u_feedloop`) and
narrated hidden-state tracking across an executed procedure
(`u_statechain`) — the two skills the public suite metadata names for the
only two families (menders, rites) separating the parent from the recorded
all-families goal gate at medium. Precedents: episode-protocol installs
are the line's most reliable wins; C42 (step-resolved error localization);
C14 (state-chain SFT repairs simulation with transfer); C37 (NL chaining).

## Frozen treatment corpus

`data/sft_feedloop_state.jsonl`, sha256
`e6d45ed45632f7d6bea8e300f469a4d4c076eb4d8be420d677bd14c27471083b`, 160
rows at construction seed 77,130 (amended once pre-freeze after the
adversarial review — see the review note below):

- 80 `u_feedloop`: four invented executable formalisms (troughline,
  trinketcord, crankwheel, sigilslate; 20 rows each). Each row renders one
  bounded episode in one user message: machine spec (with every
  parameterized operation's legal values explicitly documented — "exactly
  three sizes … no other exists"), candidate instruction sequence, first
  failing evidence, a plausible earlier wrong fix, and fresh rerun
  evidence. Generator-verified structure: ≥2 legal fix candidates
  consistent with round-1 evidence (the wrong attempt among them), exactly
  1 consistent with rounds 1+2, and an extended-grammar audit (amounts to
  12, full item pools) proving every out-of-bound alternative is excluded
  only by the documented legality clause; think narrates what the rerun
  eliminates among LEGAL steps; answer is the provably unique legal fix.
  The repair is easy by construction — the lesson is the loop.

  Review note (pre-freeze amendment): the adversarial review's corpus lens
  hand-simulated the machines and found the original specs documented
  unbounded parameters while the uniqueness audit enumerated only the
  finite grammar — 13/80 training and 2/20 holdout rows admitted
  out-of-grammar valid fixes, a false-promotion bias on the strict kind
  gate and a false uniqueness claim in 13 think targets. Fixed by
  documenting the bounded legality in every spec (all four formalisms,
  including the previously undocumented trinket/sigil item pools),
  rewording think targets ("of the legal steps, only step k…"), and
  hard-failing generation unless extended-grammar survivors are excluded
  by the rendered clause. Same rng draws; the 15 rows are now genuinely
  unique; the counts are test-pinned.
- 80 `u_statechain`: two invented procedure machines (brewvat, courierloft;
  40 rows each) with hidden-but-documented counters; ≥3 hidden updates
  required, stateless and last-step-only distractor answers verified
  wrong; think narrates the state chain compactly.
- Banned-vocabulary audit extends the goal-gap list with all predecessor
  surface pools and the public description nouns (program, debug, bug,
  test, repair, rerun, protocol, state-machine, flag/flags); 56-token
  fresh-surface audit against 20 hash-pinned predecessor corpora/streams:
  zero word-boundary hits. Replay pool `sft_blend.jsonl`
  `25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2`,
  byte-identical to every predecessor's copy.

## Frozen exposure match

1,280 shared replay rows (family+kind stratified, deterministic sha256
rank, namespace seed 55,127), position-aligned and byte-identical across
both arm files, plus one 240-row variable block per arm (control: 240
replay; candidate: 160 treatment + 80 replay fillers), matched EXACTLY
(limit 0) on forward tokens, nonzero loss-bearing targets, and absolute
loss mass in fifths as one MILP. Receipt: per-arm forward 1,393,242,
nonzero targets 584,414, loss-mass×5 640,286; zero encoder-skipped rows;
trainer bytes bound (`train_think.py` e0eca2a2…, encode_row byte-identical
to the reviewed reference).

## Frozen training events

Control first, then candidate, each behind a clean pushed green
checkpoint: one epoch over 1,520 rows, batch 1, accumulation 8 (190
updates), LR 1e-5, cosine, warmup 0.03, rank 32 alpha 64 dropout 0.05,
think/close 0.2/0.2, answer 1.0, max length 4,096, training seed 61, fresh
adapter from the hygiene_explore composite. Train loss is never capability
evidence.

## Frozen local gate

One evaluation event: the parent composite and both newly merged arms,
sequential authenticated vLLM engine runs over four frozen oracle-free
inputs per arm — the 40-row axis holdout at seed 88,026 (20 per episode
kind, fresh instances) and three 104-row retention screens at seeds
88,027 / 88,028 / 88,030 (8 per original skill; overlap receipts vs all
thirteen predecessor gates, 21 pinned corpora/streams, both training
streams, and each other).

Promotion — `feedloop_state` promotes iff ALL hold:

1. Axis total: candidate correct on the 40 holdout rows STRICTLY greater
   than the parent's total AND the replay control's total.
2. Both kinds individually: candidate strictly beats BOTH controls on
   `u_feedloop` AND on `u_statechain` (ties fail the kind).
3. Retention under the calibrated pooled_k3 protocol (first use): on the
   POOLED MEAN across the three screens — candidate correct ≥ parent − 5
   and ≥ replay − 5; caps ≤ parent + 3 and ≤ replay + 3; parsed ≥
   parent − 3 and ≥ replay − 3 (implemented as exact integer sums with
   3× bands; mathematically identical).

No absolute per-kind floors. If the candidate does not promote, seed
78,151 is permanently sealed and the experiment closes as a local result
with the per-kind mechanism reading.

## Frozen conditional benchmark event

Only on local promotion: one medium-tier aggregate event through the
trusted gateway at think budget 1,024, sealed seed 78,151, four explicit
merged models on the same seed — base (weights b654e033…), the
hygiene_explore parent, replay_ctl, and feedloop_state — all
tree-recomputed and bound pre-event, hardened runner (review-verdict and
code-pin checks at the seed boundary, write-ahead one-seed ledger,
finiteness guards), clean pushed green main.

Pilot gates, all required: candidate aggregate strictly greater than base,
strictly greater than replay_ctl, strictly greater than the parent.

The goal gate — every public family strictly above base — is recorded from
the same event either way. Frozen power statement: the parent sits at 8/10
with ties at menders and rites (both 0.0 at seed 78,150); the candidate
needs exactly those two families > 0 while keeping the other eight strict
wins. The tier forensics' historical pass rate is 9/94 for gym-trained
arms; this line is contamination-free, so a PASS is the program milestone
and MUST then survive the confirmation law (independent seeds +
same-backend matched-compute sample-more) in a result-separated successor
before any claim; a FAIL is the expected outcome (prior ~15–30%) and is
read via the per-kind axis counts and per-family deltas.

## Mandatory checkpoint order

1. Model-free construction — committed, pushed, green.
2. `train-control`; 3. `train-candidate` (requires `PASS_CONTROL_TRAINING`
   in `reports/compute_review.md` before stage 3); 4. `merge-arms`
   (requires `PASS_CONTROL_MERGE` in `reports/local_design_review.md`);
   5. `local` (requires `PASS_LOCAL_EVENT` in the same file);
   6. conditional `benchmark` (requires `PASS_BENCHMARK_EVENT` in
   `reports/benchmark_design_review.md` + the committed promotion receipt).

Every receipt binds runner inputs, artifact hashes, git state, and
environment; no stage may re-run its model event.

## Interpretation limits

The package-level causal unit is the whole arm; per-kind attribution uses
the preregistered holdout counts. One aggregate seed, one tier: a pilot
win is a map reading until confirmed. The absolute-loss-mass axis carries
the per-row-normalization caveat recorded by predecessors. Benchmark
firewall: `benchmarks/` content is never read; only the gateway's
aggregate and public per-family scores (and the two suites' top-level
README/CONTRACT metadata) are consumed.

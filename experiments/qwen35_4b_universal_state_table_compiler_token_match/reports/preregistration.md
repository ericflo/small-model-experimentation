# Preregistration

## Frozen identities

- Experiment: `qwen35_4b_universal_state_table_compiler_token_match`.
- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: `close_xi`, weights `16e9dc75...c179`, config `de953bd5...7ff`.
- Construction/training/local/conditional aggregate seeds:
  `77112/46/88008/78138`.
- Candidate: `state_table_after_close`.
- Active control: `replay_after_close`.
- Parent evaluation label: `close_xi_parent`.

## Frozen data and compute

- Curriculum source: 80 rows, SHA-256 `a7b453af...e88bb`; 20 each execute,
  independent hypothesis score, repair, and commit.
- Replay source: 2,240 rows, SHA-256 `25a9595f...f0c2`.
- Inherited partition manifest: SHA-256 `abf8b505...966f`.
- Replay stream: SHA-256 `2727e29a...a2b5`.
- Candidate stream: SHA-256 `8e1b8fdc...1355`.
- Token receipt: SHA-256 `163e40a6...f0b8`.
- Each arm: 320 rows, 286,814 forward tokens, zero skips, 40 optimizer
  steps; 200 replay positions are byte-identical across arms.
- Training: one epoch, learning rate `1e-5`, LoRA rank/alpha `32/64`, batch
  1, gradient accumulation 8, maximum length 4,096, ordinary thought/close
  weights `0.2/0.2`, seed 46.

## Mandatory checkpoint order

No multi-stage command is authorized.

1. From the pushed, CI-green design commit, run `--stage train-control`.
2. Preserve the control adapter hashes, log, and receipt; run `make check`, commit,
   fetch/rebase, push `main`, and verify Validate Repository plus Publish Research
   Site.
3. From that clean checkpoint, run `--stage train-candidate`.
4. Preserve candidate adapter hashes, log, and receipt; repeat the full publish gate.
5. From that clean checkpoint, consume the single fresh local event with
   `--stage local`.
6. Preserve either the empty or eligible promotion receipt and publish it before any
   merge. Failure ends this experiment with aggregate seed 78,138 sealed.
7. Only an authenticated eligible receipt permits `--stage merge`. Publish all merge
   receipts before broad evaluation.
8. Only then may `--stage benchmark` consume the one conditional aggregate event.

The harness enforces a clean worktree and committed predecessor receipts at every
transition. Rebase conflicts must be resolved and all checks rerun before push.

## Fresh local admission

Run parent, replay, and candidate together in one greedy Transformers process at seed
88,008, batch size 4, and a 1,024-token cap. Candidate must satisfy all:

- accuracy ≥0.65;
- parse rate ≥0.90;
- cap contacts ≤2;
- fewer than two feasible-route abstentions;
- `u_execute`, `u_induct`, and `u_probe` accuracy each ≥0.50;
- total correct strictly greater than both parent and replay;
- correct over `u_execute` + `u_induct` + `u_probe` strictly greater than both parent
  and replay.

No tie promotes. No threshold or target subset may change after seeing output.

## Conditional aggregate pilot

Run exactly six merged models—base, blend, replay refresh, immediate parent, active
replay, and candidate—in one aggregate-only quick event on `qwen_vllm`, seed 78,138,
think budget 1,024. Candidate promotion requires:

- aggregate strictly above base;
- strict positive delta versus base on all ten public families;
- aggregate at least blend;
- aggregate strictly above replay refresh, immediate parent, and active replay.

Failure is preserved and ends the experiment. A pass is a pilot, not the goal: open a
fresh result-separated higher-tier confirmation with independent seed(s) and a
matched-compute sample-more baseline before claiming universal installation.

## Interpretation limits

- The causal unit is the four-stage package, not any one stage.
- Exact forward compute does not equalize prompt/thought/answer composition.
- Train loss is never compared as capability evidence.
- The local suite is a mechanism screen, not a broad benchmark.
- No benchmark item, transcript, source, or private result may enter training or agent
  context.

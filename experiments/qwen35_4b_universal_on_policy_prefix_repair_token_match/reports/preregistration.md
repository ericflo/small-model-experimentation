# Preregistration

## Frozen identities

- Experiment: `qwen35_4b_universal_on_policy_prefix_repair_token_match`.
- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: authenticated `close_xi`, weights `16e9dc75...c179`, config
  `de953bd5...7ff`; failed scaffold and state-table adapters are not inherited.
- Construction/rollout/training/local/conditional-aggregate seeds:
  `77113/66113/47/88009/78139`.
- Planned candidate/control labels: `prefix_repair_after_close` and
  `replay_after_close`; neither training arm exists or is authorized yet.

## Frozen collection substrate

- Source: 288 fresh truth-audited procedural tasks, SHA-256
  `32589348...1172`.
- Model-facing input: SHA-256 `7a643e96...a5485c`; hidden oracle and answer fields
  are absent.
- Six classes, 48 tasks each: declaration-versus-operation, state transition,
  bounded induction, probe scoring, repair propagation, commit serialization.
- Inherited clean replay source: 2,240 rows, SHA-256 `25a9595f...f0c2`.
- Inherited replay partition manifest: SHA-256 `abf8b505...966f`.
- Fresh local seed 88,009 is not materialized during collection or selection.

## Mandatory checkpoint order

No multi-stage invocation is authorized.

1. From this pushed, CI-green design checkpoint, run `--stage merge-parent`.
2. Preserve merge hashes/log/receipt; run the smoke and `make check`, commit, fetch,
   rebase, rerun checks, push `main`, and verify both required workflows.
3. From that clean checkpoint, run `--stage collect-parent` once: all 288 tasks in one
   vLLM event, merged composite, natural thinking, greedy, one sample, seed 66,113,
   1,024-token cap.
4. Preserve rollout JSONL, metadata, log, and receipt; repeat the full publish gate.
5. From that clean checkpoint, run `--stage mine-prefixes`. Selection is model-free
   and fixed; preserve either 60 repairs or the insufficient-quota negative.
6. If and only if all six quotas pass, materialize exact-token streams and perform a
   second adversarial compute review in a new committed checkpoint.
7. Only that second review may expose control training, candidate training, and the
   fresh local event, each with its own publish gate.

Rebase conflicts must be regenerated or resolved from the combined source tree, and
all gates must be rerun before every push to `main`.

## Frozen failure selection

- A row is eligible only for cap contact, missing answer, wrong answer,
  noncanonical serialization, declaration-as-operation language, or commit thinking
  longer than 32 tokens.
- Exactly ten reachable failures per class are selected by frozen severity and
  deterministic seed. No class borrowing, replacement, or threshold tuning.
- Delayed commit cuts at token 33. Other free-form failures cut at their first
  machine-observable close/answer or cap boundary; no latent-first-error claim.
- Parent prefix token ids are stored exactly and masked fully from loss. Empty,
  close-bearing, EOS-bearing, decode-mismatched, or overlength prefixes are rejected.
- An undersupplied class writes an inventory, yields no trainable source, and stops
  the experiment before training.

## Prospective compute and gates

The intended future arms have 320 rows, 200 position-aligned shared replay rows,
40 effective-batch-eight optimizer steps, one epoch, and training seed 47. Candidate
variable content is 60 prefix repairs plus 60 replay fillers; replay control variable
content is 120 independent replay rows. Exact forward-token equality and zero skips
remain requirements, not facts, until actual prefixes are tokenized.

Local admission will require the sole candidate to satisfy every absolute gate and
strictly beat the unchanged parent and replay continuation both overall and over
execute/induct/probe. No tie promotes. Failure writes an empty promotion receipt and
keeps aggregate seed 78,139 sealed.

A local pass still requires strict positive lift on every public benchmark family in
one same-backend aggregate event, then a result-separated higher-tier confirmation
and matched-compute sample-more comparison. No broad event can provide training data.

## Interpretation limits

- The causal unit is the six-class masked prefix-repair package.
- Selection is failure-conditioned; it is not an estimate of natural task frequency.
- Correct-but-delayed commit is a deployed-policy failure, not an accuracy failure.
- Exact forward compute will not imply equal target-token composition.
- Train loss is never capability evidence.
- No benchmark item, transcript, source, or private result may enter training or
  agent context.

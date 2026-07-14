# Adversarial Review: Explicit Composites and Fresh Local Gate

**Date:** 2026-07-14
**Scope:** trained-arm deployment, fresh procedural evaluation, promotion rules, and
checkpoint order after paired training
**Verdict:** `PASS_CONTROL_MERGE`.

This review was completed before either current adapter was merged and before any
seed-88,010 model call or capability result existed. It authorizes only the replay
control merge after this design is committed to a clean, pushed, green `main`.
Candidate merge and local evaluation remain separately checkpointed stages.

## Observation and contamination boundary

- The only model is `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- The review used experiment-owned procedural code, published training and parent
  receipts, and external artifact identities. It inspected no held-out suite item,
  transcript, family implementation, generated item, or detailed result.
- Fresh local seed 88,010 deterministically creates 26 executable-truth tasks: two
  for each of the 13 universal skills. Model-facing rows contain only `id`,
  `messages`, and public `meta`; answers and executable audits stay outside the
  runner input.
- Source, runner-input, and design-receipt SHA-256 values are
  `7b69473bb9b15b4b75b63587825afa0e9a8e5a9220ebdd98b66f3f56c34d975f`,
  `6efefc92bf47b7d44cc9587004dbd9b2238d78beed82dfffdc184e9aa65d15e2`,
  and `124bbf99607105c22613e80a1965cf528a04e6211573b9c823ab1fc6e3e82db5`.
- Canonical message bytes are checked against both final training streams, the
  parent collection source, and prior local seeds 88,000 through 88,009. Any overlap
  stops materialization.
- Aggregate seed 78,140 and every held-out result remain sealed. A local pass only
  licenses a new, separately frozen aggregate-gateway stage.

## Deployment audit

Runtime LoRA is forbidden because it is a verified silent no-op for this Qwen3.5
adapter/module layout in the pinned vLLM stack. The unchanged published parent and
both trained arms therefore use explicit full composite checkpoints. Every arm is
served by the same experiment-local vLLM runner with natural thinking, greedy
decoding, one sample, seed 88,010, a 1,024-token cap, 4,096-token context, 16
sequences, 8,192 batched tokens, and CUDA-graph sizes 1/2/4/8/16.

The pinned merger loads the official composite in bfloat16 and computes each LoRA
product in float32 before casting the updated weight back to bfloat16. That cast can
erase sufficiently small deltas; the result therefore concerns the deployed
composite produced by this method, not an abstract infinite-precision adapter. The
method is symmetric across arms, applies exactly 128 nonzero modules at scale 2,
disables CUDA TF32, and records finite positive delta norms.

The wrapper closes a prior artifact-integrity gap. It now rejects symlinks, nested
entries, or any file set other than the seven expected composite files; verifies the
Qwen3.5-4B config/tokenizer fingerprint; and records name, byte size, and SHA-256 for
the complete weight/config/tokenizer tree. The evaluator rehashes that full tree.
The older replay parent predates this receipt schema, so its exact seven-file
manifest and canonical tree hash are frozen directly in this design.

## Evaluation transaction audit

The only authorized order is:

1. publish this design and obtain both green repository workflows;
2. merge replay control, preserve its log and receipt, publish, and obtain both
   green workflows;
3. merge the counterfactual-restart candidate from its independently trained
   adapter, then preserve, publish, and obtain both green workflows;
4. run the one logical local event across parent, replay control, and candidate;
5. preserve all raw outputs, metadata, logs, grading, and either a promotion or an
   empty-promotion receipt.

Every stage refuses overwrite and requires clean pushed `main`. The local wrapper
reauthenticates committed design bytes, the current full model tree, branch, `HEAD`,
and `origin/main` immediately before and after every arm process. A process or
authentication failure receives a durable failure receipt; partial artifacts are
not deleted or silently resampled. Runner metadata must bind the model override,
runner/input hashes, exact engine and sampling geometry, resolved greedy sampling,
resolved CUDA graphs, request/completion counts, and expected dirty status caused by
opening the durable wrapper artifacts.

## Promotion-rule audit

The candidate must parse at least 24 of 26 answers, solve at least 17, contact the
token cap at most twice, abstain on at most one of the two route items, and solve at
least one item separately for execute, induct, and probe. It must also strictly beat
both the unchanged parent and matched-exposure replay control on total correct and
on the six execute/induct/probe items. A tie fails.

Before scoring, the checker requires exactly 78 graded rows, exactly the three
frozen labels, the same 26 unique task ids and task-to-kind mapping for every arm,
and exactly two rows for every registered kind. Correctness and cap fields must be
JSON booleans. The parser uses only the last exact `ANSWER:` line, and exact
procedural truth determines correctness. Training loss, merge norms, and control
weakness cannot independently promote the candidate.

## Remaining risks

1. Two items per skill make the local gate noisy. Strict wins against two controls
   and a fixed absolute floor reduce false promotion but may reject a useful model.
2. All local items share the experiment's procedural interface. Freshness prevents
   item reuse, but a pass is still local transfer evidence, not held-out benchmark
   evidence.
3. The three arms load sequentially rather than in one engine process. The wrapper
   fixes identical bytes and geometry and authenticates every boundary, but makes no
   common-random-number claim.
4. Exact forward tokens, target tokens, loss mass, optimizer steps, and shared-row
   alignment isolate exposure better than the predecessor, but the candidate and
   replay streams retain a documented zero-weight span-composition difference.

No remaining issue permits outcome-aware adjustment. After this checkpoint is
published and both workflows are green, exactly `--stage merge-control` is
authorized.

## Post-control receipt

The authorized replay-control merge later completed from pushed-green commit
`3b8b46aa`: 128/128 nonzero modules at scale 2, full weight SHA-256
`e48ed4a03ba8d040c4007f115af5346df51b886566a5798250586a10e989ae17`, and
complete-tree SHA-256
`d1a8336d1648190cd2143fbf8e3bf9031b6572f611ebc51e6649beef6d456027`.
Run/log/external-receipt hashes are `751a0152...f72f`, `8a438197...281b`, and
`bcb0060e...53e2`. This records the frozen deployment artifact and does not expand
the original verdict. Candidate merge becomes authorized only after this receipt
checkpoint is itself published and green in both workflows.

## Post-candidate receipt

The authorized candidate merge later completed from pushed-green commit `6c551000`:
128/128 nonzero modules at scale 2, full weight SHA-256
`d704af190171c133b77ce0bfc96e92096be3828e51124020200514859fd049a9`, and
complete-tree SHA-256
`9f64dc55f290de8f0b70bd51259f40cfe2a1c5f823614d0e7547e94201a14a1b`.
Run/log/external-receipt hashes are `2956fa41...8ea7`, `e138a06c...b483`, and
`97edeb08...6df6`. This records the second frozen deployment artifact. Local
evaluation becomes authorized only after this checkpoint is itself published and
green in both workflows; the aggregate remains sealed.

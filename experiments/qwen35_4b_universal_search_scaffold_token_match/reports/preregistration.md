# Preregistration

## Question and hypothesis

Can independently supervised, executable substates of two-operation rule search
install the missing execute/induct behavior from the `close_xi` near-miss better than
an exact-token replay continuation? The registered expectation is that training
apply-first, fit-second, reject-first, execute-pair, and bounded-search interfaces as
separately scored tasks makes those transitions addressable enough to cross the
unchanged local gate.

## Frozen model, parent, sources, and arms

- Only model: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: published `close_xi` adapter from
  `qwen35_4b_universal_close_weight_token_match`, authenticated by weights SHA-256
  `16e9dc75a0e33e182e916600ff6e1d75fc46dfa45e870216e2c149a41253c179`
  and config SHA-256
  `de953bd57502ff728a12d1627d5aacab6284b045428ec7b83026388afd8c47ff`.
- Scaffold source: 80 rows at construction seed 77,111, SHA-256
  `5854c218479a500f969bf2dbcfdbc30cd8a6095fa38aeaa652a220219b50a093`.
- Replay source: 2,240 rows, SHA-256
  `25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2`.
- The predecessor partition is authenticated by SHA-256
  `abf8b5055e68c0fb2bb6e32a29f7be3b3677a0dd179e77397647777a2aa0966f`.

The two arms are:

- `replay_after_close`: the inherited 200-row replay core plus its inherited
  120-row replay-control block;
- `scaffold_after_close`: the same 200-row core plus 80 scaffold rows (16 per
  stage) and a disjoint 40-row exact-token replay filler.

Both streams contain 320 rows, exactly 286,814 forward tokens, zero max-length
skips, and exactly 200 byte-identical shuffled positions. Their SHA-256 identities
are `c157fb135f0934375de3c36d3258b4d2621a09f9831f4eb9f1a8f5bb959c355d`
and `79a8d7c933a220b809447f144f07c2352f89f462198b07b64b30275cf8790b90`.
The token receipt SHA-256 is
`eeb12b95c915e9a32755e73db94b5eb69a5aec53788461e1a50aa9b72f1e4a0f`.

Every scaffold answer is generated from executable state. Tests independently
recompute the operation semantics, unique fitting pair, dead first operation,
intermediate states, and answer. The 16 reject rows are balanced 8 `FIT` and 8
`NO_FIT`. `_audit` construction state is stored but never rendered into a model
prompt or target.

## Frozen training

Each arm starts independently from the authenticated parent and runs one epoch,
learning rate `1e-5`, rank 32, alpha 64, batch size 1, gradient accumulation 8,
max length 4,096, thought weight 0.2, ordinary autonomous-close weight 0.2, and
seed 45. Each run has 40 optimizer steps. Target-specific close weighting is absent
from both the trainer and wrapper.

The wrapper refuses overwrites, authenticates the parent, stream, token receipt,
hyperparameters, and external output path, then preserves the command, log, package
versions, git state, loss, wall time, and adapter hashes. The replay control is
trained, committed, pushed, and CI-verified before candidate training begins.

Exact forward-token equality does not imply identical supervised-span composition.
Replay has 116,036 prompt, 167,411 thought, 640 close, and 2,727 answer tokens;
candidate has 124,245 prompt, 158,311 thought, 640 close, and 3,618 answer tokens.
That difference is part of the registered curriculum intervention.

## Fresh local promotion event

After both arms are published, evaluate the immediate parent, replay continuation,
and scaffold continuation together on the same 26 experiment-owned procedural cases
at seed 88,007. Use greedy Transformers inference, 1,024 maximum new tokens, and
batch size 4. Preserve every completion and summary.

The absolute checks are:

- accuracy at least 0.65;
- parse rate at least 0.90;
- at most two generation-cap contacts;
- fewer than two feasible-route abstentions;
- `u_execute` accuracy at least 0.50;
- `u_induct` accuracy at least 0.50.

Only `scaffold_after_close` may promote. Parent and replay gates are reported controls,
never alternate candidates. If the candidate fails any check, write the negative
promotion receipt, stop nonzero, and leave aggregate seed 78,137 sealed.

## Conditional paired aggregate pilot

If and only if the scaffold candidate passes locally, explicitly merge the parent,
replay control, and candidate. Run one aggregate-only quick@1,024 event at seed
78,137 through the trusted gateway with exactly six models: base, `blend`, inherited
`replay_refresh`, immediate `close_xi_parent`, active `replay_after_close`, and
`scaffold_after_close`. Every arm uses the `qwen_vllm` backend and the same benchmark
implementation receipt.

The candidate passes the pilot only if it has a strictly positive delta versus base
on all ten reported families, aggregate at least `blend`, and aggregate strictly
above inherited replay refresh, its immediate parent, and the active replay control.

## Claim and continuation boundary

The five-stage package is the causal treatment; the final search target demonstrates
one rejected and one successful branch but does not exhaustively enumerate the full
operation universe. A local pass is mechanism qualification only. A pilot pass is
exploratory evidence, not generalized installation. The universal claim requires a
new result-separated confirmation experiment with fresh quick seeds, medium@2,048,
paired uncertainty, and a matched-compute sample-more baseline. No benchmark item,
source, transcript, or private result detail may be read.

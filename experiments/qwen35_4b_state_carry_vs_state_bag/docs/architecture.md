# Architecture and Implementation Contract

## The Counterfactual

Let `x` be the fixed token sequence, including eight state slots before the query. Let `P`, `R`, and `C` be Qwen text layers `[0,12)`, `[12,20)`, and `[20,32)`.

The untouched first pass computes:

```text
h_P = P(embed(x))
h_1 = R(h_P)                 # recurrence LoRA disabled
m   = reset_memory(h_1)      # fixed non-state positions
z_1 = gather_state(h_1)
```

For `K>1`, a trainable low-rank initializer maps `z_1` into the recurrent coordinate system. This initializer is bypassed entirely for K=1.

For each extra call `t`:

```text
Carry source_t = z_(t-1)
Bag   source_t = z_1

u_t       = source_t + projected_sinusoid(t)
c_t       = gather_state(R(scatter(m, u_t)))
z_t       = source_t + sigmoid(damping) * (c_t - source_t)
z_t       = semantic_echo(z_t)  # identity in the primary continuous config
```

The same aggregator is used in both arms:

```text
z_out = g * z_K + (1-g) * mean(z_1 ... z_K)
answer_logits = LM_head(C(scatter(m, z_out))[answer_position])
```

At K=1, `mean(z_1)=z_1`, so aggregation is identity for every value of `g`.

## Why State-Bag Is Strong

Bag is not a no-compute or frozen control. Every branch receives:

- the complete first-pass prompt memory;
- the initialized state;
- a different sinusoidal step identity;
- a full application of the same eight Qwen layers with the same LoRA;
- the same damping and optional semantic channel; and
- the same final aggregator and coda.

It can learn a direct shallow map from `(world, initial state, requested depth, step identity)` to the state at step `t`. What it cannot do is consume another branch's discovered state. If Bag matches Carry, the evidence favors shallow direct representations or ensembling rather than serial representational depth.

## Causal Query Placement

The rendered order is:

```text
world + rule + initial state + requested depth
<state slot> × 8
query kind + choices
assistant answer position
```

Qwen's Gated DeltaNet and full-attention mixers are causal. State positions cannot see later query tokens. The auxiliary node/phase/checksum decoders also receive no query embedding. A useful state must therefore support multiple potential future questions.

The node target is its visible row position in the rendered world table, not the generator's hidden random node number. This gives the shared decoder a legitimate task-relative identity while surface names and table order continue to change across examples.

## Qwen-Specific Forward

The wrapper follows Transformers 5.13.0 `Qwen3_5TextModel.forward`:

1. Embed the complete unpadded batch-one sequence.
2. Construct 4-axis position IDs; use axis 0 for text masks and axes 1–3 for text rotary embeddings, matching upstream.
3. Build `create_causal_mask` for full-attention layers and `create_recurrent_attention_mask` for GDN layers.
4. Call decoder layers with `past_key_values=None` and `use_cache=False`.
5. Reuse position embeddings and masks for every repeated R call.
6. Apply the original final RMSNorm and frozen LM head.

The experiment does not use generation cache inside recurrence. Repeating a block with an old hybrid cache would represent a different and poorly defined computation.

## Adapter Locality

The loader enumerates actual `torch.nn.Linear` modules whose fully qualified names lie under text layers 12–19. PEFT LoRA is attached only to those exact modules. After attachment, every trainable LoRA parameter name is parsed back to a layer number and must remain inside `[12,20)`.

LoRA is disabled during:

- prelude;
- the first R application in Carry/Bag;
- coda; and
- direct K=1 parity evaluation.

It is enabled during every extra R application and nowhere on the ordinary K=1 path.

Rank 32 constrains each learned weight update, not the dimensionality of the recurrent activation.
Every update is added to a full pretrained projection; eight nonlinear Qwen layers, residual paths,
and repeated applications can therefore move the full-width hidden state through a high-dimensional
trajectory. The genuine risk is narrower: the task-specific change available at each projection may
be too low-rank to install the required transition operator. Dense joint-state supervision and the
full-width eight-slot workspace make LoRA plausible, while preregistration section 10's full-rank
extra-call successor resolves that remaining capacity counterfactual if state formation fails.

## Stabilization

- Extra-step projection begins at zero.
- State initializer's output projection begins at zero.
- Damping begins at 0.125.
- Last-state aggregation weight begins at 0.90.
- State after task completion receives a fixed-point penalty.
- Step identity is sinusoidal rather than a learned lookup, so K=5–12 has a defined input.

These choices make the first extra computation a small refinement while leaving room for training to increase its magnitude.

## Interface Scope

This experiment carries the continuous Qwen state unchanged. An earlier conditional semantic
decode/re-embed branch was removed in the final pre-run review: it changes the architecture,
lacked its promised shuffled/wrong-task controls, and would turn a confirmatory experiment into
outcome-dependent design search. If the continuous run produces a readable-but-unused signature,
that interface intervention belongs in a separately preregistered successor experiment with its own
Carry/Bag pair and controls.

## Compute Accounting

For sequence length `L`, K recurrent applications, 32 total layers, and 8 loop layers:

```text
layer-token applications = L * (32 + 8 * (K - 1))
```

Prelude and coda run once; the first R is already part of the 32-layer base pass. Carry and Bag have identical receipts. Explicit-CoT sampling may spend no more than this many decoder-layer token applications, computed from its own prompt and allocated generation lengths. Its sampling parameters and depth-aware minimum allowance are frozen before confirmation; cap contact or parse failure beyond the registered limits invalidates a deployment comparison rather than manufacturing a recurrence win.

This unit is transparent and architecture-local, not a claim of exact wall-clock FLOP equivalence. GPU seconds and peak VRAM are retained as diagnostics.

## Checkpoint Contract

Every checkpoint contains:

- PEFT adapter directory;
- `loop_state.pt` for initializer, step projection, sufficiency heads, and scalars;
- `checkpoint.json` with model/revision/backend/config hash, arm, seed, phase, fixed-final step, parameter receipt, ordered-row digest, critical-source digest, training-lock digest, package/device inventory, canonical checkpoint identity, and adapter hashes.

Loading rejects stale config, source, lock, phase, step, seed, or tensor identity. Checkpoints are external under `large_artifacts/` and must never be committed. Interrupted training is deliberately non-resumable: preserve the partial attempt and restart step zero in a fresh attempt directory; never evaluate an intermediate checkpoint.

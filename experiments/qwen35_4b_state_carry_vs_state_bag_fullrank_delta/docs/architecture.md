# Architecture Contract

The pinned Qwen text stack is partitioned `P=0..11`, `R=12..19`, `C=20..31`.
R contains two complete `[GDN,GDN,GDN,attention]` motifs. Eight causal
`<|fim_pad|>` state slots precede the later question.

The first `P→R` pass runs under `torch.no_grad()` with direct deltas disabled.
For K>1, each extra R call receives frozen first-pass non-state memory plus either
the previous state (Carry) or first state (Bag), the same sinusoidal step signal,
and the same damping. Only state slots cross calls. A last-plus-mean state is
scattered into frozen memory and passed through an unadapted C and LM head.

Each discovered `nn.Linear` in R gets a zero-initialized FP32 delta of identical
shape. A forward hook adds `2 * DeltaW(dropout(x,0.05))` only while the explicit
extra-call context is open. The 62-target manifest and 892,272,640 count are
verified live. The base remains frozen and is never serialized into checkpoints.

K=1 bypasses the state initializer and aggregator and uses raw first-R state.
This makes its path algebraically identical to the base model over the identical
token sequence; the live logit-parity gate remains authoritative.


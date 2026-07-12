# Preregistration: Commit-Slot Semantic Power Replication

Frozen before any model call. CPU task generation, exact enumeration,
fingerprint comparison, lens hashing, unit tests, gate arithmetic, and a power
calculation from the already-terminal parent may precede the immutable design
commit. No new model outcome informed these rules.

## 1. Registered question

Does the parent's cap-1,024 ordered-thought advantage replicate as a broad,
task-level semantic effect under the identical fixed answer slot, rather than a
pooled gain concentrated in five mixed tasks and a few aliases?

This experiment tests only the answer seam. A positive result licenses separate
implementation/audit of J value; it is not itself a J result. A negative closes
this fixed-1,024 interface branch before any new cap or decoder is considered.

## 2. Parent boundary

`qwen35_4b_commit_slot_jacobian_value_transport` is terminal
`COMMIT_SLOT_SEAM_FAIL`. At cap 1,024 it produced:

- real ordered thought 15/48 (0.3125);
- no-thought 4/16 (0.25; equivalent to 12/48);
- exact-token-multiset shuffle 11/48 (0.22917);
- five mixed tasks versus six required;
- task real-minus-shuffled mean 0.08333, SD 0.35486, two-sided bootstrap
  interval crossing zero; and
- unmasked alias top 41/48 with mean total alias mass 0.68474.

No cap was selected, confirmation stayed unopened, and no J feature was fit.
The current experiment is a new powered qualification/confirmation study, not a
late confirmation or reinterpretation of the failed parent.

## 3. Fixed model, lens, and backend

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Exact lens SHA-256
  `e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`;
  frozen layers 4--8 and rank 24 each. No J feature is read in seam stages.
- Transformers bf16 SDPA, unpadded batch one for every arm. Native and
  close-only generation use KV cache; all slot arms use the same exact
  cache-free full prefill. No vLLM or backend mixing.
- Temperature 0.6, top-p 0.95, top-k 20 for sampled text. Slot decoding is
  deterministic constrained argmax.
- Fixed cap 1,024. Caps 256/512, cap 2,048, scale sweeps, and decoder calibration
  are absent.

Outcome-blind model smoke may record architecture, revision, packages/GPU,
token IDs, lens ranks, cache lengths, fixed-slot tokenization, and finite logits.
It may not record correctness, trace text, chosen alias, or comparative metrics.

## 4. Fresh procedural substrate

Generate 322 new tasks: 113 seam qualification, 113 untouched seam
confirmation, 48 value fit, and 48 causal confirmation. Each has eight visible
and eight hidden examples from the self-contained depth-two list DSL.

For every row, exhaustive CPU checks require a unique visible first-operation
type across all consistent concrete depth-two pipelines and no concrete
depth-one fit. `negate` remains second/distractor only. First-operation counts
differ by at most one per split. All fingerprints must be unique and have zero
overlap with five direct J/seam parents. Only visible examples enter prompts.
No file under `benchmarks/` is read or imported.

## 5. Fixed commit-slot policy

For each task, sample three native paths to 1,024 generated thought tokens. If a
natural close occurs first, retain only thought before that close. Otherwise use
the exact 1,024-token prefix. EOS/short malformed termination remains an
incorrect nonfinite row in every denominator and is never replayed.

Append exactly `</think>\n\nFirst:` and read the next logits. The registered
choice is argmax over the 12 public one-token aliases. Preserve full-vocabulary
logits from the same forward and report top-is-alias, total alias mass, and the
gold alias's unmasked probability. The controller supplies syntax/vocabulary,
never identity.

## 6. Matched controls

For every real prefix:

1. deterministically permute positions while preserving the exact thought-token
   multiset and length, then run the identical slot;
2. run close-only free-form generation from the real prefix for at most 16
   tokens with a disjoint stable seed; and
3. once per task, run the immediate no-thought close plus identical slot.

Runtime stores source/shuffle hashes and moved-position rate and aborts if length
or multiset changes. Every evaluated shuffled/no-thought row must be finite.
Close-only is diagnostic because its decoding differs. Real, shuffled, and
no-thought slots are the load-bearing matched comparison.

## 7. Power plan

From the terminal parent's 16 task-level real-minus-shuffled differences:

- mean = 0.0833333;
- sample SD = 0.3548604;
- one-sided alpha = 0.05; and
- target power = 0.80.

The normal planning approximation requires
`ceil(((z_.95 + z_.80) * SD / mean)^2) = 113` tasks. Both seam stages use 113,
for approximate power 0.802745. This is power for the parent's observed primary
effect, not a promise of replication and not power for the no-thought gap. The
actual gate is a 10,000-resample nonparametric task bootstrap. Runtime verifies
the power receipt, parent-analysis hash, and configured task counts.

## 8. Qualification gate

On 113 tasks/339 traces at fixed cap 1,024, all of the following must pass:

- real exact accuracy in `[0.20, 0.70]`;
- at least 28 tasks with both correct and incorrect real traces;
- real accuracy minus no-thought accuracy `>= 0.03`;
- real accuracy minus shuffled accuracy `>= 0.05`;
- one-sided 95% task-bootstrap lower bound for real-minus-shuffled `> 0`;
- correct real rows span at least eight distinct gold aliases;
- real chosen aliases span at least eight aliases;
- full-vocabulary top-is-alias rate `>= 0.75`;
- mean total full-vocabulary alias probability mass `>= 0.50`; and
- finite real-row rate exactly 1.0, with all evaluated controls finite.

The mixed-task floor scales the parent's 6/16 requirement conservatively to
28/113 (24.8%, below the parent's observed 31.25%). Diversity gates prevent a
few easy aliases from creating the headline. The interface gates establish that
the mask is not choosing among negligible alternatives. Any miss yields terminal
`POWERED_COMMIT_SLOT_SEAM_FAIL` and seals confirmation/J stages.

## 9. Untouched confirmation

Only a qualification pass opens the 113 untouched tasks. Verify all five raw
qualification file hashes and summary before model loading. Run the identical
fixed cap, traces, controls, bootstrap, and every threshold above. No threshold
relaxes. Passing yields `POWERED_COMMIT_SLOT_SEAM_REPLICATED`; failure yields
`POWERED_COMMIT_SLOT_SEAM_NOT_REPLICATED`.

Qualification and confirmation may be described side by side but may not be
pooled to rescue either miss. No additional split, seed, cap, decoder, alias
mapping, or subgroup may rescue the registered decision.

## 10. J/value boundary

Value-fit and causal task files are reserved to prove freshness but remain
unopened. CLI stages `prefix-value`, `control-calibration`, and
`causal-confirmation` raise a fatal unavailable error. Only replicated seam
label `POWERED_COMMIT_SLOT_SEAM_REPLICATED` licenses a later code commit and a
new outcome-blind implementation audit.

If opened later, the inherited direction remains: deterministic gold-alias
probability at 0.5/1.0 of the fixed cap, task-held-out J ranking beyond
correct-alias activity and ordinary margin, then exact post-bf16 scalar/random/
identity/non-J causal controls. Those rules must be audited again; this seam
preregistration does not manufacture a placeholder pass.

## 11. Interpretation labels

- Qualification miss: `POWERED_COMMIT_SLOT_SEAM_FAIL`.
- Qualification pass, confirmation miss:
  `POWERED_COMMIT_SLOT_SEAM_NOT_REPLICATED`.
- Both pass: `POWERED_COMMIT_SLOT_SEAM_REPLICATED`.

A replicated seam means ordered thought contributes task-general information to
the constrained slot. It does not prove internal certainty, J-space causality,
natural/free-form answering, or installed capability.

## 12. Capability and compute boundary

Gold answers evaluate gates, although the deployed slot/control policies are
label-free. Report native generated tokens, slot/control prefill tokens,
free-form answer tokens, forward calls, runtime, and peak memory. This study has
no matched-sampling capability endpoint. Any successor controller must beat
frozen and matched-compute sampling on fresh procedural tasks under one backend.

## 13. Stage and artifact discipline

Immutable design -> CPU/power receipts -> outcome-blind model smoke -> one
qualification -> at most one untouched confirmation. Every stage hash-unlocks
the next. Exact row counts are 339 traces, 339 real slots, 339 shuffled slots,
339 free-form controls, and 113 no-thought rows per seam stage. Scientific raw
files and summaries are written only after cache, finite, multiset, and
cardinality checks pass.

Preserve negatives and controls, update all three programs/synthesis at terminal
gates, synchronize before pushes, and reserve no claim ID during the claim
re-grade.

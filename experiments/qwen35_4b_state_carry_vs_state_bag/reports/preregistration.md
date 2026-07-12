# Preregistration: State-Carry Versus State-Bag

Frozen before any model-bearing call. CPU generation, code review, tests, and repository validation may precede this design commit. The live model-smoke is mechanics evidence only and cannot tune scientific thresholds.

## 1. Fixed Scientific Question

Does carrying one state serially through tied applications of Qwen's middle layers produce a deeper representation than aggregating the same number of independently computed shallow states?

A raw K benefit is insufficient. The causal contrast is separately trained Carry versus separately trained Bag at equal parameter count and decoder-layer token applications.

## 2. Model and Backend

- Only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Transformers 5.13.0, bf16, SDPA, cache-free full-sequence forward.
- Prelude layers 0–11, recurrent block 12–19, coda 20–31.
- The live model must expose exactly 32 text layers with repeating `linear,linear,linear,full` types.
- Every result-bearing arm and comparator uses Transformers. No vLLM result enters this experiment.
- At least 44 GiB exposed VRAM and both Qwen fast paths are mandatory.

## 3. Architecture Contract

The prompt contains eight single-token state slots before `Query:`. Causal masking therefore prevents every state slot from observing query type, choices, or answer.

The first middle-block application has all recurrence LoRA disabled. At K=1 the manual forward must match the standard CausalLM answer-position logits with maximum absolute error at most `1e-5`. Failure is terminal `MODEL_PATCH_PARITY_FAIL` until code is corrected; the tolerance cannot be relaxed.

For K>1:

- Carry call `t` receives state from call `t-1`.
- Bag call `t` receives the same initialized first-pass state, never another branch's state.
- Both use identical fixed sinusoidal step encoding, trainable projection, damping, low-rank state initializer, last-plus-mean aggregator, recurrence-only LoRA, and auxiliary heads.
- Non-state hidden positions reset to the untouched first-pass middle representation after every extra call.
- Coda executes exactly once.

Carry and Bag are separately trained from the same initialization seed and data order. The implementation records exact trainable names, counts, initial-value hashes, prompt-token totals, and decoder-layer-token totals; analysis refuses any mismatched pair.

## 4. Procedural Task and Splits

Each fresh world has 16 randomly named nodes. Node records contain left/right edges, a toggle, and weight. The evolving joint state is `(node, phase, checksum mod 8)`.

Train:

- families `phase_branch` and `checksum_branch`;
- surface templates `ledger` and `prose`;
- semantic depths 1–4;
- 12,000 deterministic examples.

Untouched evaluation:

- 1,024 validation examples in the training domain;
- 3,200 depth-extrapolation examples at depths 5–12;
- 3,200 held-out `braided_branch` examples;
- 3,200 held-out `compact` renderings;
- 3,200 joint family+rendering+depth holds;
- 512 matched counterfactual world pairs.

Generation rejects any trajectory that repeats a complete joint state or whose terminal queried value occurred at an earlier step. Structural fingerprints exclude surface labels and must have zero unintended cross-split overlap. Compressed archives have frozen gzip headers so hashes reproduce byte for byte. No benchmark file is read.

## 5. Supervision

At every trained iteration a shared head predicts the exact current node, phase, and checksum. Because query type occurs later in the causal sequence and is not an input to these heads, this is query-after-state sufficiency pressure rather than final-answer-only supervision.

The final frozen LM head predicts one of four distinct single-token answer letters. Loss weights are answer `1.0`, mean state loss `0.5`, and post-terminal fixed-point loss `0.05`. These values and the trained K=4 horizon are not tuned on extrapolation data.

## 6. Staged Run and Stop Rules

### G0: mechanics

Require model identity/revision, layer pattern, tokenizer contracts, exact K=1 parity, Carry/Bag parameter equality, nonzero LoRA/state/sufficiency gradients, and successful K=4 forward shapes. Any failure blocks training.

### G1: 300-step pilot

Train one Carry and Bag pair at seed 7411. Pilot evaluation uses 256 total depth items, paired K=4 and K=semantic-depth, a small K curve, joint holdout, and 64 swaps.

Promote only if all are true:

- no collapse or nonfinite loss;
- exact checkpoint/application receipts;
- Carry-minus-Bag is positive on the pilot depth pool;
- Carry state node accuracy is at least 0.60 or joint state accuracy at least 0.40; and
- the effect is not solely K=1 interface mass or parsing.

Pilot uncertainty is diagnostic. It cannot establish a positive or cause threshold changes. A miss triggers implementation/failure inspection, not seed shopping.

### G2: full continuous-state mechanism

Train Carry and Bag independently at seeds 7411, 7412, and 7413 for 1,500 steps. Select the fixed final checkpoint; validation cannot choose a favorable step. Every loaded final checkpoint must repeat the direct-model K=1 parity gate before evaluation. Evaluate the full registered cells.

Retained pilot bundles are excluded whenever full bundles exist. Every primary depth must contribute at least 1,000 pooled paired rows; a three-seed directory with incomplete task cells remains `UNDER_REPLICATED`.

`SERIAL_STATE_ADVANTAGE` requires:

- pooled paired Carry-minus-Bag at K=semantic-depth ≥ +0.05;
- paired 10,000-resample 95% lower bound > 0;
- all three training-seed pairs present;
- Carry K=semantic-depth beats its own K=4 on depths above four with lower bound > 0; and
- positive Carry-minus-Bag point estimates on at least six of the eight primary depths.

Failure is `NO_SERIAL_STATE_ADVANTAGE` or `TRAINED_UNROLLING_ONLY` as assigned by analysis. Carry must also reach mean per-step node accuracy 0.60 or joint node+phase+checksum accuracy 0.40 on matched-depth extrapolation; otherwise the label is `SERIAL_BUT_STATE_NOT_SUFFICIENT`. Raw answer accuracy cannot rescue these labels.

### G3: causal state

Evaluate every trained Carry checkpoint in Bag mode without retraining and report paired intact-minus-edge-cut accuracy by seed. Separately, swap a donor state at the midpoint between paired prompts sharing world, rule, label/table/choice order, depth, and query type but differing in initial state and terminal answer.

Require all three intact/edge-cut checkpoint pairs to be present and donor-follow rate minus recipient-preserve rate ≥ +0.10 for each seed. Generic accuracy damage, probe accuracy, or state decodability alone does not pass. Failure after G2 is `DEEP_BUT_NOT_CAUSALLY_IDENTIFIED`.

### G4: interface branch

The mixed semantic-echo config may be opened only if continuous Carry develops state decodability but misses final-answer or donor-use gates. Both mixed Carry and mixed Bag must be trained; echo Carry alone is uninterpretable. Shuffled/wrong-task echo evaluation must accompany any mixed-channel positive.

Mixed echo is a pre-registered branch, not permission to tune top-k or mixture after outcomes. A result explained entirely by mixed echo is an interface result, not automatically a deeper-representation result.

### G5: deployment comparator

Only after G2+G3 train the explicit textual state-trace LoRA on the identical procedural rows and loop-layer LoRA parameterization. At each item, allocate independent explicit-CoT samples so their total decoder-layer token applications do not exceed the recurrent arm's `P + K·R + C` budget.

Report majority accuracy, parse, actual sampled tokens, synchronized generation time, and exact-verifier oracle `pass@N`. A deployment claim requires all three seed-matched comparisons and a positive 10,000-resample hierarchical-bootstrap lower bound for Carry minus oracle `pass@N`; beating majority but not oracle with that criterion cannot be called a decisive sample-more win.

## 7. Statistics and Power

- Primary unit: paired fresh task, nested within training seed.
- Primary depth cells receive 400 tasks per seed (3,200 / 8); pooled across three seeds this is 1,200 paired observations per depth.
- Primary intervals: deterministic 10,000-resample paired bootstrap.
- Pilot and small K curves are diagnostics; only matched-depth full rows drive G2.
- Family/template holds are robustness gates, not opportunities to select architecture.
- Medium-size effects from one seed or a favorable checkpoint are provisional by construction.

## 8. Verdict Ladder

The terminal labels, in order, are:

1. `MODEL_PATCH_PARITY_FAIL`
2. `PILOT_MECHANISM_MISS`
3. `NO_SERIAL_STATE_ADVANTAGE`
4. `TRAINED_UNROLLING_ONLY`
5. `SERIAL_BUT_STATE_NOT_SUFFICIENT`
6. `DEEP_BUT_NOT_CAUSALLY_IDENTIFIED`
7. `MECHANISTIC_DEPTH_POSITIVE`
8. `MECHANISTIC_DEPTH_POSITIVE_SAMPLE_MORE_LOSS`
9. `DEPLOYABLE_DEPTH_BREAKTHROUGH`

Only the last licenses the full practical claim. A mechanistic positive is still fundamental evidence that serial organization changes what a fixed computation can represent.

## 9. Prohibited Repairs

Do not change the model, layer boundaries, state-slot count, train depths, families, thresholds, primary seeds, final-checkpoint rule, or backend after a result. Any such repair is a successor experiment. Preserve negative and stopped artifacts.

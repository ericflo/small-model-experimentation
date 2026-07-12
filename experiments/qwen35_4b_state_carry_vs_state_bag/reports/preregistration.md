# Preregistration: State-Carry Versus State-Bag

Frozen before any model-bearing call, then adversarially amended once—still before any model-bearing call—to repair pilot leakage, crossed-design inference, unenforced causal gates, cross-process nondeterminism, partial-checkpoint acceptance, swap geometry, and sample-more interface confounding. CPU generation, code review, tests, and repository validation may precede the amended design commit. The live model-smoke is mechanics evidence only and cannot tune scientific thresholds.

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

Carry and Bag are separately trained from the same initialization seed and data order. The implementation records exact trainable names, counts, initial-value hashes, an ordered row/K/token/compute digest, prompt-token totals, decoder-layer-token totals, critical-source digest, environment-lock digest, phase, and fixed-final step; analysis refuses any mismatched pair or stale artifact.

## 4. Procedural Task and Splits

Each fresh world has 16 randomly named nodes. Node records contain left/right edges, a toggle, and weight. The evolving joint state is `(node, phase, checksum mod 8)`.

Pilot-only firewall:

- training seed `7401`, excluded from every confirmatory estimate;
- 256 in-domain validation items from a dedicated structural seed, used only for pilot training logs and checkpoint parity;
- 256 depth-extrapolation and 256 joint-holdout items from dedicated structural seeds;
- 64 dedicated counterfactual pairs; and
- no pilot item or counterfactual pair appears in confirmation.

Train:

- families `phase_branch` and `checksum_branch`;
- surface templates `ledger` and `prose`;
- semantic depths 1–4;
- 12,000 deterministic examples.

Untouched confirmation:

- 1,024 validation examples in the training domain;
- 3,200 depth-extrapolation examples at depths 5–12;
- 3,200 held-out `braided_branch` examples;
- 3,200 held-out `compact` renderings;
- 3,200 joint family+rendering+depth holds;
- 512 matched counterfactual world pairs.

Node and checksum queries are scheduled in balanced family×template×depth cells rather than accepted from a rejection-skewed coin flip. Generation rejects any trajectory that repeats a complete joint state or whose terminal queried value occurred at an earlier step. Structural fingerprints exclude surface labels and must have zero unintended cross-split overlap, including pilot splits. Label construction is deterministic across Python hash seeds; compressed archives have frozen gzip headers. The manifest binds the critical generator-source contract. No benchmark file is read.

## 5. Supervision

At every trained iteration a shared head predicts the exact current node, phase, and checksum. Because query type occurs later in the causal sequence and is not an input to these heads, this is query-after-state joint-state tracking pressure rather than final-answer-only supervision. The mechanism gate requires mean joint node+phase+checksum step accuracy at least `0.40`; node accuracy alone cannot pass.

The final frozen LM head predicts one of four distinct single-token answer letters. Loss weights are answer `1.0`, mean state loss `0.5`, and post-terminal fixed-point loss `0.05`. These values and the trained K=4 horizon are not tuned on extrapolation data.

## 6. Staged Run and Stop Rules

### G0: mechanics

Require model identity/revision, layer pattern, tokenizer contracts, exact K=1 parity, shared-wrapper parameter identity, nonzero finite LoRA/state/step/sufficiency gradients for both Carry and Bag, successful K=4 backward, and worst-format K=12 evaluation forward with memory/timing receipt. The receipt binds config, critical source, and the training lock. Any failure blocks training in code, not only in prose.

### G1: 300-step pilot

Train one Carry and Bag pair at the pilot-only seed `7401`. Pilot evaluation uses only the dedicated 256-item depth split, paired K=4 and K=semantic-depth, the dedicated joint holdout, and 64 dedicated swaps.

Promote only if all are true:

- both trainings and evaluations complete without nonfinite loss, and exact checkpoint/application receipts pass;
- every registered matched-depth, K=4 diagnostic, joint-holdout, and bidirectional-swap pilot row is present with the exact expected identity;
- Carry-minus-Bag is positive on the pilot depth pool;
- Carry joint node+phase+checksum state accuracy is at least 0.40;
- both balanced query strata have positive Carry-minus-Bag point estimates; and
- Carry's full-vocabulary top token remains inside the registered A–D answer interface at least 95% of the time; and
- the confirmatory +0.05 absolute-gain gate is mathematically reachable given pilot Bag accuracy.

The pilot K=4 scaling, joint-holdout effect, and swap effect are diagnostic at this stage: their rows
must be complete, but their signs do not gate promotion.

Pilot uncertainty is diagnostic. It cannot establish a positive or cause threshold changes. Analysis emits `PILOT_PROMOTION_READY` or `PILOT_MECHANISM_MISS`; full training refuses to start without the former. A miss triggers implementation/failure inspection, not seed shopping.

### G2: full continuous-state mechanism

Train Carry and Bag independently from scratch at confirmation seeds 7411, 7412, and 7413 for 1,500 steps. Select only the fixed final checkpoint; phase and exact step are machine-enforced and validation cannot choose a favorable step. Interrupted runs are non-resumable and must restart at step zero in a fresh attempt directory. Every loaded final checkpoint repeats direct-model K=1 parity before evaluation. Evaluate all 400 generated rows per primary depth at K=4 and matched depth; small K-curve cells use the fixed 64-row diagnostic subset. The 3,200-row family/template/joint holdout corpora are generated and preserved, while evaluation scores the fixed first 128 rows per depth (1,024 per holdout split and seed).

Retained pilot bundles are excluded whenever full bundles exist. Every primary depth must contain the exact common matrix of 400 unique tasks crossed with all three seeds (1,200 model×task rows); any missing/extra/duplicate task cell remains `UNDER_REPLICATED`.

`SERIAL_STATE_ADVANTAGE` requires:

- pooled paired Carry-minus-Bag at K=semantic-depth ≥ +0.05;
- paired 10,000-resample 95% lower bound > 0;
- all three training-seed pairs present;
- Carry K=semantic-depth beats its own K=4 on depths above four with lower bound > 0; and
- positive Carry-minus-Bag point estimates on at least six of the eight primary depths.
- positive Carry-minus-Bag point estimates in both balanced query strata; and
- a positive joint family+surface holdout effect with crossed-bootstrap lower bound above zero.

Failure is `NO_SERIAL_STATE_ADVANTAGE`, `TRAINED_UNROLLING_ONLY`, or `DEPTH_NOT_ROBUST` as assigned by analysis. Carry must also reach mean joint node+phase+checksum step accuracy 0.40 on matched-depth extrapolation; otherwise the label is `SERIAL_BUT_STATE_NOT_SUFFICIENT`. Raw answer accuracy or node-only decoding cannot rescue these labels.

### G3: causal state

Evaluate every trained Carry checkpoint in Bag mode without retraining, on the complete primary matched-depth cells only. The intact and cut bundles must have identical canonical checkpoint identity, task keys, prompt/compute receipts, and fixed-final step. Separately, swap a donor state at the midpoint in both directions between paired prompts sharing world, initial node, rule, label/table/choice order, token/state-slot geometry, depth, and query type but differing in phase/checksum and terminal answer.

Require all three intact/edge-cut checkpoint pairs, positive intact-minus-cut point estimates in every seed, and a crossed-bootstrap lower bound above zero. For all 512 confirmation pairs per seed (1,024 directed swaps), require post-swap donor following minus pre-swap donor-choice following and donor-follow minus recipient-preserve to be at least +0.10 per seed, with raw rows hashed and reanalyzed. Generic accuracy damage, probe accuracy, or state decodability alone does not pass. Failure after G2 is `DEEP_BUT_NOT_CAUSALLY_IDENTIFIED`.

### G4: deployment comparator

Only after G2+G3 train the explicit textual state-trace LoRA on the identical procedural rows and loop-layer LoRA parameterization. Sampling is frozen at temperature `0.6`, top-p `0.95`, and top-k `20`. For depth `d`, each sample must receive at least `32 + 24d` generated-token allowance; choose the largest `N≤8` that meets that floor, then use the remaining per-sample budget up to 512 tokens. Total allocated decoder-layer token applications may not exceed the recurrent arm's `P + K·R + C` budget.

Preserve raw decoded completions and token IDs. Report majority accuracy, natural-close/parse/cap-contact rates by depth, actual sampled tokens, synchronized generation time, and exact-verifier oracle `pass@N`. A deployment comparison is invalid unless Carry's full-vocabulary top token is one of A–D on at least 95% of primary rows, explicit-CoT parse is at least 95%, and cap contact is at most 5%. It requires exactly 3,200 common task IDs for each of all three seed pairs and independently rechecked compute budgets. A deployment claim requires a positive 10,000-resample crossed task×seed bootstrap lower bound for Carry minus oracle `pass@N`; beating majority but not oracle cannot be called a decisive sample-more win.

## 7. Statistics and Power

- Primary design: the same 400 unique paired fresh tasks per depth are crossed with three independent training seeds.
- Primary intervals: deterministic 10,000-resample crossed bootstrap, resampling task IDs once across all sampled seeds and resampling training seeds separately.
- Reports distinguish 400 unique tasks from 1,200 model×task rows; they never call the latter independent observations.
- Pilot and small K curves are diagnostics; only matched-depth full rows drive G2.
- Separate family/template holds are diagnostics; the joint family+template+depth holdout is the preregistered robustness gate. None may select architecture.
- Medium-size effects from one seed or a favorable checkpoint are provisional by construction.

## 8. Verdict Ladder

The terminal labels, in order, are:

1. `MODEL_PATCH_PARITY_FAIL`
2. `PILOT_MECHANISM_MISS` (or nonterminal `PILOT_PROMOTION_READY`)
3. `NO_SERIAL_STATE_ADVANTAGE`
4. `TRAINED_UNROLLING_ONLY`
5. `SERIAL_BUT_STATE_NOT_SUFFICIENT`
6. `DEPTH_NOT_ROBUST`
7. `DEEP_BUT_NOT_CAUSALLY_IDENTIFIED`
8. `MECHANISTIC_DEPTH_POSITIVE`
9. `SAMPLE_MORE_INTERFACE_INVALID`
10. `MECHANISTIC_DEPTH_POSITIVE_SAMPLE_MORE_LOSS`
11. `DEPLOYABLE_DEPTH_BREAKTHROUGH`

Only the last licenses the full practical claim. A mechanistic positive is still fundamental evidence that serial organization changes what a fixed computation can represent.

## 9. Prohibited Repairs

Do not change the model, layer boundaries, state-slot count, train depths, families, thresholds, pilot/primary seeds, final-checkpoint rule, sampling/allocation policy, or backend after a result. Do not open the removed mixed-echo branch in place. Any such repair is a successor experiment. Preserve negative and stopped artifacts.

## 10. Mandatory LoRA-Capacity Resolution

Rank-32 loop-only LoRA is the first registered intervention because it is cheap enough to run paired,
keeps the first pass exactly frozen, and can modify every discovered linear projection in both complete
Qwen motifs. Together with the full-width repeated hidden state, trainable state initializer, step
projection, damping, and dense joint-state loss, it has a plausible path to learning the transition.
It is not a proof that low rank is sufficient.

Therefore a valid LoRA outcome that fails to establish deep state formation does not close the
serial-state question. It mandates creating **and executing** a fresh successor experiment with the
same model, substrate logic, Carry/Bag edge, pilot firewall, crossed analysis, and causal gates, but
replaces extra-call LoRA with zero-initialized **full-rank weight deltas** on layers 12–19. The base
weights remain stored/frozen; deltas are enabled only on extra R applications, so the untouched first
R application and K=1 logits remain exact. Both successor arms receive identical deltas/parameter
counts and separate optimization. This tests the low-rank bottleneck without confounding the ordinary
path with full fine-tuning.

Mechanics or data-integrity failures must be repaired in place, and a mathematically infeasible gate
requires design review; neither is evidence about LoRA capacity. If joint state is strongly readable
but causal consumption alone fails, LoRA has already formed the representation and the separately
controlled interface successor is the sharper mandatory resolution instead of full rank.

If LoRA reaches causal mechanistic depth but loses only the sample-more comparison, low rank has
already formed the relevant deeper representation and no capacity successor is licensed by that
economic loss. The report must classify the observed signature under this decision rule rather than
calling every pre-deployment miss a LoRA-capacity failure.

# Adversarial Design Review

Completed before any model-bearing call. The review assumes the attractive story is wrong and asks how the experiment could manufacture a “deep representation” from compute, supervision, leakage, instability, or weak controls.

## Verdict

**Proceed only through the live model-smoke gate.** The design is capable of falsifying its central mechanism because Carry and Bag differ by one edge in the computation graph while matching parameters, training, data, calls, and readout. The remaining irreducible risks—whether frozen Qwen state is usable, whether 48 GiB is enough, and whether recurrence optimization converges—require the target GPU and are explicitly gated rather than hand-waved.

## 1. Extra compute is mistaken for depth

If Carry is compared only to K=1, any gain is “more R calls.”

**Hardening:** the primary control is a separately trained State-Bag with the same number of R calls and the same coda/aggregator. K=1 and dummy/static arms are diagnostics only.

## 2. Bag is deliberately crippled

Resetting every branch without step identity would make K copies identical.

**Hardening:** both arms receive the same sinusoidal step encoding, projected by the same trainable map that is mathematically defined beyond K=4. Bag receives independent t-indexed shallow representations and the same mean-plus-last aggregator.

## 3. Carry has more parameters

Different modules or LoRA targets could explain the result.

**Hardening:** Carry and Bag instantiate the identical wrapper; mode changes only the source tensor used at an edge. Model smoke hashes trainable names/counts, and checkpoints store the receipt.

## 4. Carry gets a better optimization schedule

Independent runs could quietly differ in examples or steps.

**Hardening:** paired seeds, deterministic generated rows, same shuffle algorithm, fixed final checkpoint, identical optimizer/loss/K=4/steps. The arm is the only intended difference.

## 5. The natural-language query leaks into state

Then “query-after-state” is branding.

**Hardening:** all state tokens appear before the literal `Query:` substring; the live tokenizer verifies their token indices precede the query boundary. Qwen masking is causal in both GDN and full-attention layers.

## 6. Residual tokens bypass the state bottleneck

If the whole sequence carries across loops, the state claim is false.

**Hardening:** after every extra R application all non-state positions reset to the untouched first-R memory. Only eight state-slot tensors cross loop depth.

## 7. LoRA changes the K=1 base path

Then quick retention and recurrence attribution are confounded.

**Hardening:** recurrence LoRA is disabled for the first P/R pass and coda. K=1 must match the standard model's answer-position logits to `1e-5`; the run aborts otherwise.

## 8. Manual forward subtly differs from Transformers

Mask, position, cache, or normalization drift can create artifacts.

**Hardening:** the wrapper reproduces the pinned Transformers 5.13.0 Qwen text forward geometry: four-axis position IDs, model-provided rotary embeddings, `create_causal_mask`, `create_recurrent_attention_mask`, cache-free layers, and the original final norm/LM head. Live direct-forward parity is mandatory.

## 9. An incomplete Qwen motif is looped

Looping only attention or only GDN could privilege a local artifact.

**Hardening:** boundaries 12 and 20 align to two complete native `[GDN,GDN,GDN,attention]` motifs, with nonempty 12-layer prelude and coda.

## 10. A small bolt-on repeats the prior negative

The prior fast-weight hook already tested thin recurrence.

**Hardening:** R is eight full-width pretrained Qwen layers. The small modules initialize, stabilize, and measure state; they do not replace repeated Qwen computation.

## 11. Final-answer loss never identifies a state

This was the central weakness of the prior hook.

**Hardening:** a shared head predicts the complete node/phase/checksum state after every step, plus fixed-point behavior after the requested depth. Final-answer-only is not the primary training recipe.

## 12. Dense supervision merely installs a task-specific VM

Then success may not reveal a general representational principle.

**Hardening:** the primary claim is deliberately mechanistic, not universal. A held-out transition family, surface form, their conjunction, and unseen depth test whether the stationary update transfers. Any positive confined to trained families is scoped accordingly.

## 13. The task has algebraic shortcuts

Then K scaling need not reflect serial computation.

**Hardening:** every world is a fresh random pointer graph; branching depends on the live state. Repeated complete states are rejected. There is no shared transition table to memorize, and symbol skins change per item.

## 14. “Minimum depth” is overclaimed

A transformer might pointer-double or compute multiple transitions per block.

**Hardening:** call it semantic transition depth, not a circuit lower bound. The decisive evidence is empirical scaling and Carry-vs-Bag, not a theorem that one R call equals one transition.

## 15. Data contamination creates extrapolation

The same graph or task could cross splits under different labels.

**Hardening:** structural fingerprints exclude surface labels and choices, and generation asserts zero unintended cross-split overlap. Seeds are disjoint. No public benchmark or generated benchmark artifact is read.

## 16. The answer interface creates the gain

Repeated work slots or letter priors might improve formatting.

**Hardening:** all arms see identical slots and answer letters. Report full-vocabulary top-is-answer and total answer-letter mass alongside constrained accuracy. K=1 interface behavior cannot establish depth.

## 17. State decodability is called causality

Repository evidence already shows readable but inert state.

**Hardening:** decoder accuracy is necessary but insufficient. A trained Carry checkpoint must fail when its edge is cut, and matched donor state must cause donor-consistent consequences.

## 18. Swaps are generic damage

Any activation replacement can lower recipient accuracy.

**Hardening:** paired prompts share the exact world/rule/query and differ only in initial state. The outcome must follow the donor answer, not merely leave the recipient answer.

## 19. Donor state names do not exist in the recipient world

Then donor following is impossible for superficial reasons.

**Hardening:** counterfactual pairs share the full world, label mapping, table order, query, and choice order. Each recipient's choices contain the donor's terminal value; only the stated initial state and correct answer differ.

## 20. Semantic echo is presented as novel recurrence

DiscoLoop already diagnoses embedding alignment.

**Hardening:** continuous Carry-vs-Bag is primary. Mixed echo is a branch for “represented but unusable,” and any echo-only result is labeled an interface result. Mixed Carry must be paired with mixed Bag.

## 21. Naive repeated blocks diverge

Recent work finds naive loops often degrade.

**Hardening:** damped updates initialize at 0.125, extra-step projection initializes at zero, and post-terminal fixed-point loss trains halting. Overthinking curves at K=4/8/12 are registered.

## 22. Damping turns recurrence into a negligible perturbation

The system could remain near K=1 and still train the answer head.

**Hardening:** report learned damping, state deltas, state-head trajectories, and K-specific behavior. State modules alone cannot pass Carry-vs-Bag or donor transport.

## 23. Favorable checkpoints are selected

Noisy early K gains fooled the prior hook.

**Hardening:** the fixed final checkpoint is primary. Intermediate validation is operational and cannot select the scientific checkpoint.

## 24. A favorable seed is promoted

One pair can produce an unstable story.

**Hardening:** the pilot is explicitly non-evidentiary. G2 requires all three predeclared seed pairs and pooled paired intervals.

## 25. The primary cell is underpowered

Depth subdivision can leave small n despite thousands pooled.

**Hardening:** 3,200 depth items give 400/depth/seed; three seeds give 1,200 paired observations per depth. Small K curves use 64/depth and are diagnostic only.

## 26. Evaluation cost makes the protocol practically unrunnable

An exhaustive 5-split × 12-K grid would burn the budget before the primary answer.

**Hardening:** full matched-depth and K=4 rows receive the full depth sample; other K values use 64/depth. Robustness holds run only at their matched depth. Pilot is smaller and gates promotion.

## 27. Sample-more is weak or backend-confounded

Comparing trained recurrence against raw one-token sampling would be meaningless.

**Hardening:** train an explicit textual state-trace LoRA on the identical procedural rows and loop-layer LoRA parameterization. Generate independent CoTs with Transformers and report exact-verifier `pass@N` plus majority under a layer-token budget no larger than recurrence.

## 28. Layer-token compute is not true FLOPs

Prefill attention and recurrent GDN costs are not identical per token.

**Hardening:** declare decoder-layer token applications as the primary allocation unit and report measured GPU seconds/tokens as diagnostics. Do not claim hardware-normalized equality beyond this unit.

## 29. The explicit baseline sees latent placeholders

That would both confuse it and inflate its prompt compute.

**Hardening:** text-baseline rendering removes the workspace heading and tokens. Recurrence compute uses the actual recurrent prompt length; text compute uses its actual visible prompt length.

## 30. Backend mixing reappears for throughput

The repository has repeatedly found HF/vLLM nonparity.

**Hardening:** config validation rejects any backend but `transformers`; scaffold vLLM code was deleted; static tests assert it stays absent.

## 31. Qwen or PEFT version drift silently changes module names

LoRA might target the wrong layers or nothing.

**Hardening:** environment and Transformers version are pinned. Targets are discovered from actual `nn.Linear` modules within layers 12–19, then every trainable LoRA name is audited back to that interval.

## 32. The 48 GiB run OOMs and leaves misleading partial output

Repeated full blocks retain substantial activations.

**Hardening:** batch one, bf16, frozen first pass, gradient accumulation, expandable segments, external checkpoints, and single-GPU exclusivity. Model smoke includes a K=4 backward and peak-memory receipt before training.

## 33. An OOM corrupts CUDA and retries change conditions

Repository history documents persistent CUDA corruption.

**Hardening:** follow `docs/compute_environment.md`; do not launch concurrent jobs. A failed model smoke is repaired before training, and partial directories are never overwritten silently.

## 34. Pilot optional stopping becomes architecture search

Repeated tweaks after every miss could guarantee a winner.

**Hardening:** pilot promotes or stops the continuous design. Mixed echo has one predeclared trigger and fixed config. Other repairs require successor experiments.

## 35. A result is claimed before sample-more

Mechanistic interest can blur the repository's deployment standard.

**Hardening:** verdict ladder separates mechanistic depth from deployment. Only `DEPLOYABLE_DEPTH_BREAKTHROUGH` requires and clears sample-more.

## 36. Setup artifacts are mistaken for evidence

The implementation is elaborate and may feel like progress on the hypothesis.

**Hardening:** README/report state `SETUP_ONLY`; CPU smoke and model smoke record `scientific_evidence: false`. No claim ledger update is reserved.

## 37. A nominally deep answer is already correct at a shallower step

A trajectory may never repeat its complete `(node, phase, checksum)` state while its eventual queried node or checksum nevertheless occurred earlier. Such an item would overstate the minimum computation needed for the measured answer.

**Hardening:** generation and verification reject both complete-state repeats and any earlier occurrence of the terminal queried field. Counterfactual donor pairs apply the same rule to both members before fixing their shared choices.

## 38. Pilot rows leak into the full conclusion

Pilot and full evaluations share config hash, seed, arm, and task IDs. A directory scan could silently overwrite a full bundle with a smaller pilot or pool both.

**Hardening:** every evaluation summary carries a pilot flag. Analysis uses only non-pilot bundles once any exist and rejects duplicate full arm/seed cells.

## 39. Equal training is asserted but not auditable

Matching seeds and code are insufficient if one run sees different rows, prompt lengths, trainable initialization, or call totals.

**Hardening:** each checkpoint records trainable name/count/value hashes plus cumulative prompt-token and decoder-layer-token totals. Analysis refuses a Carry/Bag seed pair unless all registered equality receipts match.

## 40. A weak point estimate beats sample-more by noise

Pooling Carry and explicit-CoT results without matching task and training seed can create an apparent deployment win, especially if only one comparator seed completed.

**Hardening:** the deployable verdict requires three seed-complete task pairs and a positive hierarchical-bootstrap lower bound for Carry minus exact-verifier oracle `pass@N`. Partial sample-more leaves the narrower mechanistic label in place.

## 41. Retained artifacts are corrupt, stale, or host-relative

Checkpoint metadata can look valid while tensors or row files moved, changed, or resolve relative to a different working directory.

**Hardening:** summaries store row filenames relative to themselves; analysis verifies row hashes. Model reload verifies model/config identity and every adapter/loop-state hash, while data loading verifies the clean manifest and data-contract hash.

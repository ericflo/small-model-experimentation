# Adversarial Design Review

Completed before any model-bearing call. The review assumes the attractive story is wrong and asks how the experiment could manufacture a “deep representation” from compute, supervision, leakage, instability, or weak controls.

## Verdict

**Proceed only through the live model-smoke gate after the amended CPU checks pass.** The design is capable of falsifying its central mechanism because Carry and Bag differ by one edge in the computation graph while matching parameters, training, data, calls, and readout. The final adversarial pass below repaired pilot/confirmation leakage, crossed-design inference, causal-gate, artifact-identity, and baseline-interface failures before any model call. The remaining irreducible risks—whether frozen Qwen state is usable, whether 48 GiB is enough, and whether recurrence optimization converges—require the target GPU and are explicitly gated rather than hand-waved.

## 1. Extra compute is mistaken for depth

If Carry is compared only to K=1, any gain is “more R calls.”

**Hardening:** the primary control is a separately trained State-Bag with the same number of R calls and the same coda/aggregator. K=1 parity is a mechanics diagnostic only; the orphaned static-LoRA arm was removed before launch.

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

**Hardening:** the final pre-run review removed mixed echo from this result-bearing experiment. A “represented but unusable” outcome may motivate a fresh successor with paired Carry/Bag plus shuffled/wrong-task controls; it cannot branch this confirmation in place.

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

**Hardening:** the pilot is explicitly non-evidentiary and now uses seed 7401 plus dedicated pilot-only depth/joint/counterfactual splits. G2 uses different seeds and tasks, requires all three predeclared seed pairs, and uses crossed task×seed intervals.

## 25. The primary cell is underpowered

Depth subdivision can leave small n despite thousands pooled.

**Hardening:** 3,200 depth items give 400 unique tasks/depth crossed with three training seeds. Inference resamples the 400 task IDs once across sampled seeds; it never treats 1,200 model×task rows as independent observations. Small K curves use 64/depth and are diagnostic only.

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

**Hardening:** pilot promotes or stops the continuous design. Every architecture or interface repair—including semantic echo—requires a successor experiment.

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

## 42. Pilot selection leaks into confirmation

The original pilot reused seed 7411 and confirmation rows, then required a favorable sign to proceed.

**Hardening:** pilot seed 7401 is excluded from confirmation, and dedicated `pilot_validation`,
`pilot_depth`, `pilot_joint`, and `pilot_counterfactual` splits are inside the structural-duplicate firewall. The
analyzer emits a machine promotion decision that full training requires.

## 43. Identical tasks are pseudoreplicated across model seeds

The original nested bootstrap independently resampled the same task set within every model seed.

**Hardening:** the design is explicitly crossed. Task IDs are sampled once per replicate and shared
across sampled training seeds; reports distinguish unique tasks from model×task rows and require the
complete common task matrix.

## 44. An edge-cut file exists but the edge need not matter

The original verdict checked for three cut bundles without requiring intact accuracy to exceed cut.

**Hardening:** same-checkpoint identity, exact paired keys/compute, positive per-seed effects, complete
cells, and a crossed-bootstrap lower bound above zero are now causal gates. Identical intact/cut rows
must terminate as `DEEP_BUT_NOT_CAUSALLY_IDENTIFIED`.

## 45. Pilot and intermediate checkpoints can masquerade as full

The caller-controlled `--pilot` flag originally overrode checkpoint provenance.

**Hardening:** checkpoint metadata binds `pilot`/`full`/`text`, exact registered seed, and exact final
step. GPU loading and analysis both reject a phase/step mismatch. Interrupted training cannot resume
approximately or donate an intermediate checkpoint; it restarts from step zero in a new attempt.

## 46. Old code can consume new-looking artifacts

Configuration hashes did not change when generator or recurrence source changed.

**Hardening:** data, model-smoke, checkpoints, evaluations, and sample-more bind a versioned critical-
source digest plus the pinned environment-lock digest. A canonical checkpoint identity also binds
tensor hashes, seed, arm, phase, and step.

## 47. Corpus hashes depend on `PYTHONHASHSEED`

Random labels were accumulated in a set and converted directly to a list.

**Hardening:** labels are sorted before the seeded shuffle, and a subprocess regression compares
archives under different Python hash seeds.

## 48. Deep accepted rows silently skew query type

Randomly choosing node/checksum before rejecting repeated terminal values gives the two query types
different acceptance rates.

**Hardening:** query kind is scheduled explicitly and balanced inside each family×template×depth
cell; manifests and scored rows retain it, and both query strata must show a positive mechanism effect.

## 49. State sufficiency can pass on node-only information

The original OR gate accepted node accuracy even if phase/checksum were absent.

**Hardening:** the gate requires joint node+phase+checksum correctness. Documentation now calls the
metric current-state tracking, not increasing terminal sufficiency, which was never directly tested.

## 50. Swaps mix position geometry and use only one direction

Different initial node labels could move slot positions, and one directed swap per pair left the
intervention undercontrolled.

**Hardening:** pair members share the initial node and differ only in fixed-width phase/checksum
values; runtime asserts identical prompt/state-slot geometry. Both directions are evaluated, raw rows
are hashed and reloaded, and post-swap donor following is compared with its pre-swap rate as well as
recipient preservation.

## 51. Registered joint holdouts cannot affect the verdict

A substrate-local effect could previously receive the strongest mechanistic label.

**Hardening:** the joint held-out family+surface endpoint must be positive with a crossed-bootstrap
lower bound above zero. Other holdouts remain reported diagnostics.

## 52. Sample-more can lose only because it cannot close or parse

Carry uses a constrained four-letter readout while explicit CoT originally needed a natural close
and parse under an arbitrary 64-token minimum and 256-token cap.

**Hardening:** sampling parameters and a depth-aware allowance are frozen. Raw tokens/text, natural
close, parse, cap contact, and independently checked compute are preserved. No deployment verdict is
available unless Carry is in answer mode on at least 95% of rows, explicit CoT parses at least 95%,
and cap contact is at most 5%; failure is an interface-invalid baseline, not a recurrence win.

## 53. Long jobs have no exact resume

Optimizer, RNG, and data cursor were not checkpointed, while intermediate adapters looked loadable.

**Hardening:** this experiment intentionally chooses the simpler exact policy: result-bearing training
is non-resumable, only fixed final checkpoints are saved/evaluable, partial attempts are preserved,
and an interruption restarts from step zero in a fresh directory. Shell loops fail on the first error.

## 54. The registered run is much larger than G0 represents

One fixed K=4 smoke did not cover worst prompt geometry or K=12 evaluation, and saving every 100
steps would create 90 full adapters.

**Hardening:** G0 includes a worst-format K=12 forward and timing/memory receipt; `save_every_steps`
is the fixed final 1,500 step (pilot still saves its final 300 step); edge-cut evaluation runs only the
primary cells it analyzes. The runbook records the static workload geometry and requires pilot timing
before projecting the full run.

## 55. A reduced config can counterfeit a terminal verdict

Config hashes prove provenance but do not make a two-step, twelve-row smoke run scientific evidence.

**Hardening:** the exact default confirmatory config has a pinned canonical digest. Every model-bearing
entry point rejects any other geometry, and analysis emits `NONCONFIRMATORY_SMOKE_ONLY` for reduced
profiles. A regression test covers both boundaries.

## 56. Pilot metrics still read confirmation validation

Dedicated extrapolation splits did not prevent pilot validation logs and checkpoint parity from
reading the first confirmatory validation rows.

**Hardening:** a separately seeded `pilot_validation` split is inside the global structural firewall.
Pilot training and checkpoint parity use it exclusively; the 1,024 confirmatory validation rows are
unopened until G2.

## 57. Two swap directions are not independent tasks

Bootstrapping 1,024 directed interventions would double-count the 512 shared worlds and understate
uncertainty.

**Hardening:** analysis averages the two directions within each pair, bootstraps 512 pair means, and
still requires both donor-follow-minus-baseline and donor-follow-minus-recipient-preserve to exceed
+0.10 in every seed.

## 58. Frozen fields and diagnostics can become unearned claims

Unused mixed-interface scalars, config gates, and a supported but unscheduled static-LoRA arm made the
registered surface larger than the experiment actually analyzed.

**Hardening:** dead semantic-echo parameters, unused config fields, unreachable mixed verdicts, and the
orphaned static arm were removed. Remaining boolean invariants are frozen and enforced rather than
presented as configurable no-ops.

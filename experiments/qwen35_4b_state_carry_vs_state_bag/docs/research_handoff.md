# Research Handoff: Reconstructing the Current State of Mind

This is the primary continuity document. Read it after the repository operating guide and before editing code or launching the GPU.

## The Core Insight

The scientifically interesting variable is not “number of transformer calls.” It is **whether call `t+1` receives a representation created by call `t`**.

Most recurrence comparisons leave an escape hatch: perhaps extra FLOPs help, perhaps an ensemble helps, perhaps a step-conditioned shallow network directly predicts the answer, perhaps dense supervision teaches a special-purpose executor, or perhaps a decodable state is never used. State-Carry versus State-Bag closes the first three escape hatches with one graph edge:

```text
Carry: state_t -> R_(t+1)
Bag:   state_1 -> R_(t+1)
```

Everything else is matched. This is the closest operational test of “deep representation versus an ensemble of shallow representations” I could derive.

## What “Deeper Representation” Means Here

It does not mean that a linear probe becomes more accurate at a later layer. It means all of the following:

1. A single bounded state becomes more sufficient for multiple future queries as recurrence proceeds.
2. Later computation requires the earlier state edge: cutting it removes the benefit even in the trained checkpoint.
3. Replacing that state with a semantically valid donor state causes the corresponding donor consequence.
4. The transition remains useful when executed more times than it was trained.
5. The effect exceeds an equal-compute set of independent shallow states.

Without all five, use a narrower label.

## Why Query-After-State Matters

If a model knows the final question while constructing its latent state, it can hide one answer logit rather than represent the world. The state slots here occur before query kind and choices. Training randomly asks about node, phase, and checksum through shared heads, while the coda later answers node or checksum in natural language.

This converts “latent thought” into an information-bottleneck question: can eight full-width vectors retain a task-sufficient joint state that a fixed Qwen coda can consume?

The auxiliary heads are deliberately shared across iterations. Separate heads per depth would permit four unrelated encodings and destroy the stationary-state hypothesis.

## Why the Prompt Memory Resets

Reapplying R to the entire residual sequence and carrying all positions would recreate the residual-ensemble ambiguity. Prompt tokens could accumulate a distributed computation even if the named state slots were irrelevant.

Therefore extra calls may read the whole first-pass memory, but only the state positions survive into the next call. This is analogous to a processor repeatedly reading immutable instructions/world memory while updating a bounded register file.

## Why K=1 Must Stay Original

C54 suggests static specialization trades quick and deep capability. A recurrence mechanism that rewrites the ordinary path could simply move along that same frontier.

Here recurrence LoRA and state initialization are activated only when K>1. K=1 is the same token sequence, original weights, original layer count, and original logits. If recurrence eventually helps deep tasks, the same runtime can choose K=1 on easy tasks rather than storing both regimes in one static adapter.

This property is strategically important, not just a unit test.

## Why the Primary State Is Continuous

A semantic decode/re-embed interface is plausible and may be necessary, but recent literature already makes it an expected engineering lever. Starting mixed would leave an ambiguous positive: did serial computation matter, or did a token-like interface merely make an already-computed concept consumable?

Continuous is therefore primary. Mixed echo is opened only by a recognizable signature:

- state heads improve with iteration;
- Carry exceeds Bag internally or under edge cutting;
- final answers or donor following do not improve enough.

That is the “represented but unusable” branch. If state heads themselves remain poor, echo is not a principled rescue.

## Why State-Bag Gets a Step Signal

A weak Bag would run the same shallow computation K times. That would test recurrence against duplication, not against an ensemble of shallow representations.

The sinusoidal step signal lets Bag learn distinct direct representations for different semantic steps and is defined beyond training K. Carry receives exactly the same signal. If Bag learns to map `(world, t)` directly to state `t`, that is a legitimate shallow solution and should defeat the deep claim.

## Why the Task Looks Like a Random State Machine

Arithmetic chains make numeric accuracy and tokenizer priors dominant. Public reasoning benchmarks invite contamination and weak causal state definitions. An external VM would make the executor, not Qwen, carry the structure.

Randomly skinned pointer worlds have:

- exact execution and arbitrary fresh scale;
- a known joint intermediate state;
- state-dependent transitions that resist a fixed parallel formula;
- counterfactual worlds suitable for causal swaps;
- no public training contamination; and
- multiple families/renderings for transfer.

Do not claim a formal transformer circuit-depth lower bound. The task supplies semantic transition depth and a strong random-world shortcut barrier; the empirical Carry/Bag result does the scientific work.

## How to Interpret Outcomes

### Carry and Bag both flat

Likely optimization/interface failure. Read state-head curves:

- state heads flat: initializer or training signal failed; do not open echo;
- state heads learn but answer flat: coda/interface failure; mixed echo branch is licensed;
- training loss falls but fresh state heads flat: memorization or format capture.

### Carry and Bag both improve equally

This is useful evidence against the strong claim. Extra tied computation or a shallow step-conditioned ensemble helps, but inherited state is not the lever.

### Carry beats Bag only through K=4

Call it trained unrolling. It may still be an effective architecture, but it has not learned a stationary deeper representation.

### Carry beats Bag beyond K=4, swaps fail

The serial edge matters, but the named state may be distributed, brittle, or nonsemantic. Label it deep-but-not-causally-identified; do not use probes to promote it.

### Carry passes mechanism but loses sample-more

This is still fundamental knowledge: organizing equal internal compute serially changes the representation. It is not yet the best use of compute.

### Carry beats Bag, extrapolates, transports donor consequences, and beats sample-more

This is the intended breakthrough signature. Replicate before broad claims, then test whether conditional K resolves the C54 quick/deep frontier on a separate contamination-safe capability instrument.

## Code Map

- `src/config.py`: immutable model/backend/design validation and config hashes.
- `src/substrate.py`: exact worlds, transitions, rendering, fingerprints, and counterfactual pairs.
- `src/data_pipeline.py`: split generation, hashes, dedup, manifests.
- `src/mechanics.py`: backend-free carry/bag and compute/statistical references.
- `src/state_loop_model.py`: manual Qwen forward and the single causal edge switch.
- `src/gpu_runner.py`: model contracts, PEFT locality, smoke, training, evaluation, swaps, and text comparator.
- `src/analysis.py`: paired bootstrap and fail-closed verdict ladder.
- `scripts/run.py`: stable entry point.
- `reports/preregistration.md`: frozen scientific contract.
- `reports/design_review.md`: adversarial failure analysis.
- `docs/gpu_runbook.md`: exact operational sequence.

## What the GPU Agent May Fix Before G0

Mechanical compatibility fixes are allowed if Transformers/PEFT APIs differ despite the pin:

- exact upstream import location for mask builders;
- PEFT path unwrapping or exact target-name matching;
- dtype/autocast compatibility;
- memory checkpointing that leaves forward values unchanged;
- chat-template argument spelling; and
- output/checkpoint path bugs.

Every fix must preserve K=1 parity, layer boundaries, adapter locality, state/query ordering, arm equality, data, and preregistered thresholds. Add a test and document the operational lesson.

## What Requires a Successor Experiment

- changing Qwen identity or revision;
- moving/reducing/expanding R;
- changing state slot count;
- changing train depth or families;
- training coda or first-pass LoRA in Carry/Bag;
- per-step untied recurrence parameters;
- selecting checkpoints from extrapolation outcomes;
- relaxing gates;
- adding a new substrate after results; or
- using a different backend for any comparator.

## First Commands

Do not start training immediately. The correct sequence is environment verification, CPU tests, deterministic data, live model smoke, receipt inspection, then one paired pilot. [`gpu_runbook.md`](gpu_runbook.md) contains copy-paste commands.

## The Research Attitude

This experiment is designed so a negative result remains a map:

- Bag parity says shallow representation ensembles suffice at this scale.
- No K extrapolation says recurrence memorized an unroll.
- State-readable/answer-inert says the interface is the bottleneck.
- Swap failure says the representation is correlational or distributed.
- Sample-more loss says serial state is real but economically inferior.

Preserve those distinctions. Do not rescue the story by collapsing them into one accuracy number.

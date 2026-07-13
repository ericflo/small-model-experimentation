# Preregistration: Semantic-Anchor Coordinate Branching

Frozen before any model call in this experiment.

## 1. Scientific question

The prior native branch experiment falsified centered additive J directions at
an arbitrary final thought token. The independent positive instead used an
explicit semantic token and replaced its context-local donor coordinates. This
experiment restores exactly those ingredients at a candidate-hypothesis anchor
after 512 tokens of native thought.

## 2. Fixed model and lens

- Only `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Transformers bf16 SDPA for all scientific arms. Mechanics is batch one,
  padding-free, and cache-free; no vLLM/HF arm mixing.
- Exact frozen lens SHA-256
  `e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`.
- All 24 normalized coordinates, layers `[4,5,6,7,8]`, pseudoinverse rtol
  `1e-5`. No refit, layer choice, amplitude choice, or post-result sweep.

## 3. Fresh task firewall

Generate four mechanics, 24 qualification, and 48 confirmation exact-depth-two
procedural tasks from seed `2026072401`. Hash complete visible+hidden behavior
and reject any collision with every readable ancestor procedural artifact.
Never read or import `benchmarks/` contents. Mechanics may load only public task
fields plus diagnostic mappings that contain no correct task alias.

Each task receives a frozen balanced permutation from the 12 one-token aliases
to the 12 operation names. The visible prompt publishes the mapping. Thus `cat`
does not always mean `reverse`, and token identity cannot substitute for
prompt-local operation semantics.

## 4. Shared native prefix

Each task produces one sampled native thought prefix of exactly 512 tokens with
no close/EOS. Fail the row rather than substitute a prefix. Prompt, decoder,
seed, and cap are fixed. Candidate identity, diagnostic labels, and task gold
cannot affect prefix generation.

Prefix sampling uses the model's native cached generation path and does not
suppress `</think>`; an early natural close fails. Every subsequent anchor
capture, patch, control, and mechanics readout is batch-one, padding-free,
cache-free full recomputation over those exact locked prefix token IDs.

This is scaffolded native thought: the prefix is generated naturally, while the
hypothesis interface is explicitly inserted. It is not evidence for a
spontaneous global workspace or consciousness interpretation.

## 5. Context-local anchor

Append `\n\nCandidate first-operation alias:` inside the still-open think section.
A task-ID rule selects one valid source alias independently of gold. Every donor
context differs only in the one leading-space alias token at the same absolute
position and sequence length.

At each frozen layer, capture each clean donor activation and all-24 donor
coordinates in the exact task/prefix context. The primary arm keeps the source
token visible but replaces its all-24 coordinates with the clean target-donor
values. Desired values always come from the clean donor trajectory for that
layer, never an already patched state. Every non-identity source-to-target donor
is tested; no correct candidate is selected.

## 6. Two label-free mechanics probes

Mechanics never loads `first_op`, target pipeline, hidden examples, or a correct
alias.

The direct probe ends reasoning and scores the supplied target among the 12
alias tokens. It contains the same public result-label table as the consequence
probe plus an explicit identity-control instruction, making both suffixes exactly
216 tokens under the pinned tokenizer. This preserves the frozen `1e-3` causal-
invariance test across the hybrid sequence-shape implementation. Direct identity
diagnoses writing but cannot authorize continuation.

The primary consequence probe uses diagnostic input `[3,-1,2,0]`, with `k=2`
for parameterized operations. The 12 distinct one-step results are:

| operation | result |
| --- | --- |
| reverse | `[0,2,-1,3]` |
| sort ascending | `[-1,0,2,3]` |
| sort descending | `[3,2,0,-1]` |
| absolute value | `[3,1,2,0]` |
| square | `[9,1,4,0]` |
| negate | `[-3,1,-2,0]` |
| running sum | `[3,2,4,4]` |
| adjacent difference | `[-4,3,-2]` |
| add 2 | `[5,1,4,2]` |
| multiply by 2 | `[6,-2,4,0]` |
| take 2 | `[3,-1]` |
| rotate by 2 | `[2,0,3,-1]` |

For each task, seed `2026072409` rotates a one-to-one mapping from these result
lists to the 12 remaining one-token lens concepts. The mapping appears after
the anchor, followed by a fixed query asking for the matching result label.
Success therefore requires
`alias -> prompt-local operation -> computed result -> task-randomized label`;
raising the supplied alias logit alone cannot pass.

## 7. Mechanics arms

For each task and each of its 11 non-source donors:

1. unmodified source anchor;
2. literal target text;
3. full target-donor activation at layers 4--8;
4. all-24 target-donor J coordinates;
5. mean donor J coordinates;
6. frozen alpha-one centered additive J from the failed native experiment;
7. exact post-bf16 non-J A;
8. independent exact post-bf16 non-J B;
9. a fixed derangement's wrong-donor J coordinates; and
10. ordinary all-24 concept-logit-lens replacement.

Full donor is an upper bound, not J evidence. Additive J is an alternative
mechanism, not a negative control. Literal text is the cheaper deployable arm.

## 8. Numeric and implementation controls

- One hook application at the exact one-token anchor per frozen layer.
- Equal source/donor position and sequence length; one-token contracts for all
  aliases and result labels.
- Batch one, no padding, `use_cache=False` for mechanics.
- Anchor activation is invariant to causal suffix choice within `1e-3` maximum
  absolute difference.
- Two independent non-J controls per task/donor/probe/layer.
- Every realized non-J delta has <=`1e-5` relative norm error versus its paired
  realized donor-J delta and <=`0.01` full-J-span projection fraction.
- Finite logits/probabilities, nonzero interventions, exact row cardinality,
  parse >=0.95, and complete alias/label breadth receipts.
- Numeric repair uses only geometry. No output label, result gradient, task gold,
  or model outcome can construct, select, scale, or repair an intervention.

An outcome-blind live-bf16 receipt must make 100% of numeric rows valid before
mechanics logits may be retained.

## 9. Frozen mechanics decisions

Across 44 supplied non-identity targets:

- literal text: direct >=90%, consequence >=80%;
- full donor: direct >=75%, consequence >=65%;
- donor J: direct >=60%, consequence >=50%;
- mean donor-J-arm consequence target-probability lift over source >=+0.15;
- donor-J consequence selection minus the worse non-J arm >=35pp;
- wrong donor: own consequence >=50%, registered target <=15%;
- donor-J consequence successes span >=10 target aliases and >=10 randomized
  labels, with nonzero success on all four tasks; and
- every instrumentation/numeric row passes.

Terminal interpretation is fail-specific:

- `ANCHOR_PROBE_UNREACHABLE`: literal text fails;
- `NO_NATIVE_ANCHOR_STATE_TRANSPORT`: text passes, full donor fails;
- `NO_NATIVE_ANCHOR_J_TRANSPORT`: full donor passes, J direct fails;
- `DIRECT_ONLY_NATIVE_ANCHOR_J`: direct J passes, consequence J fails;
- `NATIVE_ANCHOR_J_CONSEQUENCE_TRANSPORT`: the complete J mechanics passes.

Separately record `ADDITIVE_ANCHOR_TRANSPORT` if additive J clears the same
direct/consequence semantic gates. That weakens donor-replacement specificity
and authorizes an additive comparator; it cannot be relabeled a control failure.
No unregistered token, layer, amplitude, prompt, or soft-only gate may rescue a
failed arm. Only `NATIVE_ANCHOR_J_CONSEQUENCE_TRANSPORT` opens continuation in
this experiment; a text-only follow-up must be a separate experiment.

## 10. Conditional capability stage

Each authorized method starts from the same 512-token prefix and fixed source
anchor, instantiates all 12 candidate targets without gold, appends an instruction
to investigate and revise the hypothesis, continues to the fixed 1,024-thought-
token boundary, and uses the identical `First:` output interface.

The primary controller is argmax mean constrained alias probability. Majority,
maximum confidence, and minimum entropy are frozen diagnostics and none may be
silently omitted when stronger. Correctness and oracle coverage are loaded only
after all outputs, selector predictions, and resource receipts are written.

Comparators include literal text, full donor, every authorized internal
mechanism/control, clean shared-prefix continuations, and a frozen 12-trace
fully independent sample-more master pool. Resource matching selects sample-
more prefixes using only sampled generated-token cost and total logical model-
token cost; full K=12 is a conservative overmatch. Prompt prefill, shared native
prefix, every donor capture, source recipient, forced scaffold, continuation,
commit, and cache/full-parity audit count. All arms remain Transformers.

## 11. Qualification and confirmation

On 24 qualification tasks, an internal arm must:

- score between 0.15 and 0.75 accuracy;
- beat every deployable/control selector by >=10pp with one-sided paired-task
  95% bootstrap lower bound >0;
- beat literal text by >=5pp unless a preregistered real compute advantage is
  established;
- improve oracle answer coverage by >=5pp versus its strongest matched pool;
- predict at least eight aliases and succeed on at least six; and
- pass every backend, resource, cache, numeric, parsing, and diversity contract.

Only a complete pass opens 48 untouched confirmation tasks with identical
rules. Splits never pool. A capability result requires the same gates on
confirmation and must beat matched-compute sample-more.

## 12. Claim boundary

Direct identity is token writing. Randomized computed-consequence mechanics is
a writable semantic-state diagnostic, not capability. A confirmation win would
be test-time elicitation, not weight installation. If visible text wins while J
does not beat it, the unlocked method is explicit hypothesis enumeration, not
J-space. No result may be called capability gain without the matched-compute
held-out endpoint.

# Adversarial Design Review: Materialized Residual Sibling Search

**Date:** 2026-07-13

**Model calls before or during review:** none

**Initial verdict:** reject before implementation

**Final rewritten-design verdict:** pass for model-free design/data lock;
mechanics implementation remains absent and requires its own audit and published
lock

## Review process

Three independent read-only reviews covered prior art and construction,
statistical/interface validity, and system/resource confounds. They reviewed
the first draft, rejected it before any model process existed, then re-read the
replacement and its executable CPU receipt. A final mechanics/result adversary
identified one remaining governance coupling in the replacement; that coupling
was removed before this lock.

No reviewer edited files, no benchmark content was opened, and no Qwen3.5-4B
model was loaded or called.

## First-pass fatal findings and resolutions

### Impossible unique-first balance

The first draft required every operation to be the unique first operation of
exact-depth-three tasks. This is impossible for `negate` and `take_k(1)` under
the registered DSL. Exhaustive common-panel enumeration independently found
zero unique-first signatures for both.

**Resolution:** unique-first labels were deleted. Tasks use one to four
multi-label public-live siblings, where liveness is exactly the existence of a
visible-fitting two-operation suffix. All 24 operations remain symmetric
candidates.

### Vacuous task fingerprints and insufficient capacity

Hashes over task-specific random inputs would not prevent the same underlying
function from recurring across splits, and the proposed unique-first quotas did
not have enough functions.

**Resolution:** exact minimum depth and disjointness are defined on a frozen
80-input common panel. Function signatures, concrete triples, and registered
suffixes are globally disjoint. Every 24-task block has exactly 8/8/4/4 tasks
with one/two/three/four public-live siblings and balanced A/B viability
orientation. The executable enumeration found 3,525 eligible exact-depth-three
function fingerprints and filled all 264 tasks.

### Tautological hidden selection

The first draft admitted tasks only when visible-consistent programs agreed on
hidden and probe outputs. That would make coverage and selected accuracy
nearly tautological.

**Resolution:** hidden competing-candidate outcomes and probe agreement never
participate in admission. Hidden and probe inputs are generated from independent
domain-separated streams and redrawn only for duplicate inputs or invalid target
trajectories. The visible/probe selector is therefore fallible.

### Ranking dominated by unguided completion

Twenty-four 512-token ranking traces plus four suffix traces would cost more
sampled tokens than completing every sibling.

**Resolution:** all-24 materialized suffix completion is the primary explorer.
Ranking is a one-position, no-think raw-logprob secondary. If it passes, its
top-four policy receives a real independent four-request suffix run; it never
reuses all-24 outputs. Exact taskwise sampled- and logical-token costs decide
whether the secondary is operationally cheaper.

### Missing suffix and parser ceiling

A ranker could appear useful while the required two-operation suffix ABI was
unusable.

**Resolution:** mechanics A evaluates every public-live sibling, not a selected
one, under materialized, name-only, deranged, and supplied-suffix echo prompts.
It also runs a separate direct three-operation ABI ceiling. Every condition has
its own parse and cap-contact gate, while task-level any-live success matches the
later all-24 policy.

### Invalid partial-operation semantics

Empty lists, rotate-on-empty, and overflow could crash or diverge between
construction and scoring.

**Resolution:** one typed `INVALID` contract is shared by construction,
enumeration, prompts, assembly, selection, and grading. Empty/invalid targets
are rejected and invalid candidates are execution-ineligible.

### Public viability mislabeled as oracle

Exact 24² suffix enumeration can recover live siblings from public examples.

**Resolution:** public viability and full 24³ enumeration are deployable
model-free references. No internal-discovery or CPU-superiority claim is made.

### Underpowered and underspecified inference

The original 48-task multi-endpoint test could not support its claimed
inference, and its bootstrap/multiplicity rule was incomplete.

**Resolution:** qualification is only a point/shard futility gate. Confirmation
uses 192 untouched tasks, four paired selected-accuracy contrasts, exact
one-sided McNemar tests, Holm familywise alpha 0.05, 0.10 point margins, and
separately reported block-stratified marginal bootstrap lower bounds. Frozen
block effects are report-only. A simulation of the entire compound rule passed
in 483/500 trials, or 0.966 with Monte Carlo standard error 0.0081, at the
registered 0.40 versus 0.20 design alternative.

### Aggregate resource matching

Aggregate matching could allow expensive tasks to subsidize cheap tasks.

**Resolution:** direct sampling uses independent conservative first-over match
points on every task for both sampled and logical model tokens. The direct pool
has a frozen order, fixed chunks, and a fail-closed ceiling. Every top-four
cost inequality is likewise taskwise.

### Incorrect training stop logic

Failure of an untrained interface would not rule out supervised residual-policy
installation.

**Resolution:** failure seals only this inference-time interface. Any training
test requires new tasks, a new experiment, and its own preregistration.

## Second-pass blockers and resolutions

### Raw-logprob authentication

The first replacement did not yet prove that all 24 requested alias scores
would be available as raw pre-temperature log probabilities.

**Resolution:** the pinned runner permits exactly 24 requested log probabilities,
sets engine `max_logprobs=24` and `logprobs_mode=raw_logprobs`, and authenticates
the resolved engine mode. Both requested binary IDs or all 24 listwise IDs must
be finite and present exactly once. No allowed-token mask, bias, grammar, or
forced answer is allowed.

### Batch non-invariance and cost ambiguity

Reusing all-24 outputs for a top-four subset would not be a real four-request
policy, and injected close tokens risked double counting.

**Resolution:** top-four suffixes are independently generated after ranks
freeze. For each completion, sampled cost is the actual sampled-token count and
logical cost is stage-one prompt tokens plus stage-two prompt tokens plus
sampled tokens. The stage-two prefill already includes injected close tokens,
so they are not counted again.

### Mechanics and primary-control incompleteness

The replacement initially described one live mechanics sibling and only three
claim-grade contrasts.

**Resolution:** mechanics now covers all 52 live `(task,candidate)` rows in the
24-task mechanics split. Shuffled alignment is one of four Holm-controlled
selected-accuracy comparators alongside name-only and both direct first-over
baselines.

### Secondary veto of the primary

Although ranking was called secondary, the replacement's qualification and
confirmation prose still made top-four success a condition of the all-24
decision.

**Resolution:** primary qualification and confirmation are decided solely by
their all-24 rules. Ranking or top-four failure can seal only the top-four
branch. The top-four qualification and confirmation decisions have separate
names and no causal path into primary authorization or outcome.

### Unsupported top-four claim strength

The secondary had point margins but was described as demonstrating a Pareto
reduction.

**Resolution:** top-four is explicitly a descriptive frozen-task operational
secondary without a significance test or capability/Pareto claim. It reports
whether preregistered coverage margins and strict taskwise cost inequalities
held and nothing stronger.

## Executable model-free receipt

The deterministic smoke constructed 24 mechanics, 48 qualification, and 192
confirmation tasks with 264 unique common-panel function fingerprints. It
found 354 shallow fingerprints and 3,525 eligible exact-depth-three signatures,
independently re-enumerated public-live sets on all mechanics tasks plus one
task per later block (34 total), exercised strict parsing/assembly/selection,
and verified taskwise resource formulas and key discrete threshold geometry.

The pinned tokenizer audit verifies distinct single-token plain aliases A-X and
single-token leading-space forms. Its complete pass over every frozen task, all
24 candidates, and every suffix/ranking/direct family finds exact rendered
prompt lengths from 259 through 941 tokens; the short minimum is the supplied
echo ceiling, and every family remains within its
correct condition-specific 4,096-token reserve. The final receipt binds the frozen
configuration, design documents, source files, and prompt/token-ID hashes. It
records `benchmarks_read=false`, `model_loaded=false`, and `model_calls=0`.

## Final scope and authorization

- Wrong-task evidence and late materialization remain outside scope.
- CPU enumeration remains exact and practical at depth three.
- No J-space, internal certainty, autonomous discovery, training, or installed
  capability claim is licensed by this experiment.
- This review authorizes only the published model-free design/data lock.
- Before constructing Qwen3.5-4B, mechanics code must be implemented, tested,
  adversarially audited against this lock, committed, pushed, and separately
  authorized.

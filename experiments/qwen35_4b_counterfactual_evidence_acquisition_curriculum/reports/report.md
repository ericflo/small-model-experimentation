# Counterfactual evidence-acquisition curriculum report

## Status

**Finished — `LINEAGE_LOCALITY_INFEASIBLE` (2026-07-13).** The first
model-bearing gate stopped the experiment before interface sampling, behavioral
qualification, or training. The chosen transaction-replay start checkpoint was
not sufficiently local to the frozen apex anchor for the registered causal
comparison. No Menagerie event occurred.

## Research-program fit

The experiment is a capability-production successor in
`agentic_breadth_installation`, with supporting roles for Active Evidence
Acquisition, Process Control and Tool Use, Posttraining and Adaptation, and
Benchmark Generalization.

The direct predecessor found that post-failure semantic recovery was generally
already strong once visible verifier evidence was in context. Its opened
rejected-state trajectories instead failed before proposal: 0/54 inferred cases
made a correct first patch and 0/72 inspected visible tests before patching.
Because that predecessor formally failed its answer-cap instrument gate, this
successor does not inherit the observation as evidence. It independently
qualifies both acquisition headroom and evidence-utilization reachability before
training.

The newly landed `qwen35_4b_early_text_hypothesis_forking` is an explicit
near-neighbor but not the same intervention. It supplies a complete bank of
first-step hypotheses before thinking on a small DSL and performs no training
or tool acquisition. The present design tests autonomous repository search and
weight-installed evidence-to-patch binding. It makes no broad novelty claim
about early textual proposal shaping.

## Frozen method

### Counterfactual causal unit

Each inferred pair holds the issue, source, tree, paths, and all
non-discriminator files byte-identical. One public evidence file flips between
opposed policies. The primary event requires the agent to acquire that evidence
before its first changed patch, pass both public and host-private checks on its
own branch, and cross-fail the paired branch. Both branches must succeed, so the
dyad is the statistical unit.

Evidence crosses test, documentation, and callsite channels. The bank,
qualification, and transfer splits have disjoint path skins. Transfer uses two
unseen families and a signature-based search query absent from the reference-
and symbol-query training/qualification regimes.

### Qualification before production

An outcome-free interface ladder chooses the smallest answer allowance among
1,024, 2,048, and 4,096 that passes invalid-action and all-length-contact gates.
Then two independent start-model blocks compare unassisted acquisition with a
host-injected correct search, a matched-operator nondiscriminating search, and
an explicit-contract control. The control's output must exclude the designated
evidence path and marker. Training is authorized only if both blocks show low
unassisted success, high injected reachability, a 30-point injected advantage
over both unassisted and nondiscriminating search, broad family/channel/query
support, and healthy interface behavior.

### Transition-balanced training

The fixed primary `evidence_binding` arm mixes 24 inferred counterfactual tasks
with 24 prior complete-loop task blocks. `explicit_redundant` provides equal
dose when the issue already states the answer. `shuffled_binding` preserves the
primary prompts and target multiset but exchanges evidence-to-patch targets
within each pair.

All three arms contain 432 rows: 48 at each of nine conditional transitions,
including ambiguous-source acquisition, evidence-to-policy patching,
rejected-patch revision, failed-test diagnosis and changed revision,
verification, and commit. Every transition receives exactly 16,000 weighted
answer tokens per epoch. Think loss is zero, physical model forwards are
unpadded batch one, and optimizer steps consume full
nine-transition supercycles.

### Transfer and sampling burden

The primary must pass direct apex-relative locality, trained-family calibration,
held-out development, untouched confirmation, explicit conditionality, and two
old-family loop-retention blocks. It must beat the exact start checkpoint, C54
apex incumbent, both trained controls, and the stronger matched sample-more
baseline.

Sample-more pools contain six shallow trajectories for start and apex. A fixed
trajectory-order prefix is selected using only actual sampled and logical model
token costs, at the first point meeting both costs of the primary deep case.
Outcomes do not determine prefix membership. The full pools are reported only
as favorable oracle coverage.

## Key registered gates

- Start and primary each must clear direct apex locality on the exact new
  48-context set: centered non-target drift at most 0.10 and entropy delta at
  least −0.05.
- Trained-family primary paired success must be at least 0.65, +0.15 over start,
  and +0.10 over each trained control.
- Both held-out blocks require +0.08 over start and apex, +0.05 over each
  control, matched-operator nondiscriminating search, and the stronger
  dual-overmatched sample-more prefix; they also require injected reachability,
  nonnegative transfer on both families, per-channel and held-out-query support,
  and a nonnegative paired bootstrap lower bound versus start.
- Rejected and failed transition retention must each reach 0.95; verification
  and commit must each reach 0.90.
- On both legacy suites, normal and recovery success may regress at most 0.03
  versus start, and no family may regress more than 0.10.
- Invalid and unusable-cap deltas may not exceed +0.02 after training.
- Menagerie remains sealed until every white-box gate passes. Its final
  strategy gate requires at least one tier at +0.02 and neither below −0.03
  versus apex.

All thresholds and stop labels are specified in
[`preregistration.md`](preregistration.md). Controls cannot replace the fixed
primary, development and confirmation cannot be pooled, and a stopped stage
cannot be repaired in this directory.

## Controls and causal interpretation

The injected-evidence condition proves only that the current interface can use
the public discriminator; it is not deployable. Matched-operator
nondiscriminating search separates specific evidence from a generic search
action and extra tool-result context. Explicit redundant acquisition tests
generic extra task/search dose. Within-dyad label shuffling tests whether the
observed evidence-to-patch direction matters while preserving frequency and
prompt structure. Start, apex, matched sampling, locality, and legacy retention
protect against nonspecific training damage.

The strongest permitted white-box conclusion is that this action-only recipe
installed an ambiguity-triggered search-and-binding policy that transferred
across the registered procedural family/path/query shifts. It would not prove
open-world search, internal confidence, verifier-free correctness, or general
coding superiority. A benchmark pass would add cross-instrument evidence, not
authorize those stronger claims.

## Firewall and artifact policy

Only `Qwen/Qwen3.5-4B` is allowed. Hidden tests and patches remain host-side.
No benchmark source, item, transcript, or result detail may be read. White-box
behavior uses one pinned vLLM backend; Transformers is limited to symmetric
logit audits and training internals.

The three deterministic banks live under
`large_artifacts/qwen35_4b_counterfactual_evidence_acquisition_curriculum` and
are bound by compact checksums. The compact locality summary is tracked in
`analysis/locality_start_vs_anchor.json`; row-level logits were not retained.
No adapter, merged candidate, behavior trajectory, sample pool, uncertainty
table, or benchmark artifact was created. The exact inventory is in
[`artifact_manifest.yaml`](artifact_manifest.yaml).

## Result

The deterministic model-free smoke and exact prompt-token equivalence checks
passed. The immutable 35-file design was locked to commit
`7311bbeeef2bffe72024eae5b4136c07bbaa7704`, and the locality gate evaluated
the registered 48 contexts symmetrically on the exact frozen checkpoints.

| Registered quantity | Observed | Gate | Result |
| --- | ---: | ---: | --- |
| Median centered non-target logit drift | 0.110735 | at most 0.100000 | fail |
| Mean entropy delta, start minus anchor direction | +0.013636 | at least -0.050000 | pass |
| Mean varentropy delta | +0.000297 | diagnostic only | neutral |
| Contexts | 48 | exactly 48 | pass |

The drift failure triggered the preregistered terminal verdict
`LINEAGE_LOCALITY_INFEASIBLE`. The answer-band ladder, qualification blocks,
all three training arms, candidate locality, calibration, transfer, retention,
uncertainty analysis, and Menagerie were sealed immediately. Zero benchmark
seeds were consumed.

## Interpretation

This result does not test evidence-acquisition behavior or the proposed
curriculum. It establishes only that the registered start/anchor pair cannot
support the intended attribution under the frozen 0.10 locality contract. The
entropy guard rules out the specific failure mode of a broad entropy collapse;
it does not rescue the failed drift gate, and varentropy remains diagnostic.

No new shared claim is warranted because the comparison is specific to this
checkpoint pair and context distribution. The durable operational lesson is
to measure exact parent-to-anchor locality before investing in a downstream
causal curriculum. A successor would need a new intake and preregistration
using either a genuinely local reference pair or a design that treats their
pre-existing difference as an explicit baseline. The threshold cannot be
widened or the run rescued in this result-bearing directory.

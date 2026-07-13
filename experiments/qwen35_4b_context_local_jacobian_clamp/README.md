# Qwen3.5-4B Context-Local Jacobian Clamp

**Status:** finished

This is the mechanism-correcting follow-up to
`qwen35_4b_jacobian_value_transport`. It asks whether a concept edit at the
earlier token that represents the selected key can change a separately computed
consequence. The intervention is a fixed set-to-donor clamp, not a repeated
pairwise swap at the answer position.

## Research Program

- Primary: `interpretability_and_diagnostics`
- Conditional secondary programs: `structured_execution_and_compilers` and
  `test_time_reasoning_budget` only if the consequence gate passes.
- Closest near-duplicate: `qwen35_4b_jacobian_value_transport`, whose averaged
  last-position coordinate changed 18/24 direct concept reports but 0/24 mapped
  consequences.
- Prior anchors: C19/C20/C30 (decodable, inert under ActAdd, usable when
  externalized), C51/C52 (actionability and intervention-locality firewalls).

## Question

When a prompt explicitly names a selected key and later asks for a fresh
key-to-digit lookup, can a token-Jacobian coordinate clamp at that selected-key
position make Qwen3.5-4B compute the counterfactual key's digit? Or do these
directions remain token-output controls even when applied at a causal antecedent?

## Hypothesis

The first experiment patched the final prediction position, where a late
token-aligned direction can control the imminent word without representing a
state that downstream computation reads. Here, each direction is fitted as the
pullback from a future direct concept report to the earlier selected-key token.
At evaluation time, the source prompt is clamped toward clean activations from
an otherwise identical target-key donor prompt across a fixed layer band.

If the coordinate is a reusable concept state, the mapped digit should change
even though no digit direction or answer gradient enters the intervention. If
only the direct key report changes, local writability still does not imply
semantic transport.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16 Transformers inference,
  cache disabled.
- Substrate: fresh prompt-local tables mapping eight one-token concepts to eight
  distinct one-token digits. Every source, target, and wrong-donor digit differs.
- Splits: 48 separate lens-fit prompts, 24 band-selection items, and 48 untouched
  confirmation items. No benchmark content is read or used.
- Causal site: the final occurrence of the concept token on the `Selected key:`
  line, before either the direct-report or mapped-consequence suffix.
- Primary intervention: replace all 24 targeted J coordinates at that site with
  the clean target-donor coordinates at every layer in the selected five-layer
  band, with alpha fixed at 1.
- Band selection: full-activation donor patch only, on the selection split;
  J-clamp results cannot select the band.
- Primary controls: untouched source, full donor activation, exact per-item and
  per-layer norm-matched random vectors orthogonal to the J dictionary, wrong
  donor, two-coordinate clamp, and logit-lens coordinate clamp.
- Primary metric: target-digit rate on the untouched mapped-consequence split.
  Direct target-key rate is a required mechanism check, not the endpoint.
- Oracle boundary: target donors, target identities, and all causal patches are
  oracle-only mechanism evidence. They cannot establish a deployable capability.
- Forbidden leakage: the target digit and its output-margin gradient cannot
  construct, scale, select, or gate the primary J intervention.

Frozen thresholds and the decision tree are in
[`reports/preregistration.md`](reports/preregistration.md). The pre-run
adversarial review is in [`reports/design_review.md`](reports/design_review.md).

## Run

CPU smoke and unit tests:

```bash
.venv/bin/python -m pytest experiments/qwen35_4b_context_local_jacobian_clamp/tests -q
.venv/bin/python experiments/qwen35_4b_context_local_jacobian_clamp/scripts/run.py --stage smoke
```

Scientific stages are restartable and refuse to cross a failed gate:

```bash
.venv/bin/python experiments/qwen35_4b_context_local_jacobian_clamp/scripts/run.py --stage model-smoke
.venv/bin/python experiments/qwen35_4b_context_local_jacobian_clamp/scripts/run.py --stage fit-lens
.venv/bin/python experiments/qwen35_4b_context_local_jacobian_clamp/scripts/run.py --stage donor-gate
.venv/bin/python experiments/qwen35_4b_context_local_jacobian_clamp/scripts/run.py --stage confirmation
```

## Results

Frozen terminal verdict: **`INVALID_CONTROL`**. The scientific signal was
otherwise perfect, but one of 96 random-control rows missed the preregistered
per-item/layer realized-norm tolerance, so this experiment cannot claim
`J_TRANSPORT`.

- The full-activation donor gate selected the earliest band 4–8. Target donors
  changed both direct key and mapped digit on 24/24 selection items; wrong donors
  changed 24/24 to their own key/digit. Four early bands passed; two late bands
  failed.
- On 48 untouched confirmation items, the all-24 J clamp changed both the direct
  key and mapped digit on 48/48. The two-coordinate J clamp changed the mapped
  digit on 47/48.
- Wrong-donor J clamps changed 48/48 to the wrong donor's own key/digit, while
  concept logit-lens clamps and span-orthogonal random controls changed 0/48.
- The paired J-minus-random consequence difference was +1.00 with bootstrap
  95% interval [1.00, 1.00]. Parsing remained 100%.
- Formal failure: one consequence random-control row had realized-norm relative
  error 1.155e-5 versus the frozen 1e-5 maximum. Requested controls were
  orthogonal to numerical precision, but bf16 application left as much as 5.7%
  of one realized random delta in the J span. Neither defect is rounded away.

See [`reports/report.md`](reports/report.md) and
[`analysis/summary.md`](analysis/summary.md).

## Interpretation

The parent experiment's late answer-position coordinate looked motor-like. The
new early selected-token clamp instead exhibits the registered signature of a
semantic state: it transports through a fresh arbitrary mapping, a wrong donor
produces its own consequence, a logit lens is inert, and a two-coordinate clamp
nearly saturates. This is high-priority evidence, not a valid positive claim,
because the exact random-control contract failed. A separate fresh replication
must enforce both realized norm and realized span orthogonality before any
native-thinking continuation.

## Knowledgebase Update

- Program evidence: update with the invalid-but-high-priority result after it is
  synchronized to `origin/main`.
- Program backlog: replicate with a quantization-aware realized-orthogonal
  control on fresh mappings before native-thinking work.
- Claim ledger: remain unclaimed while the repository-wide re-grade is open.

## Artifacts

- Small data, metrics, row-level outputs, fitted targeted lens, and receipts are
  committed when generated.
- No training or adapter is part of this experiment.
- See `reports/artifact_manifest.yaml`.

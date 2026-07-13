# Counterfactual evidence-acquisition curriculum

**Status:** finished · `LINEAGE_LOCALITY_INFEASIBLE` · 2026-07-13.

This experiment tests whether transition-balanced action supervision can teach
Qwen3.5-4B to search decisive public repository evidence *before* its first
patch, bind that evidence to the correct policy, and retain the complete coding
loop.

## Research programs

- Primary: `agentic_breadth_installation`.
- Supporting: `active_evidence_acquisition`, `process_control_and_tool_use`,
  `posttraining_and_adaptation`, and `benchmark_generalization`.
- Direct predecessor: `qwen35_4b_semantic_policy_headroom_tournament`.
- Training parent: the exact transaction-replay checkpoint from
  `qwen35_4b_transaction_invariant_recovery_curriculum`.
- Closest newly landed conceptual near-neighbor:
  `qwen35_4b_early_text_hypothesis_forking`.

The near-neighbor supplies exhaustive first-operation hypotheses before
thinking on a depth-two list DSL and performs visible-only selection. It does
not train weights or acquire evidence with tools. This experiment therefore
does not claim novelty for generic early hypothesis shaping; it asks whether an
autonomous repository evidence-search policy can be installed and transferred.

## Question

Can one Qwen3.5-4B checkpoint learn the conditional sequence

```text
ambiguous source -> search discriminator -> evidence-faithful first patch
```

across unseen coding families, evidence paths, and query wording while
preserving `rejected_patch -> changed_patch`,
`failed_test -> diagnose/revise`, `patch_ok -> verify`, and
`passed_test -> commit`?

## Result

The experiment stopped at its first model-bearing gate, before interface
sampling or training. On the frozen 48-context block, the transaction-replay
start checkpoint had median centered non-target logit drift `0.110735` from
the apex anchor, above the preregistered `0.100000` ceiling. Entropy retention
passed: mean entropy changed by `+0.013636` against a `-0.050000` floor, and
mean varentropy changed by `+0.000297`.

The terminal verdict is `LINEAGE_LOCALITY_INFEASIBLE`. The answer-band ladder,
qualification, all three training arms, transfer, retention, uncertainty
diagnostics, and Menagerie remained sealed. This does not test whether the
evidence-acquisition curriculum works; it shows that the chosen start/anchor
pair was not local enough for the frozen causal comparison. Any repair requires
a new intake and experiment rather than a wider post-result threshold here.

## Why this question is next

The predecessor found no usable post-failure semantic axis and formally stopped
on its answer-cap gate. Its opened trajectories nevertheless localized the
remaining proposal problem: every failed-test case eventually reached a
correct patch, but inferred rejected states produced 0/54 correct first patches
and none of 72 trajectories inspected visible tests before first patching. The
model appeared able to use decisive evidence once supplied, but did not seek it
before committing.

That contrast is only a hypothesis generator. This experiment first requires
two clean blocks to reproduce low unassisted acquisition, high evidence-
injected reachability, and a large advantage over matched-operator
nondiscriminating search before any training is authorized.

## Counterfactual repositories

Each inferred dyad has byte-identical issue text, source, file tree, path names,
and all non-discriminating public bytes. Exactly one public evidence file
differs and requires the opposite semantic patch. A successful dyad requires
both branches to:

- acquire the designated evidence before their first changed patch;
- make a first patch that passes that branch's visible and hidden checks; and
- produce a patch that fails on the counterfactual counterpart.

The model never sees hidden executables, hidden results, branch labels, or
oracle patches. Evidence spans public tests, documentation, and callsites.
Bank, qualification, and transfer use disjoint path-name regimes; transfer also
uses a held-out signature-query skin.

## Training design

The fixed primary arm, `evidence_binding`, mixes 24 inferred counterfactual
tasks with 24 complete blocks from the existing transaction/recovery replay.
Two matched controls use explicit-contract redundant acquisition or exchange
the evidence-to-patch labels only within each dyad.

Every arm has 432 rows: 48 rows for each of nine conditional transitions. Each
transition receives exactly 16,000 weighted answer-action tokens per epoch.
Think loss is zero. Training uses rank 32, alpha 64, LR 2e-5, three epochs, and
complete nine-transition optimizer supercycles. Logical batch-four rows are
executed as serial unpadded physical batch-of-one forwards to avoid the known
hybrid-architecture padding divergence.

The controls distinguish aligned evidence binding from generic extra training,
unconditional search, or nonspecific shuffled-label damage. The primary is
fixed before controls run and cannot be replaced after outcomes.

## Staged gates

1. Model-free task, firewall, bank, batching, compute-accounting, and context-
   geometry smoke.
2. Exact start-to-apex locality feasibility on the new 48-context block, before
   any behavioral exposure or training.
3. Outcome-free interface selection over answer rungs 1,024/2,048/4,096.
4. Two independent start-model qualification blocks: unassisted, correctly
   injected evidence, matched-operator nondiscriminating search, and
   explicit-contract control.
5. Equal-mass three-arm training, merge, and direct candidate-to-apex locality.
6. Trained-family calibration against start and both controls.
7. Held-out-family development and untouched confirmation against start, apex,
   both controls, nondiscriminating search, injected reachability, and the
   stronger of start/apex actual-compute-matched sample-more pools.
8. Explicit conditionality plus broad-recovery and transaction-loop retention.
9. Entropy/varentropy diagnostics only after outcomes are already open.
10. Fresh paired Menagerie quick/medium CLI events only after every white-box
    gate passes.

The actual sample-more comparator is selected by a fixed trajectory-index
prefix that overmatches each primary case on both sampled and logical model
tokens. Outcomes never choose prefix membership; full six-trajectory pools are
oracle-only.

Exact thresholds, stop labels, and prohibited rescues are frozen in
[`reports/preregistration.md`](reports/preregistration.md). The adversarial
attacks that shaped the design are in
[`reports/design_review.md`](reports/design_review.md).

## Firewall and interpretation boundary

Only `Qwen/Qwen3.5-4B` is permitted. Every behavior arm uses the same pinned
vLLM runner. Fresh procedural repositories are the only training and white-box
evaluation substrate. Nothing under `benchmarks/` may be read or imported, and
Menagerie may be invoked only through its public CLI after authorization.

A white-box pass would support a narrow learned policy for ambiguity-triggered
search and evidence-faithful proposal on these procedural coding families. It
would not establish open-world active search, verifier-free correctness,
internal uncertainty measurement, or a universal capability unlock. A
Menagerie pass would add cross-instrument evidence without erasing those
limits.

## Frozen run protocol

The commands below record the preregistered protocol. The terminal disposition
now blocks every scientific command in this directory; they are retained for
auditability, not as pending work.

Model-free smoke:

```bash
.venv/bin/python experiments/qwen35_4b_counterfactual_evidence_acquisition_curriculum/scripts/run.py --smoke
```

After committing and pushing the immutable design, create the lock receipt:

```bash
.venv/bin/python experiments/qwen35_4b_counterfactual_evidence_acquisition_curriculum/scripts/run.py --lock-design <commit>
```

Commit and push that receipt to `main` before any Qwen output. The stages can
then be run separately or resumed end to end:

```bash
.venv/bin/python experiments/qwen35_4b_counterfactual_evidence_acquisition_curriculum/scripts/run.py --interface
.venv/bin/python experiments/qwen35_4b_counterfactual_evidence_acquisition_curriculum/scripts/run.py --qualify
.venv/bin/python experiments/qwen35_4b_counterfactual_evidence_acquisition_curriculum/scripts/run.py --full
```

The full orchestrator must stop at every failed ancestor gate. Its Menagerie
stage remains sealed unless it has just assembled the final white-box
authorization receipt.

Every scientific stop writes `runs/terminal_disposition.json` with
`lifecycle_closed: false`. After the result is documented, checked, committed,
and pushed directly to `main`, seal the documentation commit with:

```bash
.venv/bin/python experiments/qwen35_4b_counterfactual_evidence_acquisition_curriculum/scripts/run.py --closeout
```

Regenerate the catalog to account for the changed receipt, run `make check`,
commit and push the closed receipt and generated fixpoint, then prove that exact
clean commit is on `origin/main`:

```bash
make catalog && make catalog && make check
.venv/bin/python experiments/qwen35_4b_counterfactual_evidence_acquisition_curriculum/scripts/run.py --verify-closeout
```

The semantic closeout rejects a lingering in-progress registry entry,
design-only brief/headline chart, missing program evidence, or stale generated
indexes. The experiment is not finished until the post-push verification
succeeds and CI is green.

## Final state

The deterministic model-free smoke passed, the immutable 35-file design was
locked to commit `7311bbeeef2bffe72024eae5b4136c07bbaa7704`, and the exact
lineage-locality gate then failed on drift while passing entropy retention.
Only symmetric next-token logit measurements were produced. No behavioral
trajectory, adapter, trained checkpoint, transfer score, or benchmark event
exists for this experiment, and zero Menagerie seeds were consumed.

## Artifacts

- `idea_intake.md`: program routing, prior work, and non-duplication decision.
- `reports/preregistration.md`: immutable arms, thresholds, stage order, and
  stop taxonomy.
- `reports/design_review.md`: adversarial objections and mandatory controls.
- `configs/default.yaml`: exact lineage, seeds, budgets, and gates.
- `reports/context_geometry_receipt.json`: model-free context-fit check.
- `reports/smoke_receipt.json`: deterministic bank, tokenizer, batching,
  firewall, and counterfactual-invariant smoke.
- `analysis/locality_start_vs_anchor.json`: terminal symmetric logit-locality
  result.
- `runs/terminal_disposition.json`: open/closed lifecycle record for
  `LINEAGE_LOCALITY_INFEASIBLE`.
- `reports/artifact_manifest.yaml`: exact external-bank checksums, tracked
  result artifacts, absent downstream artifacts, and lifecycle commands.

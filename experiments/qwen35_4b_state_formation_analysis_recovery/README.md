# State-Formation Analysis Recovery

**Status:** finished

This experiment recovers the exact frozen v11 state-formation analyzer through one source-bound
path seam, without changing any LoRA/full-rank scientific logic or inspecting a result value.

## Research Program

- Program: `structured_execution_and_compilers`
- Program question: can a fixed latent state interface support exact recurrent execution, and which
  update recipe can form the required deep state?
- Direct producer: `qwen35_4b_state_formation_capacity_adjudication`.
- Prior anchors: `qwen35_4b_state_carry_vs_state_bag`, its
  `qwen35_4b_state_carry_vs_state_bag_fullrank_delta` successor, and
  `end_to_end_structured_slot_executor`.

## Question

Can the exact immutable source-v11 analyzer consume its own source-bound producer receipts after its
registered `../../large_artifacts/...` path was rejected by the analyzer's lexical-canonical guard,
without changing any scientific function, result, threshold, or branch rule?

This is an operational recovery experiment, not a new scientific arm. The producer remains the
result-bearing experiment. Its source cannot be edited: doing so would invalidate every completed
setup, training, and evaluation receipt.

## Hypothesis

The failure is confined to one path-construction boundary. Accepting only the producer's exact
registered external prefix and lexically clean descendants, while delegating every other path to the
original helper, should let the unchanged v11 training graph reopen. Canonical inputs must resolve
identically under the old and recovered helpers. Any broader alias acceptance, source mismatch, or
scientific-code change falsifies the recovery.

## Setup

- Model identity carried from the producer: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; this recovery loads no model.
- Backend identity: the producer's frozen `transformers` backend; this recovery performs CPU-only
  receipt analysis.
- Dataset/task source: no new data. Inputs are only the producer's exact source-v11 receipts and
  procedurally generated evaluation outputs, reopened by their embedded SHA-256 identities.
- Baseline: unmodified v11 `_canonical_expected_path`, which reproducibly rejects its own registered
  external prefix before opening a result row.
- Mechanism-falsifying controls: exact canonical-path equivalence; rejection of unrelated `..`
  aliases; rejection of traversal beneath the registered prefix; automatic restoration of the
  original function; exact producer source/config/implementation pins.
- Primary metric: `RECOVERED_V11_ANALYSIS_COMPLETE` with an output byte hash and original receipt
  identity matching the exact v11 analyzer output.
- Oracle-only metrics: all state-formation values remain the producer's preregistered oracle-side
  measurement. This recovery defines no metric or threshold.
- Hidden-label boundary: smoke and design review open zero result rows, benchmark paths, or sealed
  contrast rows. Scientific analysis runs only after all three LoRA trigger evaluations exist.

## Recovery boundary

The immutable producer computes this registered path:

```text
experiments/qwen35_4b_state_formation_capacity_adjudication/
../../large_artifacts/qwen35_4b_state_formation_capacity_adjudication
```

Its helper rejects any absolute path whose raw spelling differs from `abspath(raw)`, so it rejects
that path before its later repository/no-symlink validation. The recovery seam recognizes only this
exact raw prefix. A descendant is accepted only when every suffix component is clean; it is then
normalized once and passed to v11's unchanged `canonical_repo_path`. All other paths still call the
original helper.

The original analysis output stays at the producer's canonical `analysis/` location because every
later branch authorization requires that exact path. This experiment writes an immutable sidecar
that binds its SHA-256, original receipt identity, frozen recovery smoke, and exact seam. A STARTED
receipt makes the narrow output-write/sidecar-write crash window recoverable without silently
adopting an unrelated result.

## Run

Tests and non-result smoke:

```bash
.venv/bin/python -B -m unittest discover \
  -s experiments/qwen35_4b_state_formation_analysis_recovery/tests -v
.venv/bin/python -B \
  experiments/qwen35_4b_state_formation_analysis_recovery/scripts/run.py --smoke
```

After all three LoRA trigger evaluations are complete:

```bash
.venv/bin/python -B \
  experiments/qwen35_4b_state_formation_analysis_recovery/scripts/run.py \
  --phase lora_joint
```

The same runner exposes only the five preregistered producer phases:
`lora_joint`, `lora_control`, `stage_b_seal`, `fullrank_joint`, and `fullrank_control`.

## Results

The non-result smoke passed at file SHA-256
`02f6f9275f9c30fddb2f49d4b061237e4e11985d92569642d60cd107b05243f7`, receipt identity
`30353be5429d4987509715cfa56a6187f24a80ab353b9b774908071de7ed2f8f`, and recovery source contract
`6ab26016b3de397307c7c8def9c685315b6660370ee98af1a757da11fe1ee94b`. All four path controls pass;
the receipt records zero result rows, benchmark paths, sealed contrast rows, and scientific analysis
calls.

After seed-7413 publication and green workflows, the recovery ran `lora_joint` exactly once. The
unchanged producer v11 analyzer emitted `LORA_JOINT_MISS_CONTROLS_REQUIRED`, next stage
`run_lora_state_only_and_fullrank_joint`, at producer output SHA-256 `cb9fee75…818a` and receipt
identity `b973bc01…a862`. The recovery sidecar identity is `d068482a…f40e`. Zero of 57 required
formation cells passed the 0.40 threshold; the maximum intact required accuracy was 0.0234375, all
three formation categories missed, and adaptation contrast was uncertain. This licenses the
producer's registered Stage B, not any new recovery-defined action. A producer-side metadata
projection had accidentally exposed seed-7412 values before the already-fixed seed 7413 ran
unchanged; no analyzer or choice followed, but terminal interpretation must disclose imperfect
operator blinding.

## Interpretation

The recovery has now demonstrated both a narrow path repair and exact execution of the original
decision tree. Scientific interpretation remains producer-owned: the LoRA joint recipe missed, so
LoRA state-only plus full-rank joint are compulsory before attributing the miss to capacity.

## Knowledgebase Update

- Program evidence: record the producer Stage-A miss and its mandatory rank-causal controls.
- Program backlog: replace the pending adjudication with the exact Stage-B execution branch.
- Claim ledger: no claim is available from an operational repair.
- Reusable artifact: an exact-prefix, source-bound recovery pattern for immutable analyzers.

## Artifacts

- `idea_intake.md`: novelty and non-duplication decision.
- `configs/default.yaml`: exact producer and model/config/source pins.
- `src/recovery.py`: isolated importer, narrow seam, smoke, and analysis wrapper.
- `tests/test_recovery.py`: adversarial scope and restoration tests.
- `reports/design_review.md`: frozen go/no-go review.
- `runs/smoke.json`: source-bound non-result smoke receipt once executed.
- `analysis/<phase>_recovery.json`: result-output sidecars, created only when licensed.
- `reports/artifact_manifest.yaml`: producer-owned external inputs and regeneration path.

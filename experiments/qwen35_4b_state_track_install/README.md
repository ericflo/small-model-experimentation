# State-Track Installation (Stage 9)

Lifecycle 30 — stage 9 of the documented zero-root chain: a DIVERGENT single-kind installation dose of a NEW transferable skill — STATE-TRACKING UNDER DECLARATIVE UPDATES — onto the count_walk composite. ONE fresh rank-32/alpha-64 adapter trains on a fresh 160-row single-kind state-tracking curriculum (`data/sft_state_track.jsonl`, sha `66a8d5be…`) from the count_walk composite parent at fresh seed 87 with the chain's frozen QLoRA recipe, merges through the vendored external merger, must pass a two-arm three-screen pooled_k3 retention non-drift gate, and only a locally promoted candidate may consume the ONE sealed medium seed 78169 under the frozen two-directional INSTALLED_TRANSFER / BOUNDED consequence.

**Status:** in-progress · since 2026-07-17 · model-free construction frozen (design receipt, lineage package, gate instruments, generator, tests, drills all green); awaiting the staged reviews and the GPU stages

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can DESIGNED synthetic curricula install universal, transferable
  agentic skills into the one 4B — provably, on fully documented lineage?
- Prior anchors: lifecycle 27 (`qwen35_4b_count_dont_walk_enumeration` — built the
  count_walk parent, tree `d5fdc55c…`; its enumeration-SKILL dose beat the stage-6
  parent by +0.032 mean aggregate — the divergent-skill precedent this cell repeats),
  lifecycle 29 (`qwen35_4b_count_walk_replay_compound` — replay compounding BOUNDED at
  stage 8: on a replay-saturated parent, another replay dose redistributes rather than
  adds), and lifecycle 22 (`qwen35_4b_zero_root_lineage_rebuild` — the
  contamination-free six-stage chain this cell extends).

## Question

Does a DIVERGENT transferable skill (state-tracking execution) add held-out aggregate
on the replay-saturated count_walk parent, where another dose of replay just BOUNDED?
Frozen two-state consequence; the modal BOUNDED path (the skill installs locally but
does not convert to held-out aggregate at this dose) is a finding about the
install-not-equal-convert boundary, not a failure.

## Hypothesis

Replay is bounded on this parent, but a NON-OVERLAPPING skill can still add: the
chain's only prior divergent-skill dose (count_walk) beat its parent on 4 of 5 sealed
draws (mean +0.032). If a 160-row single-kind state-tracking curriculum installs a
transferable execution skill that lifts held-out families, the candidate lands
INSTALLED_TRANSFER and becomes the program reference artifact. Honest priors
(preregistration): P(aggregate strictly > parent) ≈ 0.4-0.5, but the strict
no-family-below-by->0.1 clause historically binds, so P(INSTALLED_TRANSFER) ≈ 0.30-0.40
and BOUNDED is the modestly likelier verdict.

## Setup

- Model: Qwen/Qwen3.5-4B (revision `851bf6e8…`), always.
- Treatment: `state_track` — fresh r32/a64 QLoRA on `data/sft_state_track.jsonl`
  (160 rows, single kind `u_state_track`, sha-pinned, zero skips enforced, max forward
  775 tokens) via `--model-path` on the count_walk composite; epochs 1, lr 1e-5, bs 1,
  ga 8, maxlen 4096, w_think 0.2, w_close 0.2, seed 87, 20 optimizer steps
  (`scripts/train_trial.py`, fail-closed). The ONLY designed delta is the curriculum.
- The curriculum: `scripts/gen_state_track_curriculum.py` (seed 87) — each row tracks
  3-6 invented named registers through K∈{4..8} declarative updates across four
  surfaces (plain / terse / `X += 3` / narrated), then answers a final-state query.
  EXECUTION of given updates (execute-vs-induce law). Truth-audited by independent
  re-derivation (byte-matched) + answer recomputation; banned-vocabulary audit vs the
  ten benchmark families and the reference inventory; ZERO canonical-user-message
  overlap with the replay pool, the eleven predecessor gate files, and the retention
  screens (unit-tested).
- Parent: `large_artifacts/qwen35_4b_count_dont_walk_enumeration/merged/count_walk`
  (tree `d5fdc55c…`, weights `ddd7bc4b…`), authenticated fail-closed pre-training and
  pre-merge against the IN-CELL sha-pinned provenance copy of lifecycle 27's merge
  receipt (`840edca0…`, `data/provenance/count_walk_merge.json`; the committed sibling
  original is a verification aid) plus the full 9 GB weights hash at BOTH stage
  boundaries.
- Merge: `scripts/merge_trained_arm.py` → `scripts/merge_adapter.py` (`cb9af8b4…`)
  `--base-model` count_walk → `large_artifacts/qwen35_4b_state_track_install/merged/state_track`.
- Local gate: retention-only, TWO arms (parent vs candidate; the new kind is
  deliberately NOT held out locally — transfer is priced by the sealed event), three
  pooled_k3 screens at fresh seeds 88063/88064/88065 (104 rows each, 8 per each of 13
  skills), TWO-SIDED bands on integer screen sums: correct ±15, parsed ±9, cap contacts
  ±9. All three must hold to promote. Freshness audit against sha-pinned in-cell copies
  of the three reference cells' eleven gate files (`data/predecessor_gates/`, seeds
  88052-88062). Write-ahead local ledger.
- Sealed event: medium, tb 1024, fresh seed 78169, three arms in frozen order
  base → count_walk → state_track through the trusted gateway (`53cf6533…`); three
  TODO-pin slots for the candidate (tree/weights/committed merge receipt) fail closed
  while unfilled; `run_benchmark.py` frozen by `check_design.py`'s three-slot NORMALIZED
  hash (`8e2d5420…`), with `train_trial.py` (`9396cff7…`) and `eval_local_vllm.py`
  (`8350c61a…`) pinned symmetrically; one-seed write-ahead ledger with byte-equal crash
  reconciliation.
- FROZEN CONSEQUENCE (no third state): INSTALLED_TRANSFER iff candidate aggregate
  strictly > parent AND no family below parent by more than 0.1
  (`candidate_family >= parent_family - 0.1 - 1e-9`) AND candidate aggregate strictly >
  base; aggregate comparisons carry a 1e-12 tie guard — claim: "a divergent transferable
  skill installs and adds aggregate on the replay-saturated parent; state_track becomes
  the program reference; the divergent-skill move class is not bounded where replay is."
  BOUNDED otherwise — claim: "the divergent-skill dose does not add aggregate at this
  dose on this parent; count_walk remains the reference; the install-not-equal-convert
  law extends to this skill." Goal gate vs base recorded descriptively.
- Hidden-label boundary: only `scripts/run_benchmark_aggregate.py` runs; `benchmarks/`
  contents are never parsed or read as data.

## Run

Smoke (fast, no GPU; verifies every pin, the lineage package, the generator, the gate
design, and runs the unit tests):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_state_track_install/scripts/run.py --smoke
```

Regenerate the frozen curriculum (byte-identical; seed 87):

```bash
.venv/bin/python -B experiments/qwen35_4b_state_track_install/scripts/gen_state_track_curriculum.py --seed 87 --rows 160
```

Staged (each requires clean pushed green main plus its committed review verdict;
receipts committed between stages):

```bash
# needs reports/compute_review.md with PASS_CONTROL_TRAINING (~5min GPU)
.venv/bin/python -B experiments/qwen35_4b_state_track_install/scripts/run.py --stage train
# needs reports/local_design_review.md with PASS_CONTROL_MERGE (~10min)
.venv/bin/python -B experiments/qwen35_4b_state_track_install/scripts/run.py --stage merge
# fill the three TODO pins from the committed merge receipt, commit, then:
# needs reports/local_design_review.md with PASS_LOCAL_EVENT (~30min GPU)
.venv/bin/python -B experiments/qwen35_4b_state_track_install/scripts/run.py --stage local
# needs reports/benchmark_design_review.md with PASS_BENCHMARK_EVENT + promotion (~10min GPU)
.venv/bin/python -B experiments/qwen35_4b_state_track_install/scripts/run.py --stage benchmark
```

Standalone lineage verification (also inside smoke):

```bash
.venv/bin/python -B experiments/qwen35_4b_state_track_install/scripts/rebuild_lineage.py --verify-inputs
# full GPU rebuild of stages 1-9 (~4h):
.venv/bin/python -B experiments/qwen35_4b_state_track_install/scripts/rebuild_lineage.py
```

Ops note (torn ledger / partial receipts): never edit receipts by hand; audit the
preserved artifacts, then `--stage benchmark --resume` — the summary regenerates
deterministically and must reconcile byte-identically before the ledger closes.

## Results

Pending: the GPU stages have not run. The terminal artifact will be
`runs/benchmark/medium_tb1024_seed78169_install/summary.json` carrying the frozen
INSTALLED_TRANSFER / BOUNDED consequence.

## Interpretation

Design-frozen. INSTALLED_TRANSFER promotes the composite to program reference artifact
and shows the divergent-skill move class adds where replay is bounded; BOUNDED extends
the install-not-equal-convert law to this skill and redirects to a different dose or
parent. Either way the chain's stage-9 boundary becomes a measured fact.

## Knowledgebase Update

- Program evidence updated: pending the sealed event.
- Program backlog updated: pending.
- Claim ledger updated: pending.

## Artifacts

- `scripts/` — fail-closed staged harness (`run.py`), trainer wrapper (`train_trial.py`),
  merge wrapper (`merge_trained_arm.py`), the designed curriculum generator
  (`gen_state_track_curriculum.py`), gate design/generator (`gen_local_gate.py`), gate
  rule (`check_local.py`), evaluator (`eval_local_vllm.py`), sealed-event runner
  (`run_benchmark.py`), design checker with the three-slot normalized pin
  (`check_design.py`), lineage rebuilder (`rebuild_lineage.py`), vendored production
  copies (`train_think.py`, `merge_adapter.py`, `lineage_trainers/`, `stage7_wrappers/`,
  `rebuild_clean_chain.py`, `gen_curriculum.py`).
- `data/` — the frozen state_track curriculum (`sft_state_track.jsonl`) and its
  token-exposure receipt (`state_track_token_receipt.json`), the standalone lineage
  package (`lineage/`), stage-7/8 production inputs, the parent provenance copy
  (`provenance/count_walk_merge.json`), the three reference cells' eleven sha-pinned
  gate-file copies (`predecessor_gates/`), and the frozen local gate
  (`local_design_receipt.json`, `local_tasks_seed8806*.jsonl`,
  `local_input_seed8806*.jsonl`).
- `reports/preregistration.md` — the frozen contract with honest priors.
- `reports/artifact_manifest.yaml` — external artifacts and reproduction paths.
- `tests/` — 197 unit tests (curriculum generator truth audit + contamination/overlap
  drills, consequence truth table + lattice sweeps + the 1e-12 aggregate tie guard,
  two-sided retention bands, benchmark ledger open/close/reconcile/double-consume, local
  write-ahead ledger, arm authentication tamper drills, sibling-original absent/divergent
  drills, normalized-pin probes for all three fill-slot files, stage-prerequisite refusal
  drills, lineage package integrity over stages 1-9, cross-module frozen constants, vLLM
  runner contract).

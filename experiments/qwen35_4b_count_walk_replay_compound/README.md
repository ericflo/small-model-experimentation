# Count-Walk Replay Compound (Stage 8)

Lifecycle 29 — stage 8 of the documented zero-root chain: REPLAY COMPOUNDING onto the count_walk composite. ONE fresh rank-32/alpha-64 adapter trains on the FULL 2,240-row replay pool (`data/sft_blend.jsonl`, sha `25a9595f…`) from the count_walk composite parent at fresh seed 86 with the chain's established replay-refresh recipe, merges through the vendored external merger, must pass a two-arm three-screen pooled_k3 retention non-drift gate, and only a locally promoted candidate may consume the ONE sealed medium seed 78168 under the frozen two-directional COMPOUNDED / BOUNDED consequence.

**Status:** in-progress · since 2026-07-17 · model-free construction frozen (design receipt, lineage package, gate instruments, tests, drills all green); awaiting the staged reviews and the GPU stages

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can DESIGNED synthetic curricula install universal, transferable
  agentic skills into the one 4B — provably, on fully documented lineage?
- Prior anchors: lifecycle 27 (`qwen35_4b_count_dont_walk_enumeration` — built the
  count_walk parent, tree `d5fdc55c…`; its replay control `replay_ctl7` is the
  direct precedent: a pure replay stream onto the stage-6 composite beat the
  parent on 4 of 5 sealed draws, mean +0.018), lifecycle 28
  (`qwen35_4b_count_walk_menders_confirmation` — AMBIGUOUS on menders, contrast
  closed; count_walk topped the aggregate on 2 of 4 seeds, mean 0.3634 over the
  four), and lifecycle 22 (`qwen35_4b_zero_root_lineage_rebuild` — the
  contamination-free six-stage chain this cell extends).

## Question

Does replay compounding still add aggregate at stage 8 — a fresh adapter over the
full replay pool trained ON the count_walk composite and merged back — or does
the replay-compounding law hit diminishing returns on this parent? Frozen
two-state consequence; the modal BOUNDED path (aggregate up but one family
dipped by more than an episode) is a finding about the law's boundary, not a
failure.

## Hypothesis

The chain added aggregate at every documented stage, and the stage-7 replay
control proved the exact move class (replay pool onto a composite) adds ~+0.02
mean aggregate. If the pool's value is not yet exhausted at a 0.357-mean parent,
the candidate lands COMPOUNDED and becomes the program reference artifact
feeding the raised-floor confirmation. Honest priors (preregistration):
P(aggregate strictly > parent) ≈ 0.5-0.6, but the strict no-family-below-by->0.1
clause historically binds on ~4 of 5 draws, so P(COMPOUNDED) ≈ 0.25-0.40 and
BOUNDED is the believed-likelier verdict.

## Setup

- Model: Qwen/Qwen3.5-4B (revision `851bf6e8…`), always.
- Treatment: `replay_compound` — fresh r32/a64 QLoRA on `data/sft_blend.jsonl`
  (2,240 rows, sha-pinned, zero skips enforced) via `--model-path` on the
  count_walk composite; epochs 1, lr 1e-5, bs 1, ga 8, maxlen 4096, w_think 0.2,
  w_close 0.2, seed 86, 280 optimizer steps (`scripts/train_trial.py`, fail-closed).
- Parent: `large_artifacts/qwen35_4b_count_dont_walk_enumeration/merged/count_walk`
  (tree `d5fdc55c…`, weights `ddd7bc4b…`), authenticated fail-closed pre-training
  and pre-merge against the IN-CELL sha-pinned provenance copy of lifecycle 27's
  merge receipt (`840edca0…`, `data/provenance/count_walk_merge.json`; the
  committed sibling original is a verification aid — byte-identical when
  present, skipped with a recorded note when absent) plus the full 9 GB weights
  hash at BOTH stage boundaries (train_trial.py and merge_trained_arm.py).
- Merge: `scripts/merge_trained_arm.py` → `scripts/merge_adapter.py`
  (`cb9af8b4…`) `--base-model` count_walk →
  `large_artifacts/qwen35_4b_count_walk_replay_compound/merged/replay_compound`.
- Local gate: retention-only (no axis kind exists), TWO arms (parent vs
  candidate), three pooled_k3 screens at fresh seeds 88060/88061/88062 (104 rows
  each, 8 per each of 13 skills), TWO-SIDED bands on integer screen sums:
  correct ±15, parsed ±9, cap contacts ±9. All three must hold to promote.
  Freshness audit against sha-pinned in-cell copies of the two reference cells'
  eight gate files (`data/predecessor_gates/`, seeds 88052-88059). Write-ahead
  local ledger (`runs/local/local_events.jsonl`): every engine event opens
  before launch and sha-pins its raw artifacts after validation; a torn or
  discarded attempt refuses any new local pass.
- Sealed event: medium, tb 1024, fresh seed 78168, three arms in frozen order
  base → count_walk → replay_compound through the trusted gateway
  (`53cf6533…`); three TODO-pin slots for the candidate (tree/weights/committed
  merge receipt) fail closed while unfilled; `run_benchmark.py` frozen by
  `check_design.py`'s three-slot NORMALIZED hash (`11a6cc14…`), with
  `train_trial.py` (`97c06297…`) and `eval_local_vllm.py` (`1b294792…`) pinned
  symmetrically on their own fill slots; one-seed write-ahead ledger with
  byte-equal crash reconciliation.
- FROZEN CONSEQUENCE (no third state): COMPOUNDED iff candidate aggregate
  strictly > parent AND no family below parent by more than 0.1
  (`candidate_family >= parent_family - 0.1 - 1e-9`; every family independently
  gets at most one episode of slack — the rule caps depth per family, not the
  number of families using slack) AND candidate aggregate strictly > base;
  aggregate comparisons carry a 1e-12 tie guard on the gateway-reported floats
  (a true rational tie rendered one ulp apart resolves BOUNDED) — claim:
  "replay compounding holds at stage 8; the composite
  becomes the program reference artifact and feeds the raised-floor
  confirmation." BOUNDED otherwise — claim: "the replay-compounding law hits
  diminishing returns at stage 8 on this parent; the count_walk composite
  remains the reference; further aggregate pushes need a different move class."
  Goal gate vs base (10/10 strict wins) recorded descriptively for both treated
  arms.
- Hidden-label boundary: only `scripts/run_benchmark_aggregate.py` runs;
  `benchmarks/` contents are never parsed or read as data (audited by
  `check_design.py --check` and a unit test).

## Run

Smoke (fast, no GPU; verifies every pin, the lineage package, the gate design,
and runs the unit tests):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_count_walk_replay_compound/scripts/run.py --smoke
```

Staged (each requires clean pushed green main plus its committed review
verdict; receipts committed between stages):

```bash
# needs reports/compute_review.md with PASS_CONTROL_TRAINING (~45min GPU)
.venv/bin/python -B experiments/qwen35_4b_count_walk_replay_compound/scripts/run.py --stage train
# needs reports/local_design_review.md with PASS_CONTROL_MERGE (~10min)
.venv/bin/python -B experiments/qwen35_4b_count_walk_replay_compound/scripts/run.py --stage merge
# fill the three TODO pins from the committed merge receipt, commit, then:
# needs reports/local_design_review.md with PASS_LOCAL_EVENT (~30min GPU)
.venv/bin/python -B experiments/qwen35_4b_count_walk_replay_compound/scripts/run.py --stage local
# needs reports/benchmark_design_review.md with PASS_BENCHMARK_EVENT + promotion (~10min GPU)
.venv/bin/python -B experiments/qwen35_4b_count_walk_replay_compound/scripts/run.py --stage benchmark
```

Standalone lineage verification (also inside smoke):

```bash
.venv/bin/python -B experiments/qwen35_4b_count_walk_replay_compound/scripts/rebuild_lineage.py --verify-inputs
# full GPU rebuild of stages 1-8 (~4h):
.venv/bin/python -B experiments/qwen35_4b_count_walk_replay_compound/scripts/rebuild_lineage.py
```

Ops note (torn ledger / partial receipts): never edit receipts by hand; audit
the preserved artifacts, then `--stage benchmark --resume` — the summary
regenerates deterministically and must reconcile byte-identically before the
ledger closes.

## Results

Pending: the GPU stages have not run. The terminal artifact will be
`runs/benchmark/medium_tb1024_seed78168_compound/summary.json` carrying the
frozen COMPOUNDED / BOUNDED consequence.

## Interpretation

Design-frozen. COMPOUNDED promotes the composite to program reference artifact
and funds the raised-floor confirmation cell; BOUNDED closes the
replay-compounding move class at stage 8 on this parent and redirects aggregate
pushes to a different move class. Either way the chain's stage-8 boundary
becomes a measured fact instead of an assumption.

## Knowledgebase Update

- Program evidence updated: pending the sealed event.
- Program backlog updated: pending.
- Claim ledger updated: pending.

## Artifacts

- `scripts/` — fail-closed staged harness (`run.py`), trainer wrapper
  (`train_trial.py`), merge wrapper (`merge_trained_arm.py`), gate
  design/generator (`gen_local_gate.py`), gate rule (`check_local.py`),
  evaluator (`eval_local_vllm.py`), sealed-event runner (`run_benchmark.py`),
  design checker with the three-slot normalized pin (`check_design.py`),
  lineage rebuilder (`rebuild_lineage.py`), vendored production copies
  (`train_think.py`, `merge_adapter.py`, `lineage_trainers/`,
  `stage7_wrappers/`, `rebuild_clean_chain.py`, `gen_curriculum.py`).
- `data/` — the standalone lineage package (`lineage/`), stage-7 production
  inputs, the stage-8 training pool (`sft_blend.jsonl`), the parent provenance
  copy (`provenance/count_walk_merge.json`), the two reference cells' eight
  sha-pinned gate-file copies (`predecessor_gates/`), and the frozen local gate
  (`local_design_receipt.json`, `local_tasks_seed8806*.jsonl`,
  `local_input_seed8806*.jsonl`).
- `reports/preregistration.md` — the frozen contract with honest priors.
- `reports/artifact_manifest.yaml` — external artifacts and reproduction paths.
- `tests/` — 176 unit tests (consequence truth table + lattice sweeps + the
  1e-12 aggregate tie guard over the demonstrated 1-ulp rational-tie pairs,
  two-sided retention bands, benchmark ledger open/close/reconcile/
  double-consume, local write-ahead ledger open/refuse/complete, arm
  authentication tamper drills, sibling-original absent/divergent drills,
  normalized-pin probes for all three fill-slot files, stage-prerequisite
  one-line refusal drills, lineage package integrity, cross-module frozen
  constants, vLLM runner contract).

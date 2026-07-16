# Qwen35 4b Zero Root Lineage Rebuild

Lifecycle 22 — the provenance question elevated to the program's strongest remaining bet: the six documented contamination-free training stages of the hygiene_explore composite, replayed from a FRESH zero-initialized adapter on the official base (removing the undocumented C53-era gym-line root adapter), then measured ONCE at medium against the original.

**Status:** finished · 2026-07-16 · verdict ZERO_ROOT_DEGRADED (mildly) — the six documented stages alone carry ~90% of the transfer (0.3462 vs the original's 0.3824 over base 0.0713; 7/10 strict wins, ZERO losses) while the undocumented prefix contributes ~0.036 aggregate concentrated in mirage/sirens/rites; the zero-root model beats the original on chronicle/siftstack/stockade; the original read 9/10 on this seed (menders tie — the sweep rate holding)

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family — and is the demonstrated position carried by the DOCUMENTED stages alone?
- Prior anchors: the goal-gate confirmation (AGGREGATE_ONLY; two 10/10 sweeps across four sealed medium seeds 78,154/78,157; aggregate 4/4 at ~0.33-0.38 vs base ~0.06-0.11); the standalone lineage package committed in that cell, whose root adapter carries a HARD provenance boundary (no committed creation receipt anywhere in the repo); the C53-era record that the blend root alone carried ~0.44 quick aggregate — load-bearing, but HOW load-bearing at medium is unmeasured.

## Question

The original composite's lineage is carried by ONE warm-started LoRA adapter: [undocumented C53 'blend' root] → replay_refresh(42) → designed160(43) → close_xi(44) → replay_after_close(47) → designed_fresh(51) → hygiene_explore(55). Every stage is documented, contamination-free, and byte-reconstructable — except the root. If the six documented stages are replayed from a FRESH zero-initialized adapter (same datasets, same fixed seeds, same trainer variants, same hyperparameters, raw pinned HF base everywhere), does the result still carry the demonstrated medium position, or is the undocumented prefix load-bearing?

## Hypothesis

The six stages total ~6,700 training rows over ~3 epochs-equivalent of designed curricula; the mechanism they install (think-channel structure + close-weight shaping) is what the goal-gate events measured. If that is right, the zero-root replay lands within one family of the original on the same seed (ZERO_ROOT_COMPARABLE) and the headline model is contamination-clean end-to-end. If the gym-era root's ~0.44 quick aggregate was carrying medium capability the stages merely preserved, the zero-root arm degrades and the recorded contrast IS the root's contribution.

## Setup

- Model: Qwen/Qwen3.5-4B @ 851bf6e8… (raw pinned HF revision for every training stage and the final merge; never a merged composite as training base).
- Stage replays: `scripts/rebuild_zero_root.py` — stage 1 trains a FRESH rank-32/alpha-64 adapter (trainer's default fresh path, NO `--warm-start`; LoRA-B zero-init so the delta starts at zero) on stage01_replay_refresh.jsonl at seed 42 with the exact recorded stage-1 hyperparameters (lr 1e-5, batch 1, accum 8, maxlen 4096, epochs 1, w_think 0.2, NO w_close, stage-1/2 trainer variant); stages 2-6 warm-start each from the PREVIOUS zero-root stage with their recorded per-stage recipes (stage 3: train_think_close with target_close_kinds=[u_execute,u_induct], target_w_close=1.0). Per-stage receipts → `runs/lineage/stageNN_<name>.json`; the manifest's recorded adapter hashes are CONTRAST fields only (different root ⇒ different bytes, never verification).
- Lineage package: byte-identical copy of `qwen35_4b_goal_gate_confirmation`'s committed package (manifest sha 1f49cd8b…, six datasets, three trainers, merger). The blend root is deliberately NOT vendored — its omission IS the design, and the receipt/tests fail closed if a copy appears under this cell's artifact storage.
- Merge: stage-6 zero-root adapter onto the raw base via the copied merger → `large_artifacts/qwen35_4b_zero_root_lineage_rebuild/merged/zero_root_hygiene_explore`; receipt `runs/lineage/merge.json` pins adapter shas, full output tree sha, weights sha.
- Benchmark (ONE event): medium, tb 1024, ONE fresh sealed seed 78,159 (grep-fresh audited in the design receipt), THREE arms in frozen order: `base` (b654e033…/26d8ee48…), `hygiene_explore_original` (e2112344…/9eb653d7…), `zero_root_hygiene_explore` (pinned post-merge via fail-closed TODO-pins in `scripts/run_benchmark.py`, filled from the committed merge receipt). Hardened single-seed runner: verdict + design-receipt code-pin check at the seed-consuming boundary, write-ahead opened/closed ledger whose closed record sha-pins the summary AND all three per-arm receipts, byte-equal crash reconciliation, clean-slate unopened seed, finiteness guards, implementation signature anchored to the discovery/confirmation block.
- Training seeds 42/43/44/47/51/55 are INHERITED STAGE CONSTANTS (deliberate reuse — they are what "same recipe" means); the only fresh seed in this cell is 78,159.
- Readings (no promotion): per-family table + aggregates for all three arms; goal gate vs base for BOTH composites (forensics-identical strict-win partition); the PREFIX CONTRIBUTION contrast (zero-root minus original, per family and aggregate — "the gym-era root's contribution at medium, one seed, cross-arm same-seed paired"); budget integrity; menders/rites/warren margins (the statechain→rites conversion question does not apply — no statechain stage).
- Consequence (ordered, total): `ZERO_ROOT_COMPARABLE` iff the zero-root aggregate strictly beats base AND its goal-gate strict wins ≥ (original's strict wins on this seed − 1) — "the documented stages alone carry the demonstrated position; the headline model is contamination-clean end-to-end"; `ZERO_ROOT_DEGRADED` otherwise — "the undocumented prefix is load-bearing at medium; its contribution is the recorded contrast".
- Hidden-label boundary: nothing under `benchmarks/` is ever read; gateway receipts only.

## Run

Smoke (no GPU, no writes):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_zero_root_lineage_rebuild/scripts/run.py --smoke
```

Stages (each requires its committed-at-HEAD prerequisites + literal review verdict on clean pushed main):

```bash
# ~2.5-3h GPU: six stage replays + the merge; commit runs/lineage/ after
.venv/bin/python -B experiments/qwen35_4b_zero_root_lineage_rebuild/scripts/run.py --stage rebuild   # needs PASS_REBUILD in reports/compute_review.md
# fill the three TODO-pins in scripts/run_benchmark.py from runs/lineage/merge.json, commit, review, then:
.venv/bin/python -B experiments/qwen35_4b_zero_root_lineage_rebuild/scripts/run.py --stage benchmark # needs PASS_BENCHMARK_EVENT in reports/benchmark_design_review.md
```

## Results

Pending: the rebuild and benchmark stages have not run. The terminal artifact will be `runs/benchmark/zero_root_readout.json` with the frozen consequence.

## Interpretation

Pending the sealed event.

## Knowledgebase Update

- Program evidence updated: pending.
- Program backlog updated: this cell IS the queued zero-root rebuild from the goal-gate confirmation's backlog.
- Claim ledger updated: pending.

## Artifacts

- `data/design_receipt.json`: the frozen design — package pins, stage plan (zero-root rewiring), root-omission block, seed-78,159 freshness audit, arm pins, consequence partition, code pins. `run_benchmark.py` is pinned by a NORMALIZED HASH: exactly the three TODO-pin value slots are canonicalized to a fixed placeholder before hashing, so every other byte (every guard call site included) is frozen pre- and post-fill; the digest and the normalization rule live in the receipt and are re-verified at the seed-consuming boundary.
- `data/lineage/`: byte-identical copies of the manifest + six stage datasets.
- `scripts/lineage_trainers/`, `scripts/merge_adapter.py`: byte-identical trainer/merger copies.
- `scripts/rebuild_zero_root.py`: the six-stage zero-root replay + merge (receipts to `runs/lineage/`).
- `scripts/run_benchmark.py`, `scripts/check_benchmark.py`: the hardened single-seed three-arm event and its provenance-anchored readout.
- `reports/artifact_manifest.yaml`: external artifact declarations (adapters + merged composite live under `large_artifacts/`).


## Erratum (2026-07-16)

The sweep-rate framing in this document ("two full sweeps across four
independent sealed seeds", ~50%) reflects the 78,154–78,157 window and
omits the earlier 78,150 reading (8/10, menders+rites ties). Over ALL six
recorded goal-gate readings the rate is 2/6 (exact 95% CI [0.04, 0.78]),
with menders blocking every miss. See
`experiments/qwen35_4b_sweep_rate_consolidation` for the consolidated
record; the per-seed facts in this document are unchanged.

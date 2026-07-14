# On-Policy Failure-Prefix Universal Curriculum

**Status:** in-progress · since 2026-07-14 · replay control trained; candidate training is next

This result-separated successor tests whether training corrective continuations from
the model's own fresh procedural failure prefixes installs a reusable reasoning and
commit policy better than another idealized trace curriculum or exact-token replay.

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can a contamination-free installed mechanism improve every
  held-out benchmark family rather than redistribute wins?
- Closest near-duplicate:
  `qwen35_4b_universal_state_table_compiler_token_match`.
- Additional anchors: `qwen35_4b_gauntlet_breadth_round1`,
  `qwen35_4b_interactive_policy_curriculum`,
  `qwen35_4b_verifier_conditioned_recovery_bank`, C53, C56, and C59.

## Question

Do masked corrective continuations attached to the authenticated parent's actual
failure prefixes teach bounded execution, induction, scoring, and exact commitment
at the deployed interface while preserving its broad replay policy?

## Hypothesis

The state-table predecessor was executable and truth-audited but off-policy: its
ideal traces did not resemble the model's actual declaration confusion, repeated
induction, score-count errors, or correct-state-without-commit prefixes. Fresh parent
rollouts plus executable-oracle continuations should place supervision exactly at
those states. The mechanism is false unless the sole candidate strictly beats both
the unchanged parent and a same-parent exact-forward-token replay continuation on a
fresh paired local gate, overall and on execute/induct/probe.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, pinned revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Proposed parent: authenticated `close_xi` adapter; the failed scaffold and
  state-table candidates are not inherited.
- Proposed data source: fresh experiment-owned procedural tasks, followed by
  authenticated parent rollouts and executable-oracle failure localization. No prior
  local-gate item may enter training.
- Proposed treatment: masked assistant-prefix correction at first observable failure
  states, including bounded commit, declaration-versus-operation parsing, induction
  loop termination, probe-score recomputation, repair propagation, and exact answer
  serialization.
- Mechanism-falsifying control: independent same-parent replay continuation matched
  on encoded forward tokens, optimizer steps, seed, backend, and aligned shared
  replay positions.
- Hidden-label boundary: `benchmarks/` remains read-forbidden. Only an aggregate
  gateway may run after local promotion; all-family lift, higher-tier confirmation,
  and matched-compute sample-more remain required for a universal claim.
- Reserved seeds: construction `77113`, parent rollout `66113`, training `47`, fresh
  local `88009`, and conditional aggregate `78139`.

## Run

Model-free design smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_on_policy_prefix_repair_token_match/scripts/run.py --smoke
```

The design, parent merge, parent rollout, failure inventory, and exact-token freeze are
separate published checkpoints. Verify every model-free derived artifact with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_on_policy_prefix_repair_token_match/scripts/run.py --smoke
```

After this control checkpoint is pushed to `main` and both required workflows are
green, run exactly the candidate stage from a clean worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_on_policy_prefix_repair_token_match/scripts/run.py --stage train-candidate
```

The wrapper authenticates the committed control log, receipt, and external adapter
before loading the candidate. Local capability and benchmark stages remain sealed.

## Results

CPU feasibility passed. Construction seed 77,113 deterministically freezes 288
truth-audited tasks, 48 for each of six failure classes. The model-facing JSONL omits
hidden oracle and answer fields. Tests cover exact prefix masking, failure-only
selection, delayed-commit cutoff, declaration misuse, generation caps, and the
merged-Qwen architecture gate. The authenticated `close_xi` adapter was explicitly
merged into a full composite: 128/128 applied LoRA modules were nonzero, the merged
weight hash is `4933f2dd...eb373`, and the external merge-receipt hash is
`1fbc84b3...5557`. From pushed-green commit `21e1eb59`, one frozen same-backend
parent event produced all 288/288 greedy natural-thinking rollouts, 170,252 sampled
tokens at 849.9 tokens/s. Rollout/metadata/log/receipt hashes are
`8010632f...3b17f` / `9fe81276...664` / `ed0d4fc4...26b7` /
`c6b98b79...74fa`. The original postvalidator rejected only an impossible
post-open `git_dirty=false` assertion; an explicit no-generation recovery path
authenticated every other frozen field and wrote the receipt without rerunning the
model.

The frozen miner has now graded the experiment-owned substrate. It found 230 failed
and 58 passing parent rows; all 230 failures had a reachable clean prefix. Available
failures by bounded-induction/commit/declaration/probe/repair/state class were
46/48/35/24/36/41, so every fixed quota cleared without borrowing. It selected
exactly ten per class. The 60-row repair source and complete inventory hashes are
`30141538...d84b8` / `7230af52...dfe7`. Selected prefixes contain 47,123 masked
tokens total (33 minimum, 785.4 mean, 1,024 maximum); 42 selections cut at the
generation cap, ten at the immediate-commit boundary, and eight at the answer
boundary.

The separately frozen training streams now contain 320 rows and exactly 304,313
forward tokens apiece, with zero skips, 200 byte-identical position-aligned replay
rows, and 40 optimizer steps. All repairs fit below the 4,096-token ceiling; the
largest final row is 2,991 tokens. The candidate replaces 33,421 replay target tokens
with masked context, leaving 111,983 nonzero-weight tokens and 25,049.4 absolute loss
mass versus 145,404 and 31,311.2 for control. This is an explicit intervention
caveat, not hidden behind the forward-token match. Token-receipt SHA-256 is
`eb08026f...e0cfc`; the second review verdict is `PASS_CONTROL_TRAINING`.

From pushed-green commit `a8529c04`, the replay control then trained for exactly one
epoch and 40 updates from the authenticated parent. It encoded 320/320 rows with zero
skips, consumed the registered 304,313 forward tokens, and finished with training
loss 0.4588 in 272.8 trainer seconds (292.4 wrapper seconds). The normalized
log/receipt hashes are `a49076ec...3501` / `f78f2069...d6de`; adapter config/weights
are `0dfd9bda...120f` / `bb59d3bd...5154d`. Its 256 tensors contain exactly
42,467,328 elements; every tensor is finite and nonzero. This is an operational
training result only. No candidate training, capability measurement, local
evaluation, or benchmark event exists.

## Interpretation

The parent visits every registered failure class often enough to test the mechanism,
and the exact-forward-compute comparison is runnable. This is not evidence that
on-policy correction works. Selection is dominated by long severe prefixes, while
the candidate has fewer supervised tokens and lower loss mass than replay. A win
would show targeted repair beats more replay under matched forward compute; it would
not isolate prefix conditioning from target-composition differences.

## Knowledgebase Update

- Program evidence: unchanged until a capability result exists.
- Program backlog: records the quota-satisfying inventory, exposure caveat, and
  authenticated replay control.
- Claim ledger and shared synthesis: unchanged.

## Artifacts

- `idea_intake.md`
- `configs/default.yaml`
- `scripts/run.py`
- `scripts/gen_rollout_tasks.py`
- `scripts/mine_prefix_repairs.py`
- `scripts/measure_source_tokens.py`
- `scripts/materialize_streams.py`
- `scripts/validate_streams.py`
- `scripts/train_trial.py`
- `data/design_receipt.json`
- `data/rollout_task_manifest.json`
- `data/prefix_failure_inventory.json`
- `data/prefix_repair_source.jsonl`
- `data/source_token_lengths.json`
- `data/stream_manifest.json`
- `data/stream_token_receipt.json`
- `data/replay_after_close.jsonl`
- `data/prefix_repair_after_close.jsonl`
- `runs/parent_rollout/seed66113.receipt.json`
- `runs/training/replay_after_close.log`
- `runs/training/replay_after_close.json`
- `analysis/prefix_failure_inventory.md`
- `reports/design_review.md`
- `reports/compute_review.md`
- `reports/preregistration.md`
- `reports/report.md`
- `reports/artifact_manifest.yaml`

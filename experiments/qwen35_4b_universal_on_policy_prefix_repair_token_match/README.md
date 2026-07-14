# On-Policy Failure-Prefix Universal Curriculum

**Status:** in-progress · since 2026-07-14 · authenticated parent rollout collected; prefix mining is next

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

The design, parent-merge, and parent-rollout checkpoints are published separately.
After the rollout receipt is pushed and both required workflows are green, run exactly
one model-free stage:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_on_policy_prefix_repair_token_match/scripts/run.py --stage mine-prefixes
```

Publish the resulting failure inventory before any further stage. Training remains
unavailable until actual prefix lengths support exact-token streams, a zero-skip
receipt, and a second adversarial compute review.

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
model. Failure outcomes have not been graded. No training, capability measurement,
local evaluation, or benchmark event exists.

## Interpretation

This is a collection-authentication result, not evidence that on-policy correction
works. The explicit merge closes vLLM's runtime-LoRA silent no-op and the rollout is
complete, but failure quotas may still prove unreachable; that outcome stops training
and will be preserved.

## Knowledgebase Update

- Program evidence: unchanged until a result exists.
- Program backlog: records this active result-separated intake.
- Claim ledger and shared synthesis: unchanged.

## Artifacts

- `idea_intake.md`
- `configs/default.yaml`
- `scripts/run.py`
- `scripts/gen_rollout_tasks.py`
- `scripts/mine_prefix_repairs.py`
- `data/design_receipt.json`
- `data/rollout_task_manifest.json`
- `runs/parent_rollout/seed66113.receipt.json`
- `reports/design_review.md`
- `reports/preregistration.md`
- `reports/report.md`
- `reports/artifact_manifest.yaml`

# On-Policy Failure-Prefix Universal Curriculum

**Status:** in-progress · since 2026-07-14 · intake only; CPU feasibility and adversarial design review remain before rollout collection

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

Intake smoke only:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_on_policy_prefix_repair_token_match/scripts/run.py --smoke
```

No rollout, materialization, training, local evaluation, merge, or benchmark command
is authorized yet. CPU feasibility and `reports/design_review.md` must freeze the
backend, prefix-masking contract, oracle boundary, exact-token control, and promotion
logic first; that design then needs its own pushed-green checkpoint.

## Results

Intake only. Related-work discovery selected the existing program, named the
state-table predecessor as the closest near-duplicate, and reserved fresh identities.
The smoke authenticates the proposed parent and repository-local intake contract. No
model generation, training, capability measurement, merge, or benchmark event exists.

## Interpretation

This is authorization to test feasibility and adversarially review the design, not
evidence that on-policy correction works. The intervention must remain a prospective
single-candidate package until a later ablation separates its failure classes.

## Knowledgebase Update

- Program evidence: unchanged until a result exists.
- Program backlog: records this active result-separated intake.
- Claim ledger and shared synthesis: unchanged.

## Artifacts

- `idea_intake.md`
- `configs/default.yaml`
- `scripts/run.py`
- `reports/report.md`
- `reports/artifact_manifest.yaml`

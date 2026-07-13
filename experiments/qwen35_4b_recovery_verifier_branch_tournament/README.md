# Public-verifier recovery branch tournament

**Status:** finished

## Research Program

- Program: `agentic_breadth_installation`
- Direct predecessor: `qwen35_4b_recovery_payload_budget_harness`.
- Prior anchors: C46 (execute visible evidence before confidence), C54
  (serial-compute capability), and the three conditional-recovery experiments.

## Question

Can a frozen public-only rule combine the complementary action-only and λ=.18
recovery policies into a stronger coding agent on wholly new procedural
families, while beating two full same-policy trajectories at identical reserved
model-token compute?

## Hypothesis

The predecessor's two local policies tied at 68.75% on confirmation but had a
78.75% hidden-success union on both independent transfer blocks. Their public
visible-test state was informative: choosing action-only only when its final
workspace passed visible tests and λ=.18's did not scored 75.0% on both blocks,
captured 95.2% of the union, and exceeded the exact randomized-policy
expectation by 6.25 points. This rule should prospectively recover useful policy
diversity without predicting a latent winner or changing shared weights.

The hypothesis fails if the retrospective rule does not transfer to new
algorithm families, if same-policy sample-more matches it, or if its apparent
gain requires hidden outcomes.

## Frozen Intervention

For each rejected-patch or failed-visible-test state:

1. Run one greedy six-call branch with action-only and one with λ=.18.
2. Run the public visible suite on each final workspace.
3. Choose action-only iff action passes and λ=.18 does not; otherwise choose
   λ=.18.

No family label, hidden test, model self-report, entropy threshold, or posthoc
margin enters the decision. Each branch receives 512 thinking + 512 answer
tokens per call. The 12-call tournament reserves 12,288 tokens/case, exactly
matching two complete six-call stochastic trajectories from candidate or
action-only. The same-policy controls receive an oracle-generous pass-if-either
score, making them harder—not easier—to beat.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, pinned revision `851bf6e...`.
- Frozen checkpoints: C54 apex context control, recovery action-only, and the
  locality-safe λ=.18 reason mixture; exact hashes are in the config.
- Selector qualification: checksum-frozen predecessor dev/confirm trajectories;
  this is retrospective design evidence only.
- Prospective substrate: four new procedural repository families—deadline
  queues, labeled interval coalescing, atomic reservations, and fallback-chain
  resolution—absent from every prior recovery block.
- Blocks: 80 controlled recovery cases at seed 85000, then an independent 80 at
  seed 85100. The rule cannot adapt after either.
- Controls: each source policy, C54 apex, candidate pass-if-either sample-more,
  action pass-if-either sample-more, exact expected random policy choice, and a
  deterministic random-choice diagnostic.
- Firewall: hidden executable code/output remain host-only. No `benchmarks/`
  source, item, transcript, or result is read or imported.

## Gate Order

1. Verify all model/source hashes, procedural unresolved-state invariants,
   selector hidden-label independence, and exact compute reservations.
2. Run every branch and same-policy sampling control on prospective dev.
3. Before scoring the selector, prove the deterministic action/candidate union
   can mathematically clear both same-policy sampling controls by 3pp.
4. Score the frozen public selector. It must beat the best source, candidate
   sample-more, action sample-more, and exact random-choice expectation by 3pp;
   have paired-bootstrap lower bounds ≥0 versus the best source and sample-more;
   capture ≥85% of the source union; retain both recovery transitions and tool
   validity; and avoid family collapse.
5. Repeat the complete control-first battery unchanged at seed 85100.
6. Passing both blocks authorizes a separate transition-balanced winner-bank
   experiment. It does **not** authorize Menagerie here because this experiment
   changes the harness, not a single deployable checkpoint.

Exact thresholds and statistical details are frozen in
`reports/preregistration.md`.

## Run

```bash
.venv/bin/python experiments/qwen35_4b_recovery_verifier_branch_tournament/scripts/run.py --smoke
.venv/bin/python experiments/qwen35_4b_recovery_verifier_branch_tournament/scripts/run.py --gpu-smoke
.venv/bin/python experiments/qwen35_4b_recovery_verifier_branch_tournament/scripts/run.py --full
```

## Results

**Verdict: `PROSPECTIVE_DEV_INFEASIBLE`; stopped before selector scoring.**
Confirmation, winner banking, and Menagerie remained sealed.

| Prospective-dev arm | Success | Mean sampled tokens | Atomic reservations |
| --- | ---: | ---: | ---: |
| C54 base | 61.25% | 2,248 | 0/20 |
| λ=.18 greedy | 73.75% | 1,439 | 0/20 |
| action-only greedy | 73.75% | 1,537 | 0/20 |
| λ=.18 pass-if-either ×2 | 73.75% | 3,384 | 0/20 |
| action pass-if-either ×2 | **75.0%** | 3,395 | 1/20 |
| deterministic mixed-policy union | **75.0%** | — | 0/20 |

The deterministic source policies each solved 59/80 cases, with only one
exclusive win apiece; their union was 60/80. That ceiling could not reach the
registered 61.4/80-equivalent bar over either source, much less beat action
sample-more by 3pp. All three feasibility checks failed, so the public selector
was never scored prospectively.

The negative is sharply localized. Fifty-eight cases were solved by both
sources and all 20 shared failures were `atomic_reservations`; both policies
retained 100% changed-patch-within-two behavior. The traces repeatedly fixed
atomic validation or input immutability separately, then regressed the other
constraint. Action sample-more found the full conjunction once in 20 cases.
Full metrics and hashes are in `reports/result_receipt.json`.

## Interpretation

The predecessor's complementarity was algorithm-specific and did not transfer
to four new families. Public branch selection cannot create proposal coverage
when both sources share the same failure core; spending the same compute on
more action trajectories is at least as good.

This redirects the curriculum from *selecting among local policies* to
*installing a missing compositional invariant*. The shared failure is not a
loop-policy defect: agents inspect, patch, test, and revise, but oscillate
between “validate the whole request before any update” and “copy state before
updating.” A successor should use executable tool-found solutions across
multiple transactional families, preserve the existing conditional transition
bank as replay, and require transfer to unseen transactional families plus the
broad recovery suite before Menagerie.

## Knowledgebase Update

- Program evidence: records the non-transferring union and shared
  multi-constraint failure core.
- Program backlog: retires public portfolio selection here and queues
  transaction-invariant scaffold distillation with recovery replay.
- Claim ledger: unchanged; no checkpoint or Menagerie result was produced.

## Artifacts

Small selector and gate receipts are committed. Detailed model trajectories
live under `large_artifacts/qwen35_4b_recovery_verifier_branch_tournament` as
specified by `reports/artifact_manifest.yaml`.

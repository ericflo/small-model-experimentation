# Idea Intake: Long-Horizon Answer-Potential SFT

## Rough Idea

Sample many naturally completed Qwen3.5-4B thoughts, rank them by the likelihood they assign to a known
correct answer, preserve diverse full strategies, and bank them through SFT before attempting compression.

## Routing

- Primary program: `posttraining_and_adaptation`.
- Secondary programs: `evidence_conditioned_selection`, `test_time_reasoning_budget`.
- Closest duplicate: `qwen35_4b_answer_potential_trace_sft` (C51).
- Other anchors: `qwen35_4b_bank_the_thoughts` (C28),
  `qwen35_4b_gauntlet_breadth_round1` (C50), C24, C44, C45.

## Why It Is Not A Duplicate

C51 never tested SFT and almost never observed a completed thought. This follow-up changes the termination
event, candidate scale, selection objective, and downstream causal test. It is the first repository run to
compare answer-potential-selected complete self-thoughts against length-matched random, binary-success,
answer-only, and shuffled-thought SFT.

## Novelty Claim

The repository has not tested whether dense reference-answer likelihood can mine rare complete reasoning
strategies from a large natural-close self-sample and install them more effectively than binary rejection
sampling.

## Mechanism-Falsifying Control

Task-shuffle the exact selected trace multiset within family/level/length strata. If the gain persists, the
task-specific reasoning content is not causal. Random natural and empty-thought arms further isolate length,
style, and answer-seam learning.

## Hidden-Label Boundary

Canonical answers and verifiers curate and grade only. No hidden label is included in prompts or available to
the deployed model/selector. Oracle pass@k is a labeled ceiling, never a deployable headline.

## Evidence Output

Termination receipts; 95k-candidate manifests; answer/joint-potential calibration; pivot adoption; selected
SFT datasets; six-arm seed-42 adapters; fresh IID/hard/held-family evaluations; matched-compute sample-more;
and a claim-ledger update that either reverses, scopes, or strengthens C51/C28.

## Decision

Create a new experiment and commit this intake plus the full preregistration before GPU work.

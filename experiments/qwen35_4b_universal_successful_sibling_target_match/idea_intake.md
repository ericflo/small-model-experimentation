# Idea Intake: Policy-Supported Successful-Sibling Universal Curriculum

## Program Fit

- Program: `agentic_breadth_installation`.
- Existing or new program: existing; this is the queued result-separated successor in the universal installation line.
- Closest program scorecard reviewed: Agentic Breadth Installation.
- Related future queue item: `synthetic_curriculum_transfer_bakeoff` is broader; this trial is the narrower failure-selected policy-support test.
- Discovery query: `make related QUERY="successful sibling distillation greedy failure verifier correct policy support universal curriculum"`.

## Prior Evidence

- Anchor 1: C54 found that training on the parent’s own shortest correct traces can compress serial compute, although its aggregate frontier did not yield one universal winner.
- Anchor 2: `qwen35_4b_universal_on_policy_prefix_repair_token_match` showed that long masked failure-prefix continuation loses to replay and does not install earlier decisions.
- Anchor 3: C29 found SFT on correct self-samples more promising than preference training on correct/wrong pairs, while warning that self-generated signal is coverage-bounded.
- Closest near-duplicate: `qwen35_4b_universal_failure_selected_restart_target_match`. It selected the same kind of fresh parent failures but taught hand-authored oracle restarts. Those restarts reduced caps while losing correctness and every execute/induct/probe target row.

## Novelty Claim

This is the first universal-curriculum trial to distill short verifier-correct trajectories sampled from the same deployed parent specifically where that parent failed greedily, under a prospective per-skill availability gate and exact-exposure replay.

## Mechanism

The sampled successful sibling is evidence that both the reasoning path and answer seam are reachable under the deployed policy. Teaching the shortest reachable trajectory from the original prompt should avoid the off-policy oracle-language problem that survived both prefix removal and exact exposure matching. The explanation is false if the candidate does not strictly beat both the unchanged parent and independently trained replay overall and on execute/induct/probe.

## Control Plan

- Baseline: authenticated explicit `replay_after_close` parent.
- Mechanism-falsifying control: independent continuation of that same parent on replay only, with exact forward-token, loss-bearing-target, loss-mass, update, and shared-row equality.
- Shift or robustness check: fresh 26-task local gate across all 13 skills, followed conditionally by aggregate-only held-out transfer and result-separated confirmation.
- Hidden-label boundary: executable procedural truth may grade collection outputs but never enters model collection input; `benchmarks/` is unread and aggregate seed 78,141 is sealed.

## Evidence Output

- Program evidence update: two collection receipts, greedy-failure inventory, per-skill sibling availability, selected source provenance, exact-exposure receipt, paired training/deployment, and conditional promotion.
- Claim ledger or synthesis update: only after a result changes the supported interface law or reaches held-out transfer.
- Reusable artifact: prospective two-event same-parent collection and shortest-success selection harness.
- Stop or branch condition: fewer than four greedy failures or fewer than four qualified successful-sibling tasks in any skill ends this experiment; no threshold change, quota borrowing, new sampling event, or oracle fallback is allowed.

## Decision

- Run experiment: proceed through the committed model-free design and one authenticated greedy event only.
- Create program: no.
- Write synthesis only: no; policy-supported complete trajectories are materially untested in this line.
- Defer: sibling sampling until the greedy receipt is published green; all training until successful-sibling quotas and exact exposure pass a second review.

Reserved construction/greedy/sibling/selection/training/local/aggregate seeds are `77115/66115/66116/55115/49/88011/78141`. They cannot change after their corresponding event.

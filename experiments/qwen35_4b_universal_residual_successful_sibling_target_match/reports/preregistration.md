# Preregistration

## Causal question

Does distilling the deployed parent’s shortest correct sampled trajectory on its live residual skills improve fresh greedy behavior beyond exact-exposure replay, while replay preserves skills that no longer supply enough failures?

The predecessor stopped before sampling because a four-per-all-13-skills failure quota was infeasible. This result-separated experiment uses that published outcome prospectively: treatment is restricted to the ten skills with at least four hard failures, while all 13 remain in evaluation.

## Fixed identity and inherited lineage

- Only model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: authenticated explicit `replay_after_close` composite; runtime LoRA forbidden.
- Inherited source/failure-inventory/greedy-receipt/stop-receipt hashes: `9071ce57...603e9` / `8e21caf8...d783` / `cee1f19d...4962` / `3397b773...2a6e`.
- Inherited construction/greedy seeds: 77,115 / 66,115.
- Fresh sibling/selection/training/local/aggregate seeds: 66,117 / 55,116 / 50 / 88,012 / 78,142.

## Prospective residual policy

Residual treatment skills are exactly induct, execute, trace, verify, repair, optimize, abstain, state, order, and probe. Their inherited hard-failure counts are 46, 39, 12, 19, 30, 24, 6, 17, 11, and 21, totaling 225. Select, count, and route are excluded from treatment because their published counts are 2, 0, and 0. This definition cannot change after sibling outcomes.

The derived model input contains original messages, IDs, and public metadata only. It excludes answers, reference thoughts, audits, truth flags, and expected answers.

## Sole sibling event

Run all 225 residual failures through the same authenticated parent and pinned vLLM runner in one event: natural thinking, `n=16`, seed 66,117, temperature 0.6, top-p 0.95, top-k 20, 1,024 generation tokens, model length 4,096, and the frozen engine geometry. No per-skill resampling, new seed, larger pool, or changed temperature is allowed.

A sibling qualifies only if it naturally stops without truncation, closes thinking exactly once, has the exact canonical answer tail, matches executable truth, and uses at most 768 thinking tokens. Choose the shortest qualified sample per task and then the four shortest task winners per residual skill with seeded ties. Any residual skill below four stops the experiment. Never fill a missing sibling with oracle text or a row from a retention skill.

## Conditional exact-exposure training

Only a balanced 40-row source may open compute construction. Copy the published 2,240-row replay pool and 200-row aligned replay core. Prospectively target 320 rows per arm: candidate = 200 common replay + 40 siblings + 80 replay fillers; control = 200 common replay + 120 disjoint replay rows. Match exactly on forward tokens, loss-bearing target tokens, and absolute loss mass with zero skips.

Both arms independently warm-start from the same parent for one epoch, batch size one, gradient accumulation eight, 40 updates, LR `1e-5`, thought/close weights `0.2/0.2`, and seed 50. An exact solver failure is terminal. Do not pad, duplicate, truncate, mask, rewrite, or reweight sibling targets. A second adversarial compute review must be committed and green before control training.

## Fresh local and held-out gates

Fresh seed 88,012 retains two tasks for each of all 13 skills. Parent, replay, and candidate use identical explicit-composite vLLM geometry. Candidate must pass the unchanged absolute accuracy/parse/cap/route and per-execute/induct/probe floors, then strictly beat both controls on total correct and the six execute+induct+probe rows. Retention skills remain visible and any candidate regression can block the total gate; additional per-skill counts are reported.

Only local promotion may open aggregate seed 78,142. Candidate must strictly lift aggregate and every reported public family versus parent and strictly beat active replay aggregate. A pilot pass still requires result-separated independent quick seeds, the higher tier, paired uncertainty, and same-backend matched-compute sample-more before a universal claim.

## Checkpoint order

1. Publish this design on `main`, rebase, rerun smoke/check, push, and verify both workflows green.
2. Run only `collect-siblings`; preserve and publish every raw artifact.
3. Run only model-free `select-siblings`; preserve a quota pass or terminal stop and publish it.
4. Give every stream, review, training, merge, local, and conditional aggregate stage its own checked, rebased, pushed, two-workflow-green checkpoint.

Every expensive stage requires clean synchronized `main` and its prerequisite receipt committed at `HEAD`. Generated conflicts are resolved by regeneration from the combined source tree.

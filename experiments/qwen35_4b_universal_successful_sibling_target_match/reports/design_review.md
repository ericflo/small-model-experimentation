# Adversarial Design Review

**Verdict:** `PASS_GREEDY_COLLECTION`.

This verdict authorizes exactly one authenticated greedy parent event after the design commit is pushed and both required workflows are green. It does not authorize sibling sampling, training, merge, local evaluation, or benchmark access.

## 1. Is this merely rejection sampling?

Yes at collection time, deliberately—and that is the mechanism under test, not yet the claim. C11 says self-harvest is bounded by existing sample coverage. This experiment asks the narrower unresolved question: when a correct complete path is present in support but absent greedily, can banking that path change the deployed mode? It cannot establish a capability absent from all 16 samples. The per-skill availability gate exposes that limitation rather than filling it with oracle text.

## 2. Could oracle truth leak into generation?

The source contains executable truth, but both runner inputs are independently serialized from original user messages and public metadata. The greedy input excludes answer, reference thought, and audit fields. The sibling input is created only from committed greedy failures and additionally excludes expected answers. Truth is used after generation only as a verifier. Tests reject every named oracle field in model-facing bytes.

## 3. Is sampling adaptive after seeing outcomes?

No. Greedy and sibling seeds, `n=16`, temperature/top-p/top-k, cap, model, runner, and selection thresholds are frozen now. All hard greedy failures are sampled once. There is no second pool, per-skill resampling, changed temperature, larger `n`, or new seed if availability is thin.

## 4. Is “short” a post-hoc label?

No. The maximum is 768 thinking tokens, fixed before collection. Within that set the deterministic shortest qualified sibling wins. This cap rejects successful paths that consume most of the 1,024-token deployment budget. It may make a rare skill unavailable; that is an informative feasibility stop, not grounds to relax it.

## 5. Could answer correctness hide malformed or unusable traces?

Qualification also requires one natural close, a natural stop, no truncation, and a canonical exact answer tail. This does not prove the reasoning prose is causally faithful. The mechanism claim therefore belongs to the complete selected package, and fresh semantic transfer—not trace aesthetics—is the discriminator.

## 6. Could selection favor easy failures and erase breadth?

The trial requires four independently failed tasks in every one of 13 skills, then four policy-supported successes per skill. No skill may donate quota. Shortest-first selection may favor easier instances within a skill, but that is prospectively part of the compression mechanism. The unchanged fresh local gate requires execute, induct, and probe wins, preventing aggregate closure gains alone from promoting.

## 7. Is replay a neutral control?

No. Replay is an active intervention and has repeatedly improved held-out behavior. Both arms must independently continue the same parent, and the candidate must strictly beat replay. Future stream construction must match forward tokens, nonzero targets, absolute loss mass, optimizer steps, and 200 aligned rows. If exact matching is infeasible, this experiment stops; targets cannot be altered to rescue it.

## 8. Does the model receive more inference compute during deployment?

No. Training-time collection uses sampling, but local and aggregate deployment compare greedy explicit composites on identical backend geometry. A later universal claim still owes matched-compute sample-more on held-out tasks. This trial cannot discharge that debt using its training collection.

## 9. Are task surfaces fresh enough?

Construction seed 77,115 regenerates 624 unique messages. The manifest checks zero overlap with both closest predecessors and all reserved local seeds through 88,011. The held-out benchmark remains unread, so no benchmark shape or transcript enters this design.

## 10. Can an operational failure silently mutate the event?

Collectors refuse overwrite, bind raw output/metadata/log hashes, preserve failed logs, and permit only authentication-only recovery of already complete raw outputs. Every stage requires clean synchronized `main` and a committed prerequisite. Runtime LoRA is forbidden; the parent is the authenticated full composite.

## Decision

The main threats—coverage boundedness, oracle leakage, adaptive resampling, post-hoc shortness, replay strength, compute mismatch, and lifecycle drift—are explicit and fail closed. Run one greedy collection only. Re-review exact exposure after successful-sibling selection before any training.

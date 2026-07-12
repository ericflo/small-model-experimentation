# Preregistration — repository search-compress-bank curriculum

Frozen before any result-bearing model generation or training.

## Primary question

Does replay-minimized, operator-balanced compact plan/action banking improve one deep Qwen3.5-4B coding-agent trajectory on family-disjoint procedural repositories beyond both an identical action-only control and matched-call sampling from the strongest regenerated incumbent, without erasing verification/commit behavior or broadly perturbing unrelated next-token logits?

## Fixed model and provenance

- Only model: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Search teacher: the existing merged C53 incumbent blend, itself a checkpoint of the same model.
- Training replay: committed C54 apex data plus this experiment's own verified repository bank; no benchmark item, external model, teacher trace, or human solution is used.
- Bulk generation: the experiment-local pinned vLLM runner for every compared arm.
- Candidate deployment: merged checkpoints only; runtime LoRA comparison is forbidden by C49.

## Splits and compute

- Harvest: six train families × 24 tasks, seed 73100; four sampled trajectories/task; eight turns/trajectory; 512 think + 256 answer tokens/turn.
- Trained-family dev: six train families × 8 fresh tasks, seed 73200.
- Transfer dev: four never-trained families × 18 tasks, seed 73300.
- Transfer confirmation: the same four never-trained families × 18 new tasks, seed 73400.
- Deep policy: one greedy eight-turn trajectory.
- Matched-sampling policy: two independently seeded sampled four-turn trajectories. Both reserve eight calls and 6,144 sampled tokens/task.

Task IDs and public manifests are deterministic. Harvest, dev, confirmation, locality, training, and any later benchmark seeds are disjoint.

## Compression and banking algorithm

For each harvest task, rank successful trajectories by replay-minimized patch count, sampled tokens, then turns. Starting from source-changing patch calls, greedily remove a patch whenever a fresh repository replay still passes both visible and hidden tests. From the remaining patches, construct a canonical trace that reads each patched file, applies the patches in order, runs visible tests, and submits. The trace is admitted only if fresh replay passes visible tests, private tests, and terminal submission.

Every canonical state yields one target with a compact plan and exact JSON tool action. Operators are INSPECT, PATCH, VERIFY, and COMMIT. Set per-row weights so each operator has exactly equal total row loss mass, then multiply all repository rows by the frozen 4.0 repository dose. The action-only control is byte-identical in contexts/actions/weights and differs only by zero plan-span loss.

## Pre-training gates

Stop before training unless:

- every generator fails visible and hidden tests initially and passes both under the host-only oracle;
- no hidden executable or oracle field appears in serialized trajectories/banks;
- at least 40% of harvest tasks and 20% in every family have a verified successful trajectory;
- at least 200 compact rows survive;
- 100% of admitted canonical traces replay successfully;
- all four operators exist and have equal loss mass.

These absolute gates are feasible because all are bounded below the oracle ceiling of 1.0; the machine receipt must state this before search/training continuation.

## Training

Train `apex_replay`, `action_only`, and `compact` from the pinned base with QLoRA r32/alpha64/dropout 0.05, all seven projection modules, learning rate 2e-4 cosine, batch 4 × accumulation 4, max length 4096, seed 42, and exactly 584 optimizer steps. Row-weighted cross entropy uses absolute normalization; negative C54 contrast rows push down answer spans only. Repository compact-plan spans use weight 0.2; action-only plan spans use zero.

No arm is selected or retuned using Menagerie outcomes. One registered dose is tested. If compact fails action-only or transfer gates, this recipe stops; dose/threshold changes require a new experiment.

## Primary non-benchmark gate

On transfer dev, `compact` must satisfy all of:

- success delta ≥ +0.05 versus apex deep;
- success delta ≥ +0.03 versus action-only deep;
- success delta ≥ +0.03 versus apex matched sampling;
- deterministic paired-bootstrap 95% lower bound ≥ 0 for each required success contrast;
- trained-family delta versus apex ≥ -0.03;
- invalid-action rate no more than apex +0.02;
- verification-after-final-patch among successes ≥0.70 and no more than 0.05 below apex;
- commit-after-pass among verified trajectories ≥0.65 and no more than 0.05 below apex;
- median non-target next-token logit drift from apex on 48 frozen unrelated contexts ≤0.15 logits.

The same success/operator conditions must reproduce on transfer confirmation. Exact equality at a threshold passes.

## Menagerie license and verdict

Menagerie stays sealed until both transfer blocks and locality pass. Then, and only then, assign two fresh seeds after union-checking the benchmark seed registry, run paired aggregate-only `quick` and `medium` events through `benchmarks/menagerie/run.py`, and compare compact against regenerated apex and C53 blend on the same backend/decode.

An incremental positive requires compact to improve at least one tier by ≥0.02 over apex while regressing neither tier by >0.03, with the direction reproduced on the second paired seed. Because the concurrent `qwen35_4b_pareto_policy_integration` experiment owns the Pareto-consolidation question, this experiment will claim only coding-curriculum transfer, not Pareto integration. A gate failure preserves the negative and consumes no benchmark seed.

## Interpretation boundaries

- Passing repository transfer but flat Menagerie means a coding-harness-local install, not general capability.
- Compact ≈ action-only means the action policy, not compressed planning, caused any gain.
- Better deep but not matched sampling means extra serial depth did not beat the sampling baseline.
- Operator loss or high locality drift invalidates a headline success even if final hidden tests rise.
- Entropy/varentropy correlations are exploratory routing diagnostics only and cannot retroactively change training pressure, examples, or gates.

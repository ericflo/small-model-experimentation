# Idea Intake: Balanced-Core Answer-Potential SFT

## Rough Idea

Complete the smallest balanced slice of the already-running long-horizon trace harvest, use a
parity-gated fast scorer, and test whether answer-potential-selected full thoughts are better SFT targets
than random, successful, shortest, and task-shuffled natural thoughts.

## Routing

- Primary program: `posttraining_and_adaptation`.
- Secondary programs: `evidence_conditioned_selection`, `test_time_reasoning_budget`.
- Closest duplicate: `qwen35_4b_long_horizon_answer_potential_sft`.
- Earlier anchor: `qwen35_4b_answer_potential_trace_sft` / C51.
- Mechanism anchors: C28, C50, C24, C44, and C45.

`make related QUERY="balanced three-family long-horizon answer potential trace SFT shortest control fast
vLLM scorer"` ranked the three named programs first and the two answer-potential experiments as the closest
experimental matches.

## Why It Is Not An In-Place Amendment

The parent already revealed calibration and throughput evidence. Its original nine-family/pivot protocol
must remain frozen. This experiment prospectively fixes a smaller claim and imports only checksum-verified
raw task shards. It does not reinterpret the parent as completed.

## Novelty Claim

The repository has not causally compared answer-potential banking against the strongest observed surface
selector—trace brevity—using a large, naturally closed, task-balanced pool and fresh SFT evaluation.

## Observed Design Inputs

- Parent answer/joint AUROC: 0.597/0.678.
- Parent top-one rollout lift over random: +0.0684/+0.0625.
- Negative-length AUROC/top-one success: 0.690/0.2656.
- 331 harvested tasks: 21,184 traces and 97,883,041 sampled thought tokens.

These values motivate the shortest control and the compute funnel. They are not counted as confirmatory
evidence for this experiment.

## Mechanism-Falsifying Controls

- `shortest_natural`: tests whether compactness, not dense answer potential, identifies the useful trace.
- `potential_shuffle`: tests whether task-specific reasoning content causes any gain.
- `random_natural`: isolates long-thought/style exposure at matched treatment length.
- `success_rft`: tests ordinary binary rejection sampling.
- base sample-more, if Stage B triggers: enforces the program's matched-compute deployment bar.

## Hidden-Label Boundary

Reference answers and procedural verifiers are oracle-side curation and grading tools only. They never
enter deployment prompts. Evaluation task outputs remain sealed until the prospective design is committed.

## Evidence Output

An immutable inherited-pool receipt; a 360-task balanced harvest manifest; joint HF/vLLM parity receipt;
six exact-token SFT datasets and adapters; mandatory three-split Stage-A evaluation; conditional full and
k=8 Stage-B evaluation; paired uncertainty; and a program/claim update whether potential, brevity, ordinary
success filtering, or none of them banks useful behavior.

## Decision

Create a new experiment. Hard-stop generation at 360 tasks and optional expansion at the frozen Stage-A
trigger.

# Idea intake — think-block FTPO round 1

- **Idea**: apply final-token preference optimization (FTPO;
  docs/final_token_preference_optimization.md) inside the think channel to
  install agentic capability from the model's own generations. Primary signal:
  outcome-conditioned pivot mining (prefix-tree divergence over n verifier-
  scored rollouts) — steering, not suppression. Origin: user direction
  (2026-07-10) toward "pivot tokens that lead to better lines of thinking",
  refining an initial loop-repair framing.
- **Closest near-duplicates checked** (scripts/find_related.py + queue scan):
  - `hard_negative_training_transfer` (P1 program-seed) — mined hard negatives,
    but sequence-level and answer-seat; no single-position objective, no
    think-channel target, no menagerie arbitration.
  - `failure_mined_curriculum_generator` (P2 program-seed) — failure-slice
    curriculum; generation-side, not preference-objective.
  - `posttraining_method_shared_substrate` (P0) — method comparison harness;
    complementary, not overlapping (FTPO could join its roster later).
  - C29 (`qwen35_4b_learn_from_failures` DPO): the corpus's preference-training
    negative prior; this experiment is its direct token-level counter-test.
- **Programs**: agentic_breadth_installation (primary; different-mechanism
  recipe for the post-C50 frontier), posttraining_and_adaptation,
  test_time_reasoning_budget (census informs its loop-control mandate).
- **Decision**: new experiment (no queue item covers FTPO/think-channel
  preference training); scaffolded as qwen35_4b_think_ftpo_round1.

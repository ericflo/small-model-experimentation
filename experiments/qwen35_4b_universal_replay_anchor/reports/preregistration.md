# Preregistration

Frozen before this experiment consumes a benchmark seed.

## Arms

1. `base`: pinned reserialized Qwen/Qwen3.5-4B.
2. `blend`: immutable C53 adapter and merged composite.
3. `replay_refresh`: warm-start `blend`; one epoch over an exact materialized dose of
   1,520 replay rows.
4. `warm_union`: warm-start `blend`; one epoch over an exact materialized dose of 400
   designed rows and 1,120 replay rows.

Both trained arms use learning rate `1e-5`, rank 32, alpha 64, batch 1,
gradient accumulation 8, max length 4,096, `w_think=0.2`, seed 42, and exactly 190
optimizer steps. The 1,120 replay rows are byte-identical and shared between arms;
`replay_refresh` replaces the candidate's 400 designed rows with 400 additional replay
rows. The exact nested subsets are materialized before training rather than relying on
Trainer's fractional-epoch prefix semantics. Token receipts freeze 1,231,404 candidate
forward tokens and 1,444,589 control forward tokens; the replay-only control therefore
receives 17.3% more token compute, a conservative asymmetry for attributing any
candidate improvement to designed rows.

The fresh local screen seed is 88,003. The one allowed pilot event is Menagerie
quick@1,024 seed 78,133.

## Selection and evaluation

- Both trained arms must pass corpus and training integrity gates. The `warm_union`
  candidate alone must pass the designed-task installability gate; `replay_refresh` is
  trained only after that pass and enters the paired aggregate event as a mechanism
  control, not expected to learn held-out designed tasks it never saw.
- No benchmark is run if `warm_union` fails. If it passes, a single fresh quick@1,024
  event evaluates base, `blend`, `replay_refresh`, and `warm_union` through the trusted
  aggregate gateway.
- Promotion requires strict positive `warm_union-minus-base` deltas on aggregate and
  all ten public families, aggregate no worse than `blend`, and a positive increment
  over `replay_refresh` attributable to designed rows.
- Any replication, confirmation, or score-conditioned change is a successor experiment.

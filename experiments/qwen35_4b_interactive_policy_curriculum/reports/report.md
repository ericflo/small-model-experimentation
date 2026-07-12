# Interactive policy curriculum: final report

## Verdict

**NEGATIVE — full-sequence live-state DAgger failed its preregistered
mechanism gate.** Train-family macro terminal score fell 25.3 percentage
points and untouched-family macro fell 33.3 points. The gate cancelled
execution-reward RL, matched controls, and Menagerie before they consumed
compute or benchmark seeds.

This is a training-recipe failure, not evidence that the live-state labels
were wrong or that interactive consequence learning is impossible. The update
installed the common trace/closure surface while damaging the semantic pivot
policy that chooses when to probe, revise, verify, and commit.

## Reached Pipeline

1. Regenerated the C53 blend from all 2,240 committed rows. The frozen
   2,048-token window encoded 2,117 rows and skipped 123 exactly as receipted;
   333 optimizer steps completed in 3h23m.
2. Explicitly merged all 128 LoRA deltas into the composite Qwen checkpoint.
   The merged shard hash is `9450848e9d5d...`.
3. Collected 400 live incumbent trajectories over 200 fresh train-family
   episodes: one greedy and one sampled sibling per episode.
4. Built 2,270 DAgger rows: 1,386 unique model-visited corrections, 203
   expert-demo rows (12.8% of incremental data), and 681 stratified C53 replay
   rows. No transfer-family row entered incremental data or replay.
5. Trained 1.5 epochs from the merged incumbent. All 2,270 rows fit the
   4,096-token window; 213 optimizer steps completed in 2h11m. The DAgger
   merged shard hash is `842e3aa40e5e...`, distinct from the incumbent and
   receipt-verified back to the pinned official revision.
6. Evaluated both checkpoints greedily on the identical frozen vLLM protocol:
   480 process episodes and 560 atom-retention items per arm.

## Mechanism Gate

| Metric | Incumbent | DAgger | Delta | 95% paired CI | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| train-family episode macro | 0.6048 | 0.3517 | **-0.2531** | [-0.2954, -0.2103] | fail (`>=+0.08`) |
| transfer-family episode macro | 0.6850 | 0.3519 | **-0.3331** | [-0.3804, -0.2869] | fail (`>=0`) |
| train mean action-validity | — | — | -0.0567 | — | fail (`>=-0.03`) |
| transfer mean action-validity | — | — | -0.2524 | — | fail (`>=-0.03`) |
| train natural-close mean | — | — | +0.1033 | — | pass |
| transfer natural-close mean | — | — | +0.1293 | — | pass |
| atom family macro | 0.6926 | 0.6711 | -0.0215 | — | pass (`>=-0.03`) |
| atom parse macro | 1.0000 | 1.0000 | 0.0000 | — | pass |

Family episode deltas expose the damage:

| Family | Role | Delta |
| --- | --- | ---: |
| burrowmaze | trained | +0.0167 |
| ferrier | trained | -0.0220 |
| glyphgate | trained | **-0.6500** |
| kilnrite | trained | **-0.2658** |
| loomfix | trained | **-0.3444** |
| gatepost | unseen | -0.0500 |
| patchwheel | unseen | **-0.2944** |
| spindle | unseen | **-0.6550** |

## Failure Forensics

The clean guards localize the failure away from generic collapse:

- zero DAgger rows truncated;
- atom parsing remained 100%;
- atom macro stayed inside the retention bar;
- natural thinking closure improved materially;
- both merged checkpoint hashes changed and their receipt chains validate;
- state-aware experts pass every family/level selftest and malformed-state
  recovery tests.

The action distribution instead identifies semantic-pivot collapse. Across
all 2,270 targets, only 55 (2.4%) are `VERIFY`; visited-state data contains 19
verify rows among 1,386 (1.4%). Loomfix incremental supervision contains 316
`PATCH` versus 31 `RUN` actions. At frozen evaluation:

- incumbent loomfix emitted 459 `PATCH` and 70 `RUN`; DAgger emitted 600
  `PATCH` and zero `RUN`, scoring 0.000;
- incumbent patchwheel emitted 176 `RULE` and 372 `RUN`; DAgger emitted 599
  `RULE` and one `RUN`, scoring 0.000;
- DAgger kilnrite over-repeated locally invalid steps and fell 0.879 to 0.614;
- unseen spindle fell 0.977 to 0.322 despite retaining its `TAPE` surface.

The shared process trace therefore taught a fluent decision *format* without
preserving state-conditional decision boundaries. The update overlearned
`REVISE` and underlearned the `VERIFY` pivot, then transferred that bias to an
unseen repair family.

## Entropy And Varentropy Read

The collection contained 67/200 confident-failure groups, 25/200 groups with
nonzero terminal-score variance, and only 2/200 groups whose first coarse
semantic operator differed. With only two siblings, outcome varentropy is
mathematically degenerate whenever both outcome buckets have equal counts;
outcome variance remained informative. These measurements support routing
states, not scaling token loss or treating uncertainty as correctness.

## Stopping Decision

The one allowed interface-only repair was not invoked. The failure is not
target truncation, parser mismatch, invalid expert logic, or a bookkeeping
error. Rebalancing semantic operators, increasing replay, adding a behavior/KL
tether, lowering dose, or replacing sequence SFT with targeted pivot control
changes the training design and must receive a standalone preregistration and
fresh proxy seeds.

No RL collection, RL training, matched-SFT control, shuffled-reward control,
or Menagerie event ran. The benchmark firewall remained sealed.

## Durable Lesson

Classic DAgger's slogan—label the states the policy visits—is insufficient for
a shared small-model update. The curriculum must also preserve the *decision
operator distribution* and the incumbent policy outside corrected states.
For looping agents, verification/commit pivots are scarce but causal; broad
full-sequence imitation can erase them while every superficial metric
(closure, parse, atom retention) looks healthy.

The next credible warm start is locality-first: use live-state expert labels
to identify confident wrong semantic operators, then require a tethered
push-down/pull-up intervention to preserve neighboring logits and verify/commit
rates before any trajectory-reward stage. The active specialist-policy program
now owns the next integration test, so this mixed arm should not be rerun in
parallel.
